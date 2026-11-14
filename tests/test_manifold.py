"""Structural geometry, price fidelity and fitting checks; all inputs are synthetic."""

from dataclasses import asdict, replace
import json
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hestonlab.calibration import CalibrationSnapshot  # noqa: E402
from hestonlab.heston import HestonParameters, MonteCarloConfig, price_asian_options  # noqa: E402
from hestonlab.manifold import ManifoldNetwork, geometry_diagnostics  # noqa: E402
from hestonlab.manifold_calibration import (  # noqa: E402
    SCALE_KEY, fit_manifold, fit_per_date, main, residual_geometry,
)
from hestonlab.surrogate import (  # noqa: E402
    DTYPE, PRICE_TICK, InputScaler, OutputScaler, Surrogate, SurrogateMLP, TrainConfig,
    contract_fingerprint, price_error_scale, train,
)
from hestonlab.surrogate_data import generate, synthetic_pilot_grid  # noqa: E402


GRID = synthetic_pilot_grid(strikes=(70, 80, 90))
MC = MonteCarloConfig(paths=128, max_step_years=1 / 52)


def affine_model(contracts=GRID):
    # Known embedded plane in actual price coordinates; all labels positive.
    return Surrogate(
        ManifoldNetwork(len(contracts), width=12, depth=2),
        OutputScaler(np.full(len(contracts), 20.0), np.ones(len(contracts))),
        [c.coordinate_id for c in contracts],
        {"contract_fingerprint": contract_fingerprint(contracts),
         "monte_carlo": str(asdict(MC)),
         SCALE_KEY: [1.0] * len(contracts)},  # unit scales: weighted == USD
    )


@pytest.mark.parametrize("dimension", [5, 9, 27])
def test_nontrivial_flow_is_invertible_and_has_full_rank(dimension):
    torch.manual_seed(14)
    model = ManifoldNetwork(dimension, width=12, depth=4)
    # Exercise nonidentity transformations, including after an optimizer update.
    for layer in model.layers:
        torch.nn.init.normal_(layer.conditioner[-1].weight, std=0.08)
        torch.nn.init.normal_(layer.conditioner[-1].bias, std=0.08)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    units = torch.rand(8, 5, dtype=DTYPE)
    model(units).square().mean().backward()
    optimizer.step()
    ambient = torch.randn(8, dimension, dtype=DTYPE)
    np.testing.assert_allclose(model.ambient_inverse(model.ambient_forward(ambient)).detach(),
                               ambient, atol=1e-11, rtol=1e-11)
    inverse = model.ambient_inverse(model(units)).detach()
    np.testing.assert_allclose(inverse[:, :5] + 0.5, units, atol=1e-11)
    np.testing.assert_allclose(inverse[:, 5:], 0, atol=1e-11)
    jac = torch.func.jacrev(model)(units[0]).detach().numpy()
    assert np.linalg.matrix_rank(jac) == 5
    h = 1e-6
    difference = np.column_stack([
        ((model(units[0] + h * direction) - model(units[0] - h * direction)) / (2 * h))
        .detach().numpy() for direction in torch.eye(5, dtype=DTYPE)
    ])
    np.testing.assert_allclose(jac, difference, rtol=1e-5, atol=1e-8)


def test_rank_deficient_linear_initialization_still_embeds_and_reports_conditioning():
    model = affine_model()
    units = np.random.default_rng(0).uniform(size=(32, 5))
    # Rank-one labels do not make the initialized ambient affine map singular.
    model.network.initialize_from_data(units, np.tile(units[:, :1], (1, len(GRID))))
    out = geometry_diagnostics(model, np.vstack((units, np.zeros(5), np.ones(5))))
    assert out["numerically_resolved"] and out["minimum_numerical_rank"] == 5
    assert out["inverse_max_abs_error"] < 1e-8
    assert "learned price map only" in out["scope"]
    with torch.no_grad():
        model.network.affine.zero_()
    with pytest.raises(ValueError, match="Singular"):
        geometry_diagnostics(model, units)


def test_rejects_invalid_embedding_dimension_and_scalers():
    with pytest.raises(ValueError, match="at least 5"):
        ManifoldNetwork(4)
    with pytest.raises(ValueError, match="at least 3"):
        ManifoldNetwork(2, latent_dim=3)
    with pytest.raises(ValueError, match="strictly positive"):
        OutputScaler([0, 0], [1, 0])
    with pytest.raises(ValueError, match="finite"):
        InputScaler().to_unit([np.nan, 1, 0.04, 0.5, 0])


def test_manifold_save_load_and_legacy_checkpoint(tmp_path):
    model = affine_model()
    path = tmp_path / "model.pt"
    model.save(path)
    loaded = Surrogate.load(path)
    assert isinstance(loaded.network, ManifoldNetwork)
    np.testing.assert_array_equal(loaded(HestonParameters()), model(HestonParameters()))
    assert geometry_diagnostics(loaded, np.full((1, 5), 0.5))["numerically_resolved"]
    legacy = Surrogate(SurrogateMLP(9, 12, 2), model.output_scaler, model.coordinate_ids)
    legacy.save(path)
    blob = torch.load(path, weights_only=True)
    blob["format_version"] = 1
    del blob["architecture"]
    torch.save(blob, path)
    old = Surrogate.load(path)
    np.testing.assert_array_equal(old(HestonParameters()), legacy(HestonParameters()))
    assert geometry_diagnostics(old, np.zeros((1, 5)))["guarantee"].startswith("none")


def test_fit_recovers_known_coordinates_and_normal_residual_across_dates():
    model = affine_model()
    expected_unit = np.array([0.2, 0.3, 0.4, 0.6, 0.7])
    expected = model.input_scaler.from_unit(expected_unit)
    labels = model(expected)
    labels[-1] += 0.25  # exactly normal to this known plane
    first = CalibrationSnapshot("2024-06-14", GRID, tuple(labels))
    second = replace(first, observation_date="2024-06-17")
    result = fit_manifold([first, second], {s.observation_date: model for s in [first, second]},
                          starts=3, verification_paths=128)
    np.testing.assert_allclose(list(result["parameters"].values()), expected, rtol=1e-7)
    assert result["manifold_fit_rmse"] == pytest.approx(0.25 / 3)
    assert result["residual_geometry"]["tangent_rmse"] < 1e-8
    assert result["residual_geometry"]["normal_rmse"] == pytest.approx(0.25 / 3)
    assert result["geometry_numerically_resolved_at_fit"]
    assert result["optimizer"]["success"]
    assert not result["optimizer"]["global_minimum_certified"]
    assert not result["fully_validated"] and not result["july_labels_used"]
    assert len(result["rows"]) == 18
    for configs in result["verification_monte_carlo"].values():
        assert len({c["seed"] for c in configs} - {MC.seed}) == 2
    # Report is portable JSON, including solver status and geometry.
    json.dumps(result, allow_nan=False)


def test_boundary_fit_is_identified_and_normal_decomposition_is_orthogonal():
    model = affine_model()
    unit = np.array([1.0, 0.3, 0.4, 0.6, 0.7])
    labels = model(model.input_scaler.from_unit(unit))
    labels[0] += 0.5  # lies beyond the bounded manifold along its tangent
    snapshot = CalibrationSnapshot("2024-06-14", GRID, tuple(labels))
    result = fit_manifold([snapshot], {snapshot.observation_date: model}, starts=1,
                          verification_paths=128)
    assert "v0" in result["boundary_parameters"]
    assert result["residual_geometry"]["tangent_rmse"] > 0.1
    assert result["residual_geometry"]["normal_rmse"] < 1e-10
    rng = np.random.default_rng(2)
    jac, residual = rng.normal(size=(12, 5)), rng.normal(size=12)
    split = residual_geometry(jac, residual)
    tangent, normal = np.array(split["tangent_residual"]), np.array(split["normal_residual"])
    np.testing.assert_allclose(tangent + normal, residual)
    np.testing.assert_allclose(jac.T @ normal, 0, atol=1e-12)
    assert np.dot(tangent, normal) == pytest.approx(0, abs=1e-12)


def test_fitting_rejects_legacy_misaligned_and_unresolved_models():
    model = affine_model()
    s = CalibrationSnapshot("2024-06-14", GRID, tuple(model(HestonParameters())))
    with pytest.raises(ValueError, match="matching model"):
        fit_manifold([s], {})
    legacy = Surrogate(SurrogateMLP(9), model.output_scaler, model.coordinate_ids, model.metadata)
    with pytest.raises(ValueError, match="retrain legacy"):
        fit_manifold([s], {s.observation_date: legacy})
    wrong = replace(s, contracts=tuple(reversed(GRID)))
    with pytest.raises(ValueError, match="different contract set"):
        fit_manifold([wrong], {s.observation_date: model})
    model.network.affine[-1, -1] = 0
    with pytest.raises(ValueError, match="Singular"):
        fit_manifold([s], {s.observation_date: model})
    with pytest.raises(ValueError, match="July is held out"):
        replace(s, observation_date="2024-07-01")


def test_train_then_fit_independent_monte_carlo_prices():
    data = generate(GRID, 128, MC, seed=0)
    model, _ = train(data, GRID, TrainConfig(width=24, depth=4, max_epochs=100,
                                            batch_size=32, learning_rate=0.005))
    model.metadata[SCALE_KEY] = price_error_scale(model, data).tolist()
    truth = HestonParameters(v0=0.16, theta=0.12)
    # Targets are from a stream used by neither training nor verification.
    labels = tuple(e.price for e in price_asian_options(GRID, truth, replace(MC, seed=712)))
    snapshot = CalibrationSnapshot("2024-06-14", GRID, labels)
    result = fit_manifold([snapshot], {snapshot.observation_date: model}, starts=2,
                          verification_paths=256, max_evaluations=100)
    assert result["manifold_fit_rmse"] < result["initial_rmse"]
    assert result["geometry_numerically_resolved_at_fit"]
    assert np.isfinite(result["independent_mc_market_rmse"])
    # Fidelity is measured, not inferred from the existence of the manifold.
    assert result["surrogate_vs_independent_mc_rmse"] >= 0
    # The Heston control is never further away than the surrogate's own error at the fit.
    rows = result["rows"]
    gap = np.sqrt(np.mean([((r["manifold_price"] - r["independent_mc_price"])
                            / r["error_scale_usd_per_bbl"]) ** 2 for r in rows]))
    control = result["heston_control"]
    assert 0 < control["weighted_distance"] <= gap * (1 + 1e-6)
    assert result["market_to_control_ratio"] == pytest.approx(
        result["market_weighted_distance"] / control["weighted_distance"])
    json.dumps(result, allow_nan=False)


def test_per_date_fit_gives_each_date_its_own_parameters():
    model = affine_model()
    units = {"2024-06-14": np.array([0.2, 0.3, 0.4, 0.6, 0.7]),
             "2024-06-17": np.array([0.7, 0.6, 0.5, 0.3, 0.2])}
    snapshots = [CalibrationSnapshot(d, GRID, tuple(model(model.input_scaler.from_unit(u))))
                 for d, u in units.items()]
    models = {d: model for d in units}
    result = fit_per_date(snapshots, models, starts=3, verification_paths=128)
    assert result["mode"] == "per_date" and result["training_dates"] == list(units)
    for day, unit in units.items():
        report = result["dates"][day]
        assert report["mode"] == "single_date" and report["training_dates"] == [day]
        np.testing.assert_allclose(list(report["parameters"].values()),
                                   model.input_scaler.from_unit(unit), rtol=1e-7)
        assert report["manifold_fit_rmse"] < 1e-8
    assert [r["observation_date"] for r in result["summary"]] == list(units)
    # One shared vector cannot hit two different points on the surface.
    pooled = fit_manifold(snapshots, models, starts=3, verification_paths=128)
    assert pooled["mode"] == "pooled" and pooled["manifold_fit_rmse"] > 0.05
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="matching model"):
        fit_per_date(snapshots, {"2024-06-14": model})


def test_noise_weighting_divides_residuals_by_error_scale():
    model = affine_model()
    model.metadata[SCALE_KEY] = [1.0] * 8 + [2.0]
    unit = np.array([0.2, 0.3, 0.4, 0.6, 0.7])
    labels = model(model.input_scaler.from_unit(unit))
    labels[-1] += 0.5  # normal to the plane, on the contract with scale 2
    snapshot = CalibrationSnapshot("2024-06-14", GRID, tuple(labels))
    models = {snapshot.observation_date: model}
    result = fit_manifold([snapshot], models, starts=1, verification_paths=128)
    assert result["weighting"] == "noise"
    assert result["rows"][-1]["weighted_residual"] == pytest.approx(0.25)
    assert result["market_weighted_distance"] == pytest.approx(0.25 / 3)
    assert result["manifold_fit_rmse"] == pytest.approx(0.5 / 3)
    usd = fit_manifold([snapshot], models, starts=1, verification_paths=128, weighting="usd")
    assert usd["market_weighted_distance"] == pytest.approx(0.5 / 3)
    del model.metadata[SCALE_KEY]
    with pytest.raises(ValueError, match="error scales"):
        fit_manifold([snapshot], models, starts=1, verification_paths=128)
    assert fit_manifold([snapshot], models, starts=1, verification_paths=128,
                        weighting="usd")["weighting"] == "usd"
    with pytest.raises(ValueError, match="weighting"):
        fit_manifold([snapshot], models, weighting="vega")


def test_price_error_scale_combines_surrogate_and_mc_error_with_tick_floor():
    model = affine_model()
    params = model.input_scaler.from_unit(np.random.default_rng(3).uniform(size=(16, 5)))
    prices = model(params)
    prices[:, 0] += 0.3
    errors = np.zeros_like(prices)
    errors[:, 0] = 0.4
    data = {"coordinate_ids": np.array(model.coordinate_ids), "parameters": params,
            "prices": prices, "standard_errors": errors}
    scale = price_error_scale(model, data)
    assert scale[0] == pytest.approx(0.5)  # sqrt(0.3^2 + 0.4^2)
    np.testing.assert_allclose(scale[1:], PRICE_TICK)
    with pytest.raises(ValueError, match="different contract set"):
        price_error_scale(model, {**data, "coordinate_ids": np.array(model.coordinate_ids[::-1])})


def test_saved_model_fitting_cli_and_existing_output_error(tmp_path, monkeypatch, capsys):
    model = affine_model()
    snapshot = CalibrationSnapshot("2024-06-14", GRID, tuple(model(HestonParameters())))
    model.save(tmp_path / "models" / snapshot.observation_date / "model.pt")
    monkeypatch.setattr("hestonlab.manifold_calibration.load_june_training", lambda _: [snapshot])
    arguments = ["--root", str(tmp_path / "archive"), "--models", str(tmp_path / "models"),
                 "--out", str(tmp_path / "fit.json"), "--starts", "1",
                 "--verification-paths", "128"]
    report = main(arguments)
    assert json.loads((tmp_path / "fit.json").read_text())["model_sha256"] == report["model_sha256"]
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2 and "fresh fit output" in capsys.readouterr().err


def test_june_cli_trains_manifold_before_fitting(tmp_path, monkeypatch):
    from hestonlab import surrogate_pipeline as pipeline

    labels = tuple(e.price for e in price_asian_options(GRID, HestonParameters(), MC))
    snapshot = CalibrationSnapshot("2024-06-14", GRID, labels)
    monkeypatch.setattr(pipeline, "load_june_training", lambda _: [snapshot])
    monkeypatch.setattr(pipeline, "june_monte_carlo", lambda _: MC)
    monkeypatch.setattr(pipeline, "QUICK", pipeline.PipelineConfig(
        train_sizes=(64,), test_draws=32, benchmark_points=4, benchmark_paths=128,
        jacobian_points=32, train=TrainConfig(width=12, depth=2, max_epochs=5),
    ))
    reports = pipeline.main(["june", "--root", str(tmp_path / "archive"),
                             "--out", str(tmp_path / "experiment"), "--quick", "--fit",
                             "--starts", "1"])
    day = snapshot.observation_date
    assert reports[day]["architecture"] == "manifold"
    assert reports[day]["geometry"]["numerically_resolved"]
    assert len(reports[day]["price_error_scale_usd_per_bbl"]) == len(GRID)
    assert reports["fit"]["mode"] == "per_date"
    assert reports["fit"]["training_dates"] == [day]
    assert reports["fit"]["dates"][day]["weighting"] == "noise"
    assert reports["fit"]["summary"][0]["heston_control_weighted_distance"] > 0
    assert (tmp_path / "experiment" / "fit.json").exists()
