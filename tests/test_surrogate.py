"""Neural surrogate checks on the labeled SYNTHETIC grid only. Small and fast.
Skips if the optional `surrogate` extra (torch) is not installed."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hestonlab.calibration import LOWER, UPPER  # noqa: E402
from hestonlab.heston import HestonParameters, MonteCarloConfig  # noqa: E402
from hestonlab.surrogate import (  # noqa: E402
    InputScaler,
    OutputScaler,
    Surrogate,
    SurrogateMLP,
    TrainConfig,
    contract_fingerprint,
    evaluate,
    independent_benchmark,
    jacobian_check,
    learning_curve,
    subset,
    surface,
    train,
)
from hestonlab.surrogate_data import (  # noqa: E402
    generate,
    sample_parameters,
    synthetic_pilot_grid,
)

GRID = synthetic_pilot_grid(strikes=(70, 80, 90))
MC = MonteCarloConfig(paths=512)
FAST = TrainConfig(max_epochs=300, lr_patience=30, batch_size=16, learning_rate=1e-2)


@pytest.fixture(scope="module")
def tiny():
    train_data = generate(GRID, 256, MC, seed=0)
    test_data = generate(GRID, 64, MC, seed=1)
    model, history = train(train_data, GRID, FAST)
    return train_data, test_data, model, history


def untrained(n_outputs, seed=0):
    torch.manual_seed(seed)
    scaler = OutputScaler(np.arange(n_outputs, dtype=float), np.ones(n_outputs))
    return Surrogate(SurrogateMLP(n_outputs), scaler, [f"c{i}" for i in range(n_outputs)])


def test_input_scaler_maps_bounds_to_unit_box_and_back():
    scaler = InputScaler()
    np.testing.assert_allclose(scaler.to_unit(LOWER), np.zeros(5), atol=1e-12)
    np.testing.assert_allclose(scaler.to_unit(UPPER), np.ones(5), atol=1e-12)
    p = sample_parameters(64, seed=3)
    unit = scaler.to_unit(p)
    assert unit.min() >= 0 and unit.max() <= 1
    np.testing.assert_allclose(scaler.from_unit(unit), p, rtol=1e-12)
    # HestonParameters and dicts are accepted in PARAMETER_NAMES order.
    hp = HestonParameters()
    np.testing.assert_allclose(scaler.to_unit(hp), scaler.to_unit(vars(hp)))


def test_input_scaler_rejects_parameters_outside_training_box():
    bad = UPPER.copy()
    bad[1] = 20.0  # kappa above its upper bound
    with pytest.raises(ValueError, match="outside the calibration bounds"):
        InputScaler().to_unit(bad)


def test_output_scaler_round_trip_uses_given_data_only():
    prices = np.random.default_rng(0).normal(10, 3, size=(50, 4))
    scaler = OutputScaler.fit(prices)
    z = scaler.transform(prices)
    np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(scaler.inverse(z), prices)


@pytest.mark.parametrize("strikes", [(70, 80, 90), range(60, 101, 5)])
def test_output_size_follows_contract_set(strikes):
    grid = synthetic_pilot_grid(strikes=strikes)
    model = untrained(len(grid))
    assert model(HestonParameters()).shape == (len(grid),)
    assert model(sample_parameters(8)).shape == (8, len(grid))


def test_save_load_and_surface_agree(tmp_path):
    grid = synthetic_pilot_grid(strikes=(70, 80, 90))
    model = untrained(len(grid))
    model.metadata["contract_fingerprint"] = contract_fingerprint(grid)
    path = tmp_path / "model.pt"
    model.save(path)
    loaded = Surrogate.load(path)
    p = sample_parameters(4)
    np.testing.assert_array_equal(loaded(p), model(p))
    np.testing.assert_array_equal(surface(p, path), model(p))
    assert loaded.coordinate_ids == model.coordinate_ids
    loaded.check_contracts(grid)
    with pytest.raises(ValueError, match="different contract set"):
        loaded.check_contracts(grid[:-1])


def test_training_is_reproducible_and_beats_a_constant(tiny):
    train_data, test_data, model, history = tiny
    again, _ = train(train_data, GRID, FAST)
    p = test_data["parameters"]
    np.testing.assert_array_equal(again(p), model(p))
    report = evaluate(model, test_data)
    constant = np.sqrt(np.mean((test_data["prices"] - train_data["prices"].mean(0)) ** 2))
    assert report["rmse_usd_per_bbl"] < 0.3 * constant
    assert "not model error" in report["what"]
    assert model.metadata["training_points"] + model.metadata["validation_points"] == 256


def test_training_rejects_data_from_another_contract_set(tiny):
    train_data = tiny[0]
    with pytest.raises(ValueError, match="different contract set"):
        train(train_data, GRID[:-1], FAST)


def test_independent_benchmark_uses_a_different_stream(tiny):
    _, test_data, model, _ = tiny
    report = independent_benchmark(model, GRID, test_data, n_points=4, paths=1024)
    assert report["points"] == 4
    assert report["benchmark_monte_carlo"]["seed"] != MC.seed
    assert report["benchmark_monte_carlo"]["paths"] == 1024
    with pytest.raises(ValueError, match="different random stream"):
        independent_benchmark(model, GRID, test_data, n_points=4, paths=1024, seed=MC.seed)


def test_jacobian_has_one_row_per_contract_and_matches_finite_differences(tiny):
    model = tiny[2]
    report, arrays = jacobian_check(model, n_points=16, seed=5)
    assert arrays["singular_values"].shape == (16, 5)
    assert (np.diff(arrays["singular_values"], axis=1) <= 0).all()  # descending
    assert report["smallest_singular_value"]["min"] >= 0
    assert len(report["by_parameter_decile"]["sigma"]) == 10
    # Largest singular value = spectral norm of a finite-difference Jacobian.
    unit = model.input_scaler.to_unit(arrays["parameters"][0])
    h, jac = 1e-6, np.empty((len(GRID), 5))
    for j in range(5):
        step = np.eye(5)[j] * h
        up, down = (
            model.prices_from_unit(torch.as_tensor(unit + s, dtype=torch.float64)).detach().numpy()
            for s in (step, -step)
        )
        jac[:, j] = (up - down) / (2 * h)
    np.testing.assert_allclose(
        np.linalg.svd(jac, compute_uv=False), arrays["singular_values"][0], rtol=1e-5
    )


def test_learning_curve_trains_on_nested_prefixes(tiny):
    train_data, test_data, _, _ = tiny
    small = TrainConfig(max_epochs=5, batch_size=32)
    rows, models = learning_curve(train_data, test_data, GRID, (64, 128), small)
    assert [r["training_draws"] for r in rows] == [64, 128]
    assert set(models) == {64, 128}
    np.testing.assert_array_equal(
        subset(train_data, 64)["parameters"], train_data["parameters"][:64]
    )
    with pytest.raises(ValueError, match="only 256"):
        subset(train_data, 512)


def test_june_contract_sets_fail_clearly_without_private_root(tmp_path):
    from hestonlab.surrogate_pipeline import june_contract_sets

    with pytest.raises(FileNotFoundError, match="June contract sets unavailable.*Missing"):
        june_contract_sets(tmp_path / "no_such_root")


def test_june_monte_carlo_matches_calibrator_stream():
    from hestonlab.surrogate_pipeline import june_monte_carlo

    mc = june_monte_carlo("2024-06-14")
    assert mc.seed == MonteCarloConfig().seed + 20240614
    assert (mc.paths, mc.max_step_years) == (MonteCarloConfig().paths, 1 / 365)


def test_pipeline_end_to_end_writes_outputs_and_reuses_data(tmp_path):
    import json
    from hestonlab.surrogate_pipeline import PipelineConfig, run_pipeline

    config = PipelineConfig(
        train_sizes=(64, 128), test_draws=64, benchmark_points=4, benchmark_paths=1024,
        jacobian_points=32, train=TrainConfig(max_epochs=5, batch_size=32),
    )
    report = run_pipeline(GRID, tmp_path, MC, config, label="unit", log=lambda *_: None)
    for name in ("model.pt", "report.json", "jacobian.npz", "learning_curve.png",
                 "jacobian.png", "data/train.npz", "data/test.npz"):
        assert (tmp_path / name).is_file(), name
    assert json.loads((tmp_path / "report.json").read_text())["contracts"] == len(GRID)
    assert [r["training_draws"] for r in report["learning_curve"]] == [64, 128]
    np.testing.assert_array_equal(
        surface(HestonParameters(), tmp_path / "model.pt").shape, (len(GRID),)
    )
    stamp = (tmp_path / "data/train.npz").stat().st_mtime_ns
    messages = []
    run_pipeline(GRID, tmp_path, MC, config, label="unit", log=messages.append)
    assert (tmp_path / "data/train.npz").stat().st_mtime_ns == stamp
    assert any("reusing" in m for m in messages)


def test_pipeline_refuses_git_tracked_output_location():
    from pathlib import Path
    from hestonlab.surrogate_pipeline import check_output_location

    repo = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="tracked by git"):
        check_output_location(repo / "docs" / "surrogate_results")
    check_output_location(repo / "outputs" / "surrogate" / "anything")
