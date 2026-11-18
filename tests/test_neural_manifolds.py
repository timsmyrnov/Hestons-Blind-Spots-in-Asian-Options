from dataclasses import replace
import numpy as np
import pytest
import torch

from hestonlab.neural_embedding import ManifoldNetwork, geometry_diagnostics
from hestonlab.neural_training import NeuralSurface, fit_market
from hestonlab.neural_geometry import snapshot
from hestonlab.neural_weekly import NeuralConfig, stage_indices, cloud_units
from hestonlab.manifold_movie import align_reference


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2)
    torch.manual_seed(314)


def test_nontrivial_network_roundtrip_rank_and_checkpoint(tmp_path):
    network = ManifoldNetwork(21, width=16, depth=4)
    with torch.no_grad():
        for layer in network.layers:
            layer.conditioner[-1].weight.normal_(0, 0.02)
            layer.conditioner[-1].bias.normal_(0, 0.02)
    model = NeuralSurface(network, np.linspace(0.1, 0.3, 21), np.linspace(0.5, 2, 21))
    units = np.random.default_rng(2).uniform(0.1, 0.9, (12, 5))
    values = model.prices(units)
    assert values.shape == (12, 21)
    geometry = geometry_diagnostics(model, units)
    assert geometry["numerically_resolved"] and geometry["minimum_numerical_rank"] == 5
    assert geometry["inverse_max_abs_error"] < 1e-10
    path = tmp_path / "network.pt"
    model.save(path)
    restored = NeuralSurface.load(path)
    np.testing.assert_array_equal(restored.prices(units), values)
    np.testing.assert_allclose(model.initial_unit(values), units, atol=1e-10)
    unit = units[3]
    finite = np.column_stack(
        [
            (model.prices(unit + np.eye(5)[i] * 1e-5) - model.prices(unit - np.eye(5)[i] * 1e-5))
            / 2e-5
            for i in range(5)
        ]
    )
    np.testing.assert_allclose(model.jacobian(unit), finite, rtol=1e-7, atol=1e-9)


def test_exact_neural_sections_keep_hidden_coordinate_gap():
    h = NeuralSurface(ManifoldNetwork(21, width=8, depth=2), np.full(21, 0.5), np.ones(21))
    mean = np.full(21, 0.5)
    mean[-1] += 0.1 * np.sqrt(21)
    m = NeuralSurface(ManifoldNetwork(21, width=8, depth=2), mean, np.ones(21))
    unit = np.full(5, 0.5)
    arrays = dict(
        unit=unit,
        market_unit=unit,
        heston_prices=h.prices(unit),
        market_point=m.prices(unit),
        scale=np.ones(21),
        axes=np.eye(21)[:, :2],
    )
    row = dict(date="2022-01-07", gap_usd=0.1)
    item, meta = snapshot(h, m, arrays, row, np.eye(21)[:, :2], size=15)
    assert len(item["lines"]) == 17
    np.testing.assert_allclose(item["heston"][..., 2], 0, atol=1e-10)
    np.testing.assert_allclose(item["market"][..., 2], 0.1, atol=1e-10)
    for line in item["lines"]:
        assert line["distance"] == pytest.approx(0.1)
    aligned = align_reference([item])[0]
    np.testing.assert_allclose(aligned["market"][..., 2], 0, atol=1e-10)
    np.testing.assert_allclose(aligned["heston"][..., 2], 0.1, atol=1e-10)
    assert meta["native_price_dimension"] == 21


def test_market_is_trained_neural_embedding_in_all_price_coordinates():
    rng = np.random.default_rng(45)
    latent = rng.uniform(-1, 1, (120, 5))
    matrix = rng.normal(size=(5, 21)) * 0.01
    values = 0.05 + latent @ matrix + 0.002 * latent[:, 0, None] ** 2
    dates = [f"observation-{i}" for i in range(120)]
    model = fit_market(values, dates, replace(NeuralConfig(), market_epochs=400, width=16, depth=2))
    assert model.metadata["method"] == "neural_injective_embedding"
    assert model.metadata["native_price_dimension"] == 21
    assert model.metadata["training_dates"] == dates
    assert model.prices(model.train_units).shape == (120, 21)
    assert any(
        torch.count_nonzero(layer.conditioner[-1].weight) > 0 for layer in model.network.layers
    )
    assert geometry_diagnostics(model, model.train_units[::20])["numerically_resolved"]


def test_half_boundary_and_continuation_require_completed_half():
    first = list(stage_indices(209, "first-half"))
    assert first == list(range(105))
    with pytest.raises(ValueError, match="first half"):
        stage_indices(209, "remaining", completed=104)
    assert list(stage_indices(209, "remaining", completed=105))[105:] == list(range(105, 209))
    with pytest.raises(ValueError, match="inside the first half"):
        stage_indices(209, "first-half", limit=106)


def test_train_validation_test_parameter_streams_are_distinct():
    anchors = np.full((8, 5), 0.5)
    samples = [cloud_units(16, 8, anchors, seed) for seed in (1, 100001, 200001)]
    assert all(x.shape == (24, 5) and ((0 <= x) & (x <= 1)).all() for x in samples)
    assert len({tuple(row) for x in samples for row in x}) == 72


def test_training_resume_after_last_epoch_reproduces_frozen_model(tmp_path):
    from hestonlab.neural_training import train_heston
    from hestonlab.weekly_store import write_npz

    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(5, 21)) * .01
    for name, count in (("train", 32), ("validation", 16)):
        units = rng.uniform(.1, .9, (count, 5))
        write_npz(tmp_path / (name + ".npz"), units=units,
                  prices=.1 + units @ matrix, standard_errors=np.zeros((count, 21)))
    config = replace(NeuralConfig(), heston_epochs=2, batch_size=16, width=8, depth=2)
    model = train_heston(tmp_path, config)
    points = rng.uniform(.1, .9, (8, 5))
    expected = model.prices(points)
    (tmp_path / "model.pt").unlink()  # Simulate interruption after final training-state save.
    resumed = train_heston(tmp_path, config)
    np.testing.assert_array_equal(resumed.prices(points), expected)
    assert resumed.metadata["epochs"] == 2
    with (tmp_path / "train.npz").open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match="cloud differ"):
        train_heston(tmp_path, config)
