"""Custom parameter boxes and parallel Monte Carlo data. SYNTHETIC grid only."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hestonlab.calibration import LOWER, UPPER  # noqa: E402
from hestonlab.heston import MonteCarloConfig  # noqa: E402
from hestonlab.manifold import ManifoldNetwork  # noqa: E402
from hestonlab.surrogate import Surrogate, TrainConfig, train  # noqa: E402
from hestonlab.surrogate_box import (  # noqa: E402
    VALUE_EXPERIMENT_BOX,
    ParameterBox,
    cached_generate_in_box,
    check_data_box,
    generate_in_box,
    load_in_box,
    price_points_parallel,
    train_in_box,
)
from hestonlab.surrogate_data import sample_parameters, synthetic_pilot_grid  # noqa: E402

GRID = synthetic_pilot_grid(strikes=(70, 80, 90))
MC = MonteCarloConfig(paths=512)
BOX = VALUE_EXPERIMENT_BOX
FAST = TrainConfig(max_epochs=20, batch_size=32)


def test_calibration_box_reproduces_existing_sampler_exactly():
    box = ParameterBox.calibration()
    np.testing.assert_array_equal(box.sample(64, seed=4), sample_parameters(64, seed=4))
    np.testing.assert_array_equal(box.bounds[0], LOWER)
    np.testing.assert_array_equal(box.bounds[1], UPPER)


def test_experiment_box_samples_and_scales_inside_its_bounds():
    p = BOX.sample(256, seed=1)
    assert BOX.contains(p)
    assert p[:, 0].max() <= 0.36 and p[:, 4].max() <= 0.5
    unit = BOX.input_scaler().to_unit(p)
    assert unit.min() >= 0 and unit.max() <= 1
    with pytest.raises(ValueError, match="outside"):
        BOX.input_scaler().to_unit(np.array([0.9, 1.0, 0.04, 0.5, 0.0]))  # v0 too high


@pytest.mark.parametrize(
    "lower, upper",
    [
        ((0.01, 0.1, 0.01, 0.05), (0.36, 5, 0.36, 1.5)),  # only 4 bounds
        ((0.36, 0.1, 0.01, 0.05, -0.9), (0.01, 5, 0.36, 1.5, 0.5)),  # lower > upper
        ((0.0, 0.1, 0.01, 0.05, -0.9), (0.36, 5, 0.36, 1.5, 0.5)),  # v0 lower bound 0
        ((0.01, 0.1, 0.01, 0.05, -1.0), (0.36, 5, 0.36, 1.5, 0.5)),  # rho reaches -1
    ],
)
def test_invalid_boxes_are_rejected(lower, upper):
    with pytest.raises(ValueError):
        ParameterBox(lower, upper)


def test_parallel_pricing_is_identical_to_serial():
    p = BOX.sample(8, seed=2)
    serial, serial_se, _ = price_points_parallel(GRID, p, MC, workers=1)
    parallel, parallel_se, timing = price_points_parallel(GRID, p, MC, workers=2, chunk_size=2)
    np.testing.assert_array_equal(parallel, serial)
    np.testing.assert_array_equal(parallel_se, serial_se)
    assert timing["workers"] == 2 and timing["core_seconds"] > 0


def test_cache_reuses_matching_data_and_regenerates_for_another_box(tmp_path):
    path = tmp_path / "train.npz"
    log = []
    first = cached_generate_in_box(path, GRID, 16, MC, 0, BOX, workers=1, log=log.append)
    again = cached_generate_in_box(path, GRID, 16, MC, 0, BOX, workers=1, log=log.append)
    assert "reusing" in log[-1]
    np.testing.assert_array_equal(first["prices"], again["prices"])
    other = cached_generate_in_box(
        path, GRID, 16, MC, 0, ParameterBox.calibration(), workers=1, log=log.append
    )
    assert "generated" in log[-1]
    assert ParameterBox.calibration().matches(other["bounds_lower"], other["bounds_upper"])


def test_box_travels_with_the_model_and_mismatches_are_refused(tmp_path):
    data = generate_in_box(GRID, 64, MC, sobol_seed=0, box=BOX, workers=1)
    model, _ = train_in_box(data, GRID, BOX, FAST)
    assert isinstance(model.network, ManifoldNetwork)  # default architecture
    path = tmp_path / "model.pt"
    model.save(path)
    loaded = load_in_box(path, BOX)
    assert isinstance(loaded.network, ManifoldNetwork)
    assert model.metadata["parameter_box"]["name"] == BOX.name
    p = BOX.sample(4, seed=9)
    np.testing.assert_array_equal(loaded(p), model(p))
    np.testing.assert_array_equal(loaded.input_scaler.lower, BOX.lower)
    # The default loader (and so surface()) still expects the calibration box.
    with pytest.raises(ValueError, match="different calibration bounds"):
        Surrogate.load(path)
    with pytest.raises(ValueError, match="different parameter bounds"):
        load_in_box(path, ParameterBox.calibration())
    # Training on box data with another box is refused.
    with pytest.raises(ValueError, match="different parameter box"):
        check_data_box(data, ParameterBox.calibration())
    with pytest.raises(ValueError, match="carries no box"):
        check_data_box({"parameters": p}, BOX)


def test_default_training_still_uses_the_calibration_box(tmp_path):
    data = generate_in_box(GRID, 32, MC, sobol_seed=0, box=ParameterBox.calibration(),
                           workers=1)
    model, _ = train(data, GRID, FAST)
    np.testing.assert_array_equal(model.input_scaler.lower, LOWER)
    model.save(tmp_path / "m.pt")
    Surrogate.load(tmp_path / "m.pt")  # default loader accepts it
