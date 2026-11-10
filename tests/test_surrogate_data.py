"""Training-data generator checks on the labeled SYNTHETIC grid only."""

import numpy as np
from hestonlab.calibration import LOWER, UPPER
from hestonlab.heston import MonteCarloConfig
from hestonlab.surrogate_data import generate, sample_parameters, synthetic_pilot_grid


def test_draws_stay_inside_calibration_bounds():
    p = sample_parameters(1024)
    assert p.shape == (1024, 5)
    assert (p >= LOWER).all() and (p <= UPPER).all()


def test_generator_shapes_and_common_random_numbers():
    grid = synthetic_pilot_grid(strikes=(70, 80, 90))
    config = MonteCarloConfig(paths=512)
    a = generate(grid, 8, config)
    assert a["prices"].shape == a["standard_errors"].shape == (8, len(grid))
    assert np.isfinite(a["prices"]).all() and (a["prices"] >= 0).all()
    # Same draws and same random numbers reproduce the labels exactly.
    np.testing.assert_array_equal(a["prices"], generate(grid, 8, config)["prices"])
    # Calls fall as strike rises within each averaging month.
    by_month = a["prices"].reshape(8, 3, 3)
    assert (np.diff(by_month, axis=2) <= 0).all()
