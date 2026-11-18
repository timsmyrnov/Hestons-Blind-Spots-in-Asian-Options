import numpy as np
import pytest

from hestonlab.weekly_heatmap import price_errors, color_scale, date_edges


def test_error_units_sign_and_original_reference_are_independent():
    arrays = {
        "heston_prices": np.array([.10, .20, .30]),
        "heston_standard_errors": np.array([.001, .002, .003]),
        "scale": np.array([80., 90., 100.]),
        "market_point": np.array([9., 17., 30.]),
        "raw_market": np.array([7., 20., 31.]),
    }
    heston, fitted, original, se, fitted_error, original_error = price_errors(arrays)
    np.testing.assert_allclose(heston, [8, 18, 30])
    np.testing.assert_allclose(fitted_error, [-1, 1, 0])
    np.testing.assert_allclose(original_error, [1, -2, -1])
    np.testing.assert_allclose(se, [.08, .18, .3])
    np.testing.assert_array_equal(fitted, arrays["market_point"])
    np.testing.assert_array_equal(original, arrays["raw_market"])


def test_shared_color_scale_preserves_zero_sign_and_extreme_values():
    data = {"fitted_error": np.array([-.37, .24]), "original_error": np.array([-2.12, 1.3])}
    norm, _, limit, _ = color_scale(data)
    assert limit == pytest.approx(2.2)
    assert norm(0) == pytest.approx(.5)
    values = np.array([-2.12, -.37, -.04, 0, .04, .37, 1.3])
    colors = norm(values)
    assert (np.diff(colors) > 0).all()
    assert colors.min() > 0 and colors.max() < 1
    np.testing.assert_allclose(norm.inverse(colors), values, atol=1e-12)
    np.testing.assert_allclose(norm(-values), 1-colors, atol=1e-12)


def test_week_boundaries_respect_calendar_spacing_and_endpoints():
    centers, edges = date_edges(["2022-01-07", "2022-01-14", "2022-01-28"])
    np.testing.assert_allclose(np.diff(centers), [7, 14])
    np.testing.assert_allclose(edges[1:-1], (centers[:-1]+centers[1:])/2)
    assert edges[0] < centers[0] and edges[-1] > centers[-1]
    for k, day in enumerate(centers):
        assert np.searchsorted(edges, day, side="right")-1 == k
    with pytest.raises(ValueError):
        date_edges(["2022-01-14", "2022-01-07"])
