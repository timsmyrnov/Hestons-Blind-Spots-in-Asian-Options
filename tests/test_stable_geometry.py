"""Display continuity must not alter gaps or silently promote unsupported prices."""

from dataclasses import replace
import numpy as np
import pytest
from hestonlab.market_manifold import _fit
from hestonlab.manifold_movie import smooth_snapshots, align_reference, vertical_ranges
from hestonlab.stable_geometry import (
    chart_frame,
    fixed_snapshots,
    market_jacobians,
    local_comparisons,
)
from hestonlab.weekly_store import Config


def fixture_history():
    rng = np.random.default_rng(123)
    z = rng.normal(size=(130, 2))
    return np.column_stack((z, 0.3 + 0.2 * z[:, 0] ** 2 - 0.1 * z[:, 1] ** 2))


def test_polynomial_derivatives_match_independent_finite_difference():
    model = _fit(fixture_history(), 2, 0.01)
    units = np.array([[0.1, -0.4], [0.7, 0.2]])
    expected = np.stack([model.jacobian(unit, step=1e-5) for unit in units])
    np.testing.assert_allclose(market_jacobians(model, units), expected, atol=1e-9)


@pytest.mark.parametrize("window", ["fixed", "local"])
def test_fixed_sheets_preserve_centers_gaps_and_support_warnings_after_smoothing(window):
    history = fixture_history()
    config = replace(Config(), market_rank=2)
    scale = np.full(3, np.sqrt(3))
    panel = {"vectors": history / scale}
    arrays, rows = [], []
    for i, (index, radius, gap) in enumerate(((120, 0.4, 0.1), (125, 0.15, 0.2))):
        model = _fit(history[index - config.history : index], 2, 0.01)
        # Second center is deliberately outside observed support; it must remain
        # visible and explicitly identified as a model continuation.
        center = np.zeros(2) if i == 0 else model.high + 0.05
        q = np.linspace(-radius, radius, 11)
        xy = np.array(np.meshgrid(q, q)).reshape(2, -1).T
        units = center + xy
        point = model.prices(center)
        base = point - np.array([0, 0, np.sqrt(3) * gap])
        patch = np.column_stack((xy, 0.05 * xy[:, 0] ** 2))
        arrays.append(
            {
                "scale": scale,
                "axes": np.eye(3)[:, :2],
                "market_point": point,
                "heston_prices": base / scale,
                "heston_patch": (base / scale + patch).reshape(11, 11, 3),
                "market_patch_units": units,
                "market_patch": model.prices(units).reshape(11, 11, 3),
            }
        )
        rows.append(
            {
                "date": f"2022-01-{7 + i * 7:02}",
                "panel_index": index,
                "market": {"ridge": 0.01},
                "market_projection_supported": i == 0,
                "gap_usd": gap,
                "flags": [] if i == 0 else ["price_noise"],
            }
        )
    snapshots, metadata = fixed_snapshots(arrays, rows, panel, config, grid_size=15, window=window)
    expected_floor = snapshots[0]["heston"][..., :2]
    for snapshot, row in zip(snapshots, rows):
        for name in ("heston", "market"):
            assert np.isfinite(snapshot[name]).all()
            np.testing.assert_array_equal(snapshot[name][..., :2], expected_floor)
            np.testing.assert_array_equal(snapshot[name + "_alpha"], 1)
        assert snapshot["heston"][7, 7, 2] == pytest.approx(0)
        assert snapshot["market"][7, 7, 2] == pytest.approx(row["gap_usd"])
        assert snapshot["lines"][0]["distance"] == row["gap_usd"]
        assert len(snapshot["lines"]) == 17
        assert sum(line["day"] for line in snapshot["lines"]) == 1
    assert "beyond observed support" in snapshots[1]["warning"]
    assert "Numerical checks flagged" in snapshots[1]["warning"]
    assert metadata["heston_pricing_calls"] == metadata["heston_calibration_calls"] == 0
    assert all(
        w["display_half_width_usd_per_bbl"] < w["heston_saved_patch_half_width"]
        for w in metadata["weeks"]
    )
    if window == "local":
        widths = [w["display_half_width_usd_per_bbl"] for w in metadata["weeks"]]
        assert widths[0] > 2 * widths[1]
        assert np.max(np.abs(expected_floor)) == 1
        assert not metadata["horizontal_dollar_distances_comparable_across_weeks"]
    for snapshot in smooth_snapshots(snapshots, [0, 5], 5):
        np.testing.assert_allclose(snapshot["heston"][..., :2], expected_floor)
        np.testing.assert_allclose(snapshot["market"][..., :2], expected_floor)
        assert snapshot["market"][7, 7, 2] == pytest.approx(snapshot["lines"][0]["distance"])
        assert "beyond observed support" in snapshot["warning"]
        for line in snapshot["lines"][1:]:
            # The local lines must stay attached to BOTH smoothed sheets.
            assert np.min(np.linalg.norm(snapshot["market"] - line["start"], axis=-1)) < 1e-12
            assert np.min(np.linalg.norm(snapshot["heston"] - line["end"], axis=-1)) < 1e-12
            assert not line["day"]
    # A changed cache/model must fail rather than silently drawing another model.
    arrays[0]["market_point"][0] += 0.1
    with pytest.raises(AssertionError):
        fixed_snapshots(arrays, rows, panel, config, grid_size=15)


def test_zero_gap_has_finite_orthonormal_display_frame():
    data = {"scale": np.ones(3), "market_point": np.ones(3), "heston_prices": np.ones(3)}
    frame = chart_frame(data, np.eye(3)[:, :2])
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)


def test_local_labels_include_hidden_price_directions():
    h = np.zeros((15, 15, 3))
    m = h.copy()
    m[..., 2] = 3
    hp = np.zeros((15, 15, 4))
    mp = hp.copy()
    mp[..., 2:] = [3, 4]
    lines = local_comparisons(h, m, hp, mp)
    assert len(lines) == 16
    for line in lines:
        assert line["distance"] == 5  # Full price RMS norm, not the visible height 3.
        assert np.linalg.norm(line["start"] - line["end"]) == 3


@pytest.mark.parametrize("curved", [False, True])
def test_common_alignment_preserves_curvature_separation_and_labels(curved):
    x, y = np.meshgrid(np.linspace(-1, 1, 15), np.linspace(-1, 1, 15))
    bend_h = (0.002 * x**2 - 0.003 * y**2) * curved
    bend_m = 0.005 * x**2 * curved
    h = np.stack((x, y, 2 * x + 3 * y + bend_h), axis=-1)
    m = np.stack((x, y, 0.1 + 2.01 * x + 3.02 * y + bend_m), axis=-1)
    picks = ((7, 7), (2, 3), (12, 11))
    lines = [
        {
            "start": m[i, j].copy(),
            "end": h[i, j].copy(),
            "distance": 0.1 if k == 0 else 99.0,
            "day": k == 0,
            "visible": 1.0,
        }
        for k, (i, j) in enumerate(picks)
    ]
    original = {"heston": h.copy(), "market": m.copy(), "lines": lines}
    got = align_reference([original])[0]
    np.testing.assert_allclose(got["heston"][..., 2], 0.1 + 0.01 * x + 0.02 * y - bend_h, atol=1e-13)
    np.testing.assert_allclose(
        got["market"][..., 2], -bend_m, atol=1e-13
    )
    assert got["market"][7, 7, 2] == pytest.approx(0)
    assert got["heston"][7, 7, 2] == pytest.approx(0.1)
    np.testing.assert_allclose(
        got["heston"][..., 2] - got["market"][..., 2], m[..., 2] - h[..., 2], atol=1e-13
    )
    np.testing.assert_array_equal(original["heston"], h)
    np.testing.assert_array_equal(original["market"], m)
    for k, (i, j) in enumerate(picks):
        assert got["lines"][k]["distance"] == lines[k]["distance"]
        np.testing.assert_allclose(got["lines"][k]["start"], got["market"][i, j])
        np.testing.assert_allclose(got["lines"][k]["end"], got["heston"][i, j])
        np.testing.assert_array_equal(original["lines"][k]["start"], m[i, j])


def test_axis_zoom_covers_surfaces_and_is_monotonic_in_each_half():
    heights = [0.05, 0.08, 0.16, 0.14, 0.12, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05]
    frames = [{"heston": np.full((3, 3, 3), h), "market": np.zeros((3, 3, 3))} for h in heights]
    bounds = np.array(vertical_ranges(frames))
    np.testing.assert_allclose(-bounds[:, 0] / (bounds[:, 1] - bounds[:, 0]), 0.15 / 1.15)
    assert (np.diff(bounds[:6, 1]) >= 0).all()
    assert (np.diff(bounds[5:, 1]) <= 0).all()
    assert bounds[5, 1] > 1.5 * bounds[-1, 1]
    for i in range(len(frames) - 1):
        for w in np.linspace(0, 1, 11):
            low, high = (1 - w) * bounds[i] + w * bounds[i + 1]
            top = (1 - w) * heights[i] + w * heights[i + 1]
            assert low < 0 < top < high


def test_tangent_departure_is_zero_for_plane_and_exact_for_quadratic():
    from hestonlab.movie_render import tangent_patches

    x, y = np.meshgrid(np.linspace(-1, 1, 41), np.linspace(-1, 1, 41))
    plane = np.stack((x, y, 2 + 3 * x - 4 * y), axis=-1)
    for patch in tangent_patches(plane):
        np.testing.assert_allclose(patch["departures"], 0, atol=1e-13)
    curved = plane.copy()
    curved[..., 2] += 0.02 * x**2 + 0.03 * y**2
    patches = tangent_patches(curved)
    for patch in patches:
        # Half-width .2 in each coordinate: .02*.2² + .03*.2² USD/bbl.
        assert max(np.abs(patch["departures"])) == pytest.approx(0.002)
        px, py, _ = patch["p"]
        assert patch["u"][2] / patch["u"][0] == pytest.approx(3 + 0.04 * px)
        assert patch["v"][2] / patch["v"][1] == pytest.approx(-4 + 0.06 * py)
    transformed = curved.copy()
    transformed[..., 2] = 0.7 + 2 * x + y - transformed[..., 2]
    for before, after in zip(patches, tangent_patches(transformed), strict=True):
        np.testing.assert_allclose(after["departures"], -np.array(before["departures"]), atol=1e-13)


def test_tangent_departures_interpolate_before_taking_absolute_maximum():
    from hestonlab.movie_render import tangent_patches

    x, y = np.meshgrid(np.linspace(-1, 1, 41), np.linspace(-1, 1, 41))
    a = np.stack((x, y, 0.02 * x**2), axis=-1)
    b = np.stack((x, y, -0.02 * x**2), axis=-1)
    for p, q, middle in zip(tangent_patches(a), tangent_patches(b), tangent_patches((a+b)/2), strict=True):
        departures = (np.array(p["departures"]) + q["departures"]) / 2
        np.testing.assert_allclose(departures, middle["departures"], atol=1e-13)
        assert max(np.abs(departures)) < 1e-13
