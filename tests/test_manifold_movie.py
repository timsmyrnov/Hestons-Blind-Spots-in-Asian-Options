"""Per-day slice geometry and movie pages on known synthetic manifolds."""

import json
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from hestonlab import manifold_movie as movie  # noqa: E402
from hestonlab import market_manifold as mm  # noqa: E402
from hestonlab.heston import AsianContract  # noqa: E402
from hestonlab.manifold import ManifoldNetwork  # noqa: E402
from hestonlab.surrogate import OutputScaler, Surrogate  # noqa: E402

N = 8


def plane_heston():
    return Surrogate(ManifoldNetwork(N, width=8, depth=2),
                     OutputScaler(np.full(N, 20.0), np.ones(N)), [f"c{i}" for i in range(N)])


def lifted_market(lift):
    """The Heston plane's first 4 directions, lifted by `lift` along coordinate 7."""
    network = ManifoldNetwork(N, width=8, depth=2, latent_dim=4)
    offset = np.full(N, 20.0)
    offset[7] += lift
    return mm.MarketManifold(network, OutputScaler(offset, np.ones(N)),
                             [f"c{i}" for i in range(N)], np.full((3, 4), 0.5))


def test_day_frame_puts_the_gap_straight_up_over_one_floor_square():
    heston, market = plane_heston(), lifted_market(0.8)
    cloud = movie.heston_cloud(heston, 64)
    piece = movie.day_slice(heston, market, np.ones(N), np.full(4, 0.5), 1, extent=0.05,
                            n_grid=9, n_lines=6, cloud=cloud, starts=2)
    expected = 0.8 / np.sqrt(N)  # RMS distance over the grid
    assert piece["distance"] == pytest.approx(expected, abs=1e-8)
    day = piece["lines"][0]
    assert day["day"]
    np.testing.assert_allclose(day["end"], 0, atol=1e-8)
    np.testing.assert_allclose(day["start"], [0, 0, expected], atol=1e-8)
    # Parallel planes: two flat sheets the gap apart, over the same floor square.
    assert piece["heston"].shape == piece["market"].shape == (9, 9, 3)
    np.testing.assert_allclose(piece["heston"][..., 2], 0, atol=1e-8)
    np.testing.assert_allclose(piece["market"][..., 2], expected, atol=1e-8)
    np.testing.assert_allclose(piece["heston"][0, :, 0], np.linspace(-0.05, 0.05, 9), atol=1e-4)
    np.testing.assert_allclose(piece["heston"][..., :2], piece["market"][..., :2], atol=1e-4)
    assert len(piece["lines"]) == 7
    for line in piece["lines"]:  # straight down, every one the true gap
        assert line["distance"] == pytest.approx(expected, abs=1e-6)
        np.testing.assert_allclose(line["start"][:2], line["end"][:2], atol=1e-4)
        assert line["start"][2] - line["end"][2] == pytest.approx(expected, abs=1e-6)
    assert set(piece["heston_directions"]) == {"direction 1", "direction 2"}
    two = movie.day_slice(heston, market, np.ones(N), np.full(4, 0.5), 2, extent=0.05,
                          n_grid=5, n_lines=3, cloud=cloud, starts=1)
    assert set(two["heston_directions"]) == {"direction 3", "direction 4"}
    with pytest.raises(ValueError, match="two directions"):
        movie.day_slice(heston, market, np.ones(N), np.full(4, 0.5), 3, cloud=cloud)


def test_day_weights_rescale_distances_into_that_days_dollars():
    heston, market = plane_heston(), lifted_market(0.8)
    weight = np.full(N, 2.5)  # e.g. discount x forward of 2.5 on every coordinate
    piece = movie.day_slice(heston, market, weight, np.full(4, 0.5), 1, extent=0.1, n_grid=5,
                            n_lines=3, cloud=movie.heston_cloud(heston, 64))
    assert piece["distance"] == pytest.approx(2.5 * 0.8 / np.sqrt(N), abs=1e-8)
    assert len(piece["lines"]) == 4 and piece["heston"].shape == (5, 5, 3)


def test_direction_pairs_follow_the_previous_day():
    a, b = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    assert movie._align([a, b], None) == ([0, 1], [1.0, 1.0])
    assert movie._align([-a, b], [a, b]) == ([0, 1], [-1.0, 1.0])
    assert movie._align([b, a], [a, b]) == ([1, 0], [1.0, 1.0])


def test_next_day_keeps_the_floor_and_lines_in_place():
    heston = plane_heston()
    cloud = movie.heston_cloud(heston, 64)
    first = movie.day_slice(heston, lifted_market(0.8), np.ones(N), np.full(4, 0.5), 1,
                            extent=0.05, n_grid=5, n_lines=4, cloud=cloud)
    second = movie.day_slice(heston, lifted_market(1.6), np.ones(N), np.full(4, 0.5), 1,
                             extent=0.05, n_grid=5, n_lines=4, cloud=cloud, previous=first)
    assert len(first["lines"]) == len(second["lines"]) == 5
    gaps = [p["lines"][0]["start"][2] for p in (first, second)]
    np.testing.assert_allclose(gaps, [0.8 / np.sqrt(N), 1.6 / np.sqrt(N)], atol=1e-8)
    np.testing.assert_allclose(first["heston"], second["heston"], atol=1e-8)
    for a, b in zip(first["lines"], second["lines"]):  # same floor points: only heights move
        np.testing.assert_allclose(a["start"][:2], b["start"][:2], atol=1e-4)


def test_movie_axes_are_ranked_by_strength_and_fixed_on_every_day():
    e = np.eye(N)
    days = [{"tangents": [3 * e[0], e[1]]}, {"tangents": [-e[1], 2.5 * e[0]]}]
    axes = movie.movie_axes(days)
    np.testing.assert_allclose(axes[0], e[0], atol=1e-12)  # strongest, signed like day 1
    np.testing.assert_allclose(np.abs(axes[1]), e[1], atol=1e-12)
    heston, market = plane_heston(), lifted_market(0.8)
    setup = movie.day_setup(heston, market, np.ones(N), np.full(4, 0.5), 1,
                            movie.heston_cloud(heston, 64), starts=2)
    turned = movie.set_axes(setup, [setup["tangents"][1] / np.linalg.norm(setup["tangents"][1]),
                                    setup["tangents"][0] / np.linalg.norm(setup["tangents"][0])])
    np.testing.assert_allclose(turned["frame"][0], setup["frame"][1], atol=1e-8)
    np.testing.assert_allclose(turned["axis_reach"], 1, atol=1e-8)


def test_room_keeps_the_heston_sheet_inside_its_parameter_box():
    heston, market = plane_heston(), lifted_market(0.8)
    cloud = movie.heston_cloud(heston, 64)
    setup = movie.day_setup(heston, market, np.ones(N), np.full(4, 0.5), 1, cloud, starts=2)
    room = movie.room(heston, setup, 100.0)
    assert 0 < room < 100
    inside = movie.day_sheets(heston, market, setup, room, n_grid=7, n_lines=2, cloud=cloud)
    assert np.isfinite(inside["heston"]).all()
    wider = movie.day_sheets(heston, market, setup, 1.2 * room, n_grid=7, n_lines=2, cloud=cloud)
    assert np.isnan(wider["heston"]).any()
    assert movie.room(heston, setup, 0.5 * room) == 0.5 * room


def test_grid_scale_is_discount_times_forward_per_tenor():
    grid = mm.GridSpec(tenors=(0.2, 0.3), log_moneyness=(-0.1, 0.0, 0.1))

    def month(tau, forward, discount):
        times = tuple(np.linspace(tau - 29 / 365, tau, 21).tolist())
        return AsianContract(f"{tau}", times, (forward,) * 21, 21, 0.0, forward, "C", discount)

    scale = movie.grid_scale([month(0.15, 80.0, 0.99), month(0.45, 70.0, 0.97)], grid)
    w = (np.array([0.2, 0.3]) - 0.15) / 0.3
    per_tenor = (1 - w) * 80 * 0.99 + w * 70 * 0.97
    np.testing.assert_allclose(scale, np.repeat(per_tenor, 3))
    with pytest.raises(ValueError, match="two unseasoned"):
        movie.grid_scale([month(0.15, 80.0, 0.99)], grid)


def test_trailing_market_never_sees_its_own_day_or_later():
    grid = mm.GridSpec(tenors=(0.2, 0.3), log_moneyness=(-0.1, 0.0, 0.1))
    dates = tuple(str(d.date()) for d in pd.bdate_range("2021-01-04", periods=30))
    z = np.random.default_rng(0).uniform(size=(30, 2))
    vectors = np.column_stack([z[:, 0], z[:, 1], z.sum(1), z[:, 0] ** 2, z[:, 1] ** 2,
                               z.prod(1)])
    panel = mm.MarketPanel(dates, vectors, grid)
    model = movie.trailing_market(panel, dates[20], 10,
                                  mm.MarketConfig(latent_dim=2, width=8, depth=2, epochs=20))
    assert model.metadata["window"] == [dates[10], dates[19]]
    assert model.metadata["window_days"] == 10 and len(model.train_units) == 10


def test_typical_test_day_is_the_median_of_each_held_out_month():
    rows = [{"date": f"2024-07-{d:02d}", "month": "2024-07", "split": "test",
             "manifold_distance": v} for d, v in zip(range(1, 6), [5, 1, 4, 2, 3])]
    rows.append({"date": "2024-06-28", "month": "2024-06", "split": "train",
                 "manifold_distance": 0})
    picked = movie.typical_test_days({"test_months": ["2024-07"], "rows": rows})
    assert picked["2024-07"]["manifold_distance"] == 3


def test_movie_page_holds_each_day_and_embeds_fonts_and_three(tmp_path):
    heston = plane_heston()
    cloud = movie.heston_cloud(heston, 64)
    views = [movie.day_slice(heston, lifted_market(lift), np.ones(N), np.full(4, 0.5), 1,
                             extent=0.05, n_grid=6, n_lines=3, cloud=cloud)
             for lift in (0.8, 1.2)]
    views[0]["heston"][0, 0] = np.nan  # Heston cannot reach this floor point on day 1 only
    views[1]["market"][-1, -1] = np.nan
    views[1]["lines"][2] = {**views[1]["lines"][2], "start": np.full(3, np.nan)}
    path = movie.write_html(views, tmp_path / "clip.html", ["2021-11-19", "2021-11-22"],
                            hold=2.0, move=1.0)
    html = path.read_text()
    assert f"three@{movie.THREE_VERSION}/build/three.module.js" in html
    assert 'font-family: "CM"' in html and "data:font/ttf;base64," in html
    data = json.loads(html.split("const D = ")[1].split(";\n")[0])
    assert data["seconds"] == movie.movie_seconds(2, 2.0, 1.0) == 5.0
    assert [d["date"] for d in data["days"]] == ["Nov 19, 2021", "Nov 22, 2021"]
    # An unreachable point keeps its floor place, borrows a neighbour's height, fades out.
    corner = data["days"][0]["heston"][0][0]
    np.testing.assert_allclose(corner[:2], views[1]["heston"][0, 0, :2], atol=1e-7)
    assert corner[2] == data["days"][0]["heston"][0][1][2]
    assert data["days"][0]["hestonAlpha"][0][0] == 0
    assert data["days"][1]["hestonAlpha"][0][0] == 1
    far = data["days"][1]["market"][-1][-1]
    assert far[2] == data["days"][1]["market"][-1][-2][2]
    assert data["days"][1]["marketAlpha"][-1][-1] == 0
    # A line missing on one day borrows the other day's line and is hidden there.
    hidden = data["days"][1]["lines"][2]
    assert hidden["v"] == 0 and hidden["s"] == data["days"][0]["lines"][2]["s"]
    assert [len(d["lines"]) for d in data["days"]] == [4, 4]
    assert data["days"][0]["lines"][0]["day"] and data["units"] == "USD/bbl"
    with pytest.raises(ValueError, match="One projected slice per date"):
        movie.write_html(views, tmp_path / "bad.html", ["2021-11-19"])


def test_flow_frames_average_nearby_days_into_month_labelled_keyframes(tmp_path):
    heston = plane_heston()
    cloud = movie.heston_cloud(heston, 64)
    days = [movie.day_slice(heston, lifted_market(lift), np.ones(N), np.full(4, 0.5), 1,
                            extent=0.05, n_grid=5, n_lines=2, cloud=cloud)
            for lift in (0.8, 0.8, 1.6, 1.6, 1.6)]
    days[1]["market_hidden"] = True  # a day past its market model
    dates = ["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02", "2024-02-05"]
    frames, labels = movie.flow_frames(dates, days, every=2, sigma=0.01)
    assert labels == ["Jan 2024", "Feb 2024", "Feb 2024"]  # days 0, 2 and the last
    gaps = [f["lines"][0]["distance"] for f in frames]
    np.testing.assert_allclose(gaps, np.array([0.8, 1.6, 1.6]) / np.sqrt(N), atol=1e-8)
    wide, _ = movie.flow_frames(dates, days, every=1, sigma=50.0)  # near-flat weights
    assert 0 < wide[1]["market_alpha"].max() < 1  # the hidden day dims the average
    path = movie.write_html(frames, tmp_path / "flow.html", dates[::2], hold=0.0, move=0.5,
                            labels=labels, note="7-day smoothing")
    data = json.loads(path.read_text().split("const D = ")[1].split(";\n")[0])
    assert data["days"][0]["date"] == "Jan 2024" and data["note"] == "7-day smoothing"
    assert data["seconds"] == 1.0


def test_ragged_points_are_blanked_and_smooth_sheets_kept():
    a, b = np.meshgrid(np.linspace(-0.1, 0.1, 9), np.linspace(-0.1, 0.1, 9))
    sheet = np.stack([a, b, 0.3 * a + 2 * a * b], axis=-1)  # tilted and gently curved
    np.testing.assert_array_equal(movie._drop_ragged(sheet, 0.1), sheet)
    sheet[4, 4, 2] += 0.05  # one point on another branch
    cleaned = movie._drop_ragged(sheet, 0.1)
    assert np.isnan(cleaned[4, 4]).all() and np.isfinite(cleaned[0, 0]).all()


def test_finale_adds_its_seconds_and_math_labels(tmp_path):
    heston = plane_heston()
    cloud = movie.heston_cloud(heston, 64)
    days = [movie.day_slice(heston, lifted_market(lift), np.ones(N), np.full(4, 0.5), 1,
                            extent=0.05, n_grid=7, n_lines=2, cloud=cloud) for lift in (0.8, 1.2)]
    path = movie.write_html(days, tmp_path / "zoom.html", ["2025-12-30", "2025-12-31"],
                            hold=0.0, move=19.0, finale=7.0,
                            rescale={"at": 4.0, "over": 1.2, "from": 1})
    html = path.read_text()
    data = json.loads(html.split("const D = ")[1].split(";\n")[0])
    assert data["flow"] == 19.0 and data["finale"] == 7.0 and data["seconds"] == 26.0
    assert data["reveal"] == pytest.approx(0.25 * 19 / 26)
    assert data["rescale"] == {"at": 4.0, "over": 1.2, "from": 1}
    assert '"T_pM_h"' in html and '"M_h"' in html and '"M_m"' in html
    assert 'font-family: "CMI"' in html  # Computer Modern math italic for the labels


def test_gap_colours_use_one_fixed_scale_from_the_movies_own_gaps():
    gaps = np.r_[np.full(90, 0.04), np.linspace(0.02, 0.3, 10)]
    low, high = movie.gap_scale(gaps)
    assert low <= np.percentile(gaps, 5) and high >= np.percentile(gaps, 95)
    assert low in (0.02, 0.025, 0.03, 0.04) and high <= 0.3
    assert movie.gap_scale([]) == [0.02, 0.2]
