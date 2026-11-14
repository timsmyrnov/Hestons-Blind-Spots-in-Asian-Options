"""Gap heatmap: smoothing, colour limits and the page payload."""

import json
import numpy as np
import pytest

pytest.importorskip("torch")

from hestonlab import gap_heatmap as hm  # noqa: E402
from hestonlab.market_manifold import GridSpec  # noqa: E402


def test_smoothing_keeps_constants_and_averages_neighbours():
    flat = np.full((6, 21), 0.05)
    np.testing.assert_allclose(hm.smooth(flat, 2.0), flat)
    spike = np.zeros((7, 1))
    spike[3] = 1.0
    out = hm.smooth(spike, 1.0)[:, 0]
    assert out[3] < 1 and out[2] == pytest.approx(out[4]) and out.argmax() == 3
    np.testing.assert_array_equal(hm.smooth(spike, 0), spike)


def test_colour_limit_is_a_clean_step_above_almost_every_gap():
    gaps = np.linspace(-0.23, 0.17, 400)
    limit = hm.nice_scale(gaps)
    assert limit in (0.2, 0.25) and np.mean(np.abs(gaps) <= limit) >= 0.98


def test_ramp_is_blue_grey_red_with_matching_ends():
    assert len(hm.DIVERGING) == 9 and hm.DIVERGING[4] == "#f0efec"
    assert hm.DIVERGING[0].startswith("#1c") and hm.DIVERGING[-1].startswith("#ad")


def test_page_carries_smoothed_gaps_noise_and_labels(tmp_path):
    grid = GridSpec(tenors=(0.2, 0.3, 0.4), log_moneyness=tuple(np.linspace(-0.2, 0.2, 7)))
    rows = [{"date": f"2025-01-{d:02d}", "gap": np.full(21, 0.01 * d), "noise": np.full(21, 0.002),
             "rms": 0.01 * d} for d in (2, 3, 6)]
    path = hm.write_html(rows, grid, tmp_path / "h.html", seconds=10.0, sigma=0.0)
    data = json.loads(path.read_text().split("const D = ")[1].split(";\n")[0])
    assert data["dates"] == ["Jan 2, 2025", "Jan 3, 2025", "Jan 6, 2025"]
    assert data["tenors"] == ["2.4 mo", "3.6 mo", "4.8 mo"]
    assert data["strikes"] == ["82%", "88%", "94%", "100%", "107%", "114%", "122%"]
    np.testing.assert_allclose(data["gaps"][2], 0.06)
    assert data["seconds"] == 10.0 and data["limit"] >= 0.06
    assert 'font-family: "CM"' in path.read_text()
