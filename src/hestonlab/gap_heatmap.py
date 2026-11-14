"""Per-option gap heatmap: market settlement prices minus Heston, animated over a year.

For each day and each of the 21 grid options (3 tenors x 7 strikes):

    gap = market grid price - Heston grid price      (USD/bbl, signed)

- Heston parameters: the best Heston fit to that day's grid (nearest point on the
  Heston manifold, equal USD weights).
- Heston prices: Monte Carlo at those parameters on the day's actual listed options
  (262,144 paths), so the neural surrogate's error does not enter.
- Both sides go through the same grid interpolation (day_vector) from the same listed
  options, so interpolation bias cancels.
- Noise: each cell's Monte Carlo standard error carried through the interpolation.

    python -m hestonlab.gap_heatmap --run outputs/market_manifold/archive \
        --root "$HESTONLAB_DATA_ROOT" --year 2025 --seconds 10
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

from .calibration import PARAMETER_NAMES
from .heston import HestonParameters, MonteCarloConfig, price_asian_options
from .market_manifold import GridSpec, day_vector, load_archive_days, nearest_point
from .manifold_movie import _font, grid_scale, heston_cloud, record_video
from .surrogate import Surrogate

# Blue (market below Heston) - grey - red (market above Heston); arms equal in OKLab
# lightness step by step, poles matched (blue from the reference sequential ramp).
DIVERGING = ["#1c5cab", "#5481bd", "#87a6ce", "#bbcade", "#f0efec",
             "#e6beb8", "#d78d85", "#c45a54", "#ad1722"]

_WORKER = {}


def _init(model_path, cloud_size):
    heston = Surrogate.load(model_path)
    _WORKER.update(heston=heston, cloud=heston_cloud(heston, cloud_size))


def day_gap(date, contracts, prices, grid, paths=262144, seed=0, heston=None, cloud=None):
    """One day's 21 gaps (USD/bbl), their Monte Carlo noise, and the fit behind them."""
    heston = heston or _WORKER["heston"]
    cloud = cloud or _WORKER["cloud"]
    prices = np.asarray(prices, dtype=float)
    y = day_vector(contracts, prices, grid)
    if not np.isfinite(y).all():
        raise ValueError(f"{date}: the grid is not covered")
    weight = grid_scale(contracts, grid)  # normalized -> USD/bbl, per grid point
    fit = nearest_point(heston, y, weight, *cloud, 6)
    params = heston.input_scaler.from_unit(fit["unit"])
    use = [i for i, (c, p) in enumerate(zip(contracts, prices))
           if c.known_fixing_sum == 0 and len(c.fixing_times) == c.total_fixings and p > 0]
    listed = [contracts[i] for i in use]
    config = MonteCarloConfig(paths=paths, max_step_years=1 / 365, seed=seed)
    estimates = price_asian_options(listed, HestonParameters(**dict(zip(PARAMETER_NAMES,
                                                                        params))), config)
    model = np.array([e.price for e in estimates])
    error = np.array([e.standard_error for e in estimates])
    # Same listed options, same interpolation, for both sides.
    h = day_vector(listed, model, grid)
    h_up = day_vector(listed, model + error, grid)
    y_same = day_vector(listed, prices[use], grid)
    gap = (y_same - h) * weight
    return {"date": date, "gap": gap, "noise": np.abs(h_up - h) * weight,
            "rms": float(np.sqrt(np.mean(gap ** 2))), "surrogate_rms": fit["distance"],
            "surrogate_minus_mc": (fit["point"] - h) * weight,
            "parameters": dict(zip(PARAMETER_NAMES, np.round(params, 6).tolist())),
            "at_bound": bool(fit["boundary"])}


def year_gaps(run_dir, root, year, jobs=4, paths=262144, log=print):
    """day_gap for every complete grid day of `year`, in `jobs` processes."""
    run_dir = Path(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    grid = GridSpec(**{k: tuple(v) if isinstance(v, list) else v
                       for k, v in report["grid"].items()})
    days = load_archive_days(root, start=f"{year}-01-01", end=f"{year}-12-31")
    work = []
    for i, (date, contracts, prices) in enumerate(days):
        if np.isfinite(day_vector(contracts, prices, grid)).all():
            work.append((str(pd.Timestamp(date).date()), contracts, prices, grid, paths, 1000 + i))
    if not work:
        raise ValueError(f"No complete grid day in {year}")
    log(f"{year}: {len(work)} days, {jobs} processes, {paths} Monte Carlo paths")
    with ProcessPoolExecutor(jobs, initializer=_init,
                             initargs=(run_dir / "heston" / "model.pt", 2048)) as pool:
        rows = list(pool.map(_day_gap_args, work))
    return grid, sorted(rows, key=lambda r: r["date"])


def _day_gap_args(args):
    return day_gap(*args)


def smooth(values, sigma):
    """Gaussian average over neighbouring days (sd `sigma` days), renormalized at ends."""
    values = np.asarray(values, dtype=float)
    if sigma <= 0:
        return values.copy()
    days = np.arange(len(values))
    weights = np.exp(-0.5 * ((days[:, None] - days[None, :]) / sigma) ** 2)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.tensordot(weights, values, axes=(1, 0))


def nice_scale(gaps):
    """Symmetric colour limit: 98th percentile of |gap|, rounded up to a clean step."""
    top = float(np.percentile(np.abs(gaps), 98)) or 1e-3
    step = 10 ** np.floor(np.log10(top))
    for m in (1, 2, 2.5, 5, 10):
        if m * step >= top:
            return m * step
    return 10 * step


def write_html(rows, grid, path, seconds=10.0, sigma=2.0):
    """Self-contained animated heatmap page (canvas 2-D, Computer Modern)."""
    gaps = np.array([r["gap"] for r in rows])
    noise = np.array([r["noise"] for r in rows])
    rms = np.array([r["rms"] for r in rows])
    shown = smooth(gaps, sigma)
    data = {
        "dates": [f"{pd.Timestamp(r['date']):%b} {pd.Timestamp(r['date']).day}, "
                  f"{pd.Timestamp(r['date']).year}" for r in rows],
        "months": [f"{pd.Timestamp(r['date']):%b}" for r in rows],
        "gaps": np.round(shown, 5).tolist(), "noise": np.round(smooth(noise, sigma), 5).tolist(),
        "rms": np.round(rms, 5).tolist(),
        "rmsSmooth": np.round(np.sqrt(smooth(gaps ** 2, sigma).mean(axis=1)), 5).tolist(),
        "tenors": [f"{12 * t:.1f} months" for t in grid.tenors],
        "strikes": [f"{100 * np.exp(x):.0f}%" for x in grid.log_moneyness],
        "limit": nice_scale(shown), "ramp": DIVERGING, "seconds": seconds,
        "note": f"{2 * sigma / 5:.0f}-week average" if sigma >= 2.5 else
                f"{max(1, round(2.35 * sigma)):.0f}-day average",
    }
    html = (HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
            .replace("__CMR__", _font("cmr10.ttf")).replace("__CMB__", _font("cmb10.ttf")))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gap heatmap</title>
<style>
  @font-face { font-family: "CM"; font-weight: 400; src: url(data:font/ttf;base64,__CMR__) format("truetype"); }
  @font-face { font-family: "CM"; font-weight: 700; src: url(data:font/ttf;base64,__CMB__) format("truetype"); }
  html, body { margin: 0; height: 100%; overflow: hidden; background: #fcfcfb; }
  canvas { display: block; width: 100vw; height: 100vh; }
</style></head>
<body><canvas id="c"></canvas>
<script>
const D = __DATA__;
const q = new URLSearchParams(location.search);
const exportTo = q.get("export"), frozen = q.has("t") ? Math.min(Math.max(+q.get("t"), 0), 1) : null;
const canvas = document.getElementById("c"), c = canvas.getContext("2d");
const VW = 1920, VH = 1080, n = D.dates.length;
const INK = "#1f2328", MUTED = "#5f6672", FAINT = "#8a8f98", RULE = "#d9dbe0";
const F = (w, s) => `${w} ${s}px "CM", serif`;
const hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
const RAMP = D.ramp.map(hex);
function shade(v) {  // v in [-1, 1] -> diverging colour
  const x = (Math.min(Math.max(v, -1), 1) + 1) / 2 * (RAMP.length - 1);
  const i = Math.min(Math.floor(x), RAMP.length - 2), t = x - i;
  return RAMP[i].map((a, k) => Math.round(a + (RAMP[i + 1][k] - a) * t));
}
const rgb = a => `rgb(${a[0]},${a[1]},${a[2]})`;
const lum = a => { const l = a.map(v => { v /= 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; });
                   return 0.2126 * l[0] + 0.7152 * l[1] + 0.0722 * l[2]; };
const sign = v => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(3);
const lerp = (a, b, w) => a + (b - a) * w;

// layout (virtual 1920 x 1080)
const X0 = 420, Y0 = 300, CW = 168, CH = 124, GAP = 6, COLS = 7, ROWS = 3;
const HW = COLS * CW + (COLS - 1) * GAP, HH = ROWS * CH + (ROWS - 1) * GAP;
const T0 = 790, TH = 120;  // average-gap line: top and height
const rmsMax = Math.max(...D.rmsSmooth) * 1.1;

const cents = v => { const c = Math.round(v * 100); return c === 0 ? "0" : (c > 0 ? "+" : "−") + Math.abs(c); };
function arrow(x1, x2, y, head) {  // thin line with an arrowhead at x2
  c.strokeStyle = FAINT; c.fillStyle = FAINT; c.lineWidth = 1.2;
  c.beginPath(); c.moveTo(x1, y); c.lineTo(x2, y); c.stroke();
  const d = Math.sign(x2 - x1);
  c.beginPath(); c.moveTo(x2, y); c.lineTo(x2 - d * head, y - head / 2); c.lineTo(x2 - d * head, y + head / 2); c.fill();
}
function frame(progress) {
  const x = progress * (n - 1), k = Math.min(Math.floor(x), Math.max(n - 2, 0)), w = n > 1 ? x - k : 0;
  const k2 = Math.min(k + 1, n - 1), day = Math.round(x);
  c.setTransform(canvas.width / VW, 0, 0, canvas.height / VH, 0, 0);
  c.fillStyle = "#fcfcfb"; c.fillRect(0, 0, VW, VH);
  c.textBaseline = "alphabetic"; c.textAlign = "left";
  c.fillStyle = INK; c.font = F(700, 32); c.fillText(D.dates[day], 96, 104);
  c.fillStyle = MUTED; c.font = F(400, 22);
  c.fillText("Market price minus Heston price, per option   (cents per barrel)", 96, 144);

  // puts <-- at the money --> calls, then the strikes
  const mid = X0 + 3 * (CW + GAP) + CW / 2, yA = Y0 - 78;
  c.font = F(400, 19); c.fillStyle = MUTED;
  c.textAlign = "left"; c.fillText("puts", X0, yA + 6);
  c.textAlign = "right"; c.fillText("calls", X0 + HW, yA + 6);
  c.textAlign = "center"; c.fillText("at the money", mid, yA + 6);
  const half = c.measureText("at the money").width / 2 + 18;
  arrow(mid - half, X0 + 58, yA, 9);
  arrow(mid + half, X0 + HW - 62, yA, 9);
  c.font = F(400, 19); c.fillStyle = MUTED;
  for (let j = 0; j < COLS; j++) c.fillText(D.strikes[j], X0 + j * (CW + GAP) + CW / 2, Y0 - 20);
  c.textAlign = "left"; c.font = F(400, 16); c.fillStyle = FAINT;
  c.fillText("strike, % of forward", X0 + HW + 22, Y0 - 20);
  c.textAlign = "right"; c.font = F(400, 20); c.fillStyle = MUTED;
  for (let i = 0; i < ROWS; i++) c.fillText(D.tenors[i], X0 - 22, Y0 + i * (CH + GAP) + CH / 2 + 7);

  // cells: colour and whole cents
  for (let i = 0; i < ROWS; i++) for (let j = 0; j < COLS; j++) {
    const idx = i * COLS + j, v = lerp(D.gaps[k][idx], D.gaps[k2][idx], w);
    const col = shade(v / D.limit), cx = X0 + j * (CW + GAP), cy = Y0 + i * (CH + GAP);
    c.fillStyle = rgb(col);
    c.beginPath(); c.roundRect(cx, cy, CW, CH, 6); c.fill();
    c.textAlign = "center"; c.font = F(400, 36);
    c.fillStyle = lum(col) < 0.22 ? "#ffffff" : INK;
    c.fillText(cents(v), cx + CW / 2, cy + CH / 2 + 12);
  }

  // one-line key
  const yK = Y0 + HH + 52;
  c.font = F(400, 19); c.textAlign = "left";
  let kx = X0;
  for (const [f, text] of [[0.85, "red = market above Heston"], [-0.85, "blue = market below Heston"]]) {
    c.fillStyle = rgb(shade(f)); c.beginPath(); c.roundRect(kx, yK - 14, 16, 16, 3); c.fill();
    c.fillStyle = MUTED; c.fillText(text, kx + 26, yK);
    kx += 26 + c.measureText(text).width + 48;
  }

  // average gap over the year: one line, the past dark, a dot now
  const TX = X0 + 150, TW = HW - 150, tx = i => TX + (n > 1 ? i / (n - 1) : 0) * TW;
  const ty = v => T0 + TH - v / rmsMax * TH;
  c.fillStyle = MUTED; c.font = F(400, 19); c.textAlign = "left";
  c.fillText("average gap", X0, T0 + TH / 2 + 6);
  c.lineJoin = "round"; c.lineCap = "round";
  c.strokeStyle = "#d4d6dc"; c.lineWidth = 2; c.beginPath();
  D.rmsSmooth.forEach((v, i) => i ? c.lineTo(tx(i), ty(v)) : c.moveTo(tx(i), ty(v))); c.stroke();
  const now = lerp(D.rmsSmooth[k], D.rmsSmooth[k2], w);
  c.strokeStyle = INK; c.lineWidth = 2.5; c.beginPath();
  for (let i = 0; i <= k; i++) i ? c.lineTo(tx(i), ty(D.rmsSmooth[i])) : c.moveTo(tx(i), ty(D.rmsSmooth[i]));
  c.lineTo(tx(x), ty(now)); c.stroke();
  c.fillStyle = INK; c.beginPath(); c.arc(tx(x), ty(now), 5.5, 0, 2 * Math.PI); c.fill();
  c.font = F(700, 19); c.fillText(`${(now * 100).toFixed(1)} cents`, tx(x) + 12, ty(now) - 10);
  c.font = F(400, 16); c.fillStyle = FAINT; c.textAlign = "center";
  const first = { Jan: -1, Jun: -1, Dec: -1 };
  D.months.forEach((m, i) => { if (m in first && first[m] < 0) first[m] = i; });
  for (const [m, i] of Object.entries(first)) if (i >= 0) c.fillText(m, tx(i), T0 + TH + 30);
  c.textAlign = "right"; c.fillText(D.note, VW - 96, VH - 56);
}

function size(w, h) { canvas.width = w; canvas.height = h; }
async function record() {
  await document.fonts.load(F(400, 16)); await document.fonts.load(F(700, 16));
  const fps = +(q.get("fps") || 60), count = Math.round(D.seconds * fps);
  size(+(q.get("w") || 1920), +(q.get("h") || 1080));
  const first = +(q.get("from") || 0), last = Math.min(+(q.get("to") || count), count);
  for (let i = first; i < last; i++) {
    frame(count > 1 ? i / (count - 1) : 1);
    const png = await new Promise(r => canvas.toBlob(r, "image/png"));
    await fetch(`${exportTo}/frame/${i}`, { method: "POST", body: png });
  }
  await fetch(`${exportTo}/done`, { method: "POST", body: String(last - first) });
}
let start = performance.now();
function tick(now) {
  frame(frozen ?? Math.min((now - start) / 1000 / D.seconds, 1));
  requestAnimationFrame(tick);
}
if (exportTo) record();
else document.fonts.ready.then(() => {
  size(Math.round(innerWidth * devicePixelRatio), Math.round(innerHeight * devicePixelRatio));
  addEventListener("resize", () => size(Math.round(innerWidth * devicePixelRatio), Math.round(innerHeight * devicePixelRatio)));
  addEventListener("click", () => { start = performance.now(); });
  requestAnimationFrame(tick);
});
</script></body></html>
"""


def build(run_dir, root, year, seconds=10.0, sigma=2.0, jobs=4, video=True, log=print):
    run_dir = Path(run_dir)
    grid, rows = year_gaps(run_dir, root, year, jobs, log=log)
    path = run_dir / "heatmaps" / f"gaps_{year}_{seconds:g}s.html"
    write_html(rows, grid, path, seconds, sigma)
    gaps = np.array([r["gap"] for r in rows])
    path.with_suffix(".json").write_text(json.dumps({
        "units": "USD/bbl, market minus Heston, per grid option",
        "grid": {"tenors": list(grid.tenors), "log_moneyness": list(grid.log_moneyness)},
        "mean_gap": np.round(gaps.mean(axis=0), 5).tolist(),
        "median_abs_gap": np.round(np.median(np.abs(gaps), axis=0), 5).tolist(),
        "days": [{**r, "gap": np.round(r["gap"], 5).tolist(),
                  "noise": np.round(r["noise"], 5).tolist(),
                  "surrogate_minus_mc": np.round(r["surrogate_minus_mc"], 5).tolist()}
                 for r in rows]}, indent=1))
    if video:
        log("recording video ...")
        record_video(path, path.with_suffix(".mp4"), seconds, 60)
    return path.with_suffix(".mp4") if video else path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="market_manifold output folder")
    parser.add_argument("--root", type=Path, required=True, help="private AO archive")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=2.0, help="smoothing sd, trading days")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args(argv)
    try:
        made = build(args.run, args.root, args.year, args.seconds, args.sigma, args.jobs,
                     not args.no_video)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Heatmap unavailable: {exc}\n")
    print(made)
    return made


if __name__ == "__main__":
    main()
