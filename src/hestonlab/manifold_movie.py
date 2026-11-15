"""Render the cached weekly C++ results. This module never runs Monte Carlo or fits models."""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from .movie_render import write_html, record_video
from .weekly_store import Config, write_json, sha_file


def smoothing_weights(positions, sigma):
    """Centered retrospective smoothing in trading-day coordinates, not snapshot indices."""
    positions = np.asarray(positions, dtype=float)
    if sigma <= 0 or len(positions) < 2 or not (np.diff(positions) > 0).all():
        raise ValueError("Increasing trading-day positions and positive bandwidth required")
    # Irregular intervals (holidays/missing weeks) must not receive equal temporal mass.
    widths = np.r_[
        positions[1] - positions[0],
        (positions[2:] - positions[:-2]) / 2,
        positions[-1] - positions[-2],
    ]
    weights = (
        np.exp(-0.5 * ((positions[:, None] - positions[None, :]) / sigma) ** 2) * widths[None, :]
    )
    return weights / weights.sum(axis=1, keepdims=True)


def project_snapshot(arrays, summary):
    n = len(arrays["scale"])
    scale = arrays["scale"] / np.sqrt(n)
    center = arrays["heston_prices"] * scale
    gap = arrays["market_point"] / np.sqrt(n) - center
    floor = arrays["axes"]
    up = gap - floor @ (floor.T @ gap)
    if np.linalg.norm(up) < 1e-10:
        candidate = np.eye(n)[:, np.argmin(np.sum(floor**2, axis=1))]
        up = candidate - floor @ (floor.T @ candidate)
    up /= np.linalg.norm(up)
    frame = np.column_stack([floor, up])
    # Anchor common-random-number price differences at the independently validated
    # fitted price. Otherwise sampling noise moves the patch center off its gap line.
    middle = arrays["heston_patch"].shape[0] // 2
    anchor = arrays["heston_patch"][middle, middle]
    h = ((arrays["heston_patch"] - anchor) * scale) @ frame
    m = (arrays["market_patch"] / np.sqrt(n) - center) @ frame
    # Retain unsupported coordinates only as invisible placeholders for WebGL topology.
    # No missing observations or numerical gaps are synthesized.
    ha = np.isfinite(h).all(axis=-1).astype(float)
    ma = np.isfinite(m).all(axis=-1).astype(float)
    if not bool(arrays["market_supported"]):
        ma[:] = 0
    h = np.nan_to_num(h)
    m = np.nan_to_num(m)
    return {
        "slice": 1,
        "heston": h,
        "market": m,
        "heston_alpha": ha,
        "market_alpha": ma,
        "lines": [
            {
                "start": gap @ frame,
                "end": np.zeros(3),
                "distance": summary["gap_usd"],
                "day": True,
                "visible": float(bool(arrays["market_supported"])),
            }
        ],
        "status": summary["status"],
        "date": summary["date"],
    }


def smooth_snapshots(snapshots, positions, sigma):
    weights = smoothing_weights(positions, sigma)
    result = []
    for i, w in enumerate(weights):
        item = {"slice": 1, "lines": []}
        for name in ("heston", "market"):
            points = np.array([s[name] for s in snapshots])
            alpha = np.array([s[name + "_alpha"] for s in snapshots])
            mass = w[:, None, None] * alpha
            total = mass.sum(axis=0)
            average = np.einsum("dij,dijk->ijk", mass, points) / np.maximum(total[..., None], 1e-12)
            # Low confidence stays faded. Smoothing is display-only.
            item[name] = average
            item[name + "_alpha"] = total
        lines = [s["lines"][0] for s in snapshots]
        visible = np.array([x.get("visible", 1) for x in lines])
        mass = w * visible
        total = mass.sum()
        use = mass / total if total > 0 else w
        item["lines"] = [
            {
                "start": use @ np.array([x["start"] for x in lines]),
                "end": use @ np.array([x["end"] for x in lines]),
                "distance": float(use @ np.array([x["distance"] for x in lines])),
                "day": True,
                "visible": float(total),
            }
        ]
        result.append(item)
    return result


def render(out, video=True):
    out = Path(out)
    config = Config.read(out / "config.json")
    report = json.loads((out / "report.json").read_text())
    fingerprint = report["fingerprint"]
    snapshots, positions, dates, omitted = [], [], [], []
    for summary in report["rows"]:
        data = out / "weeks" / fingerprint[:20] / (summary["date"] + ".npz")
        if sha_file(data) != summary["arrays_sha256"]:
            raise ValueError(f"Corrupt cached geometry: {data}")
        with np.load(data, allow_pickle=False) as z:
            arrays = {key: z[key] for key in z.files}
        if "heston_patch" not in arrays:
            omitted.append(summary["date"])
            continue
        snapshots.append(project_snapshot(arrays, summary))
        positions.append(summary["panel_index"])
        dates.append(summary["date"])
    if len(snapshots) < 2:
        raise ValueError("At least two cached surfaces are required to render")
    smoothed = smooth_snapshots(snapshots, positions, config.smoothing_days)
    # Preserve timing of missing/irregular weeks: interpolate display geometry onto
    # evenly spaced trading-day positions. These are explicitly not daily estimates.
    uniform = np.linspace(positions[0], positions[-1], len(positions))
    frames = []
    labels = []
    date_days = np.array([pd.Timestamp(d).value for d in dates], dtype=float)
    for p in uniform:
        hi = min(int(np.searchsorted(positions, p, side="right")), len(positions) - 1)
        lo = max(0, hi - 1)
        w = (p - positions[lo]) / max(positions[hi] - positions[lo], 1)
        a, b = smoothed[lo], smoothed[hi]
        frame = {"slice": 1}
        for name in ("heston", "market", "heston_alpha", "market_alpha"):
            frame[name] = (1 - w) * a[name] + w * b[name]
        x, y = a["lines"][0], b["lines"][0]
        frame["lines"] = [
            {
                name: (1 - w) * np.asarray(x[name]) + w * np.asarray(y[name])
                for name in ("start", "end", "distance", "visible")
            }
        ]
        frame["lines"][0]["day"] = True
        frames.append(frame)
        labels.append(str(pd.Timestamp(int((1 - w) * date_days[lo] + w * date_days[hi])).date()))
    partial = len(report["rows"]) < report["total_weeks"]
    flagged = any(r["flags"] for r in report["rows"])
    note = "SYNTHETIC TEST · " if report.get("synthetic") else ""
    note += "weekly estimates · retrospective 2-week smoothing · projected slices"
    if partial:
        note += " · PARTIAL RUN"
    if flagged:
        note += " · accuracy flags: see report"
    if omitted:
        note += " · unavailable surfaces omitted"
    path = out / "animation.html"
    write_html(
        frames,
        path,
        dates,
        labels=labels,
        hold=0,
        move=config.seconds / (len(frames) - 1),
        note=note,
    )
    # Offline assets are prepared before production; no CDN dependency while rendering.
    asset_root = Path(__file__).resolve().parents[2] / "outputs" / "web_assets"
    if (asset_root / "three.module.js").exists() and (asset_root / "CSS2DRenderer.js").exists():
        import shutil

        local = out / "assets"
        local.mkdir(exist_ok=True)
        for name in ("three.module.js", "CSS2DRenderer.js", "LICENSE"):
            shutil.copyfile(asset_root / name, local / name)
        html = path.read_text().replace(
            "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
            "./assets/three.module.js",
        )
        html = html.replace("https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/", "./assets/")
        html = html.replace(
            "three/addons/renderers/CSS2DRenderer.js", "three/addons/CSS2DRenderer.js"
        )
        path.write_text(html)
    else:
        raise FileNotFoundError("Offline renderer assets missing: run scripts/prepare_weekly.py")
    write_json(
        out / "animation_manifest.json",
        {
            "fingerprint": fingerprint,
            "frames": len(frames),
            "sample_dates": dates,
            "omitted_no_surface_dates": omitted,
            "smoothing_standard_deviation_trading_days": config.smoothing_days,
            "centered_smoothing_uses_future_snapshots": True,
            "interpolated_frames_are_not_daily_estimates": True,
            "displayed_sheet_is_projected_parameter_slice": True,
            "heston_display_uses_crn_differences_anchored_at_validated_price": True,
            "partial": partial,
            "accuracy_flags": flagged,
        },
    )
    if video:
        record_video(path, out / "animation.mp4", config.seconds, config.fps)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/weekly_cpp"))
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args(argv)
    render(args.out, not args.no_video)


if __name__ == "__main__":
    main()
