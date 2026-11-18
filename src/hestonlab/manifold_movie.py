"""Render cached weekly results without Monte Carlo or Heston calibration."""

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
        if "paired_rms" in snapshots[0]:
            item["paired_rms"] = np.einsum(
                "d,dij->ij", w, np.array([s["paired_rms"] for s in snapshots])
            )
        for name in ("heston", "market"):
            points = np.array([s[name] for s in snapshots])
            alpha = np.array([s[name + "_alpha"] for s in snapshots])
            mass = w[:, None, None] * alpha
            total = mass.sum(axis=0)
            average = np.einsum("dij,dijk->ijk", mass, points) / np.maximum(total[..., None], 1e-12)
            # Low confidence stays faded. Smoothing is display-only.
            item[name] = average
            item[name + "_alpha"] = total
        for lines in zip(*(s["lines"] for s in snapshots), strict=True):
            visible = np.array([x.get("visible", 1) for x in lines])
            mass = w * visible
            total = mass.sum()
            use = mass / total if total > 0 else w
            item["lines"].append(
                {
                    "start": use @ np.array([x["start"] for x in lines]),
                    "end": use @ np.array([x["end"] for x in lines]),
                    "distance": float(use @ np.array([x["distance"] for x in lines])),
                    "day": lines[0]["day"],
                    "visible": float(total),
                }
            )
        warnings = {
            warning
            for s, weight in zip(snapshots, w)
            if weight > 0.05
            for warning in s.get("warning", "").split(" · ")
        }
        item["warning"] = " · ".join(sorted(w for w in warnings if w))
        result.append(item)
    return result


def align_reference(frames):
    """Apply one common affine shear, preserving both shapes and their separation.

    Anchor the market centre and remove its fitted linear tilt from both sheets.
    Reverse the common vertical direction so Heston lies above the market.
    No curvature is amplified and no model surface is replaced with a distance field.
    """
    from copy import deepcopy

    transformed = deepcopy(frames)
    for frame in transformed:
        h, m = frame["heston"], frame["market"]
        np.testing.assert_allclose(h[..., :2], m[..., :2], atol=1e-12)
        xy = h[..., :2].reshape(-1, 2)
        slopes = np.linalg.lstsq(
            np.column_stack((np.ones(len(xy)), xy)), m[..., 2].ravel(), rcond=None
        )[0][1:]
        centre = m[m.shape[0] // 2, m.shape[1] // 2]
        intercept = float(centre[2] - centre[:2] @ slopes)
        plane = intercept + (xy @ slopes).reshape(h.shape[:2])
        h[..., 2] = plane - h[..., 2]
        m[..., 2] = plane - m[..., 2]
        for line in frame["lines"]:
            for endpoint in ("start", "end"):
                point = line[endpoint]
                point[2] = intercept + point[:2] @ slopes - point[2]
    return transformed


def vertical_ranges(frames):
    """Widen the z range in the first half, tighten it in the second, without clipping.

    Prefix/suffix envelopes retain every vertex. Linear interpolation of these
    bounds encloses the linearly interpolated meshes between keyframes too.
    Zero stays at a fixed height in the scene, anchoring the market during zoom.
    """
    lower = np.array([min(float(f[k][..., 2].min()) for k in ("heston", "market")) for f in frames])
    upper = np.array([max(float(f[k][..., 2].max()) for k in ("heston", "market")) for f in frames])
    below_reference = 0.15
    required = 1.06 * np.maximum(np.maximum(upper, -lower / below_reference), 1e-5)
    middle = len(frames) // 2
    peak = float(required.max())
    start, end = float(required[:3].max()), float(required[-3:].max())
    t = np.linspace(0, 1, len(frames))

    def ease(u):
        return 0.5 - 0.5 * np.cos(np.pi * u)

    desired = np.where(
        t <= 0.5, start + (peak - start) * ease(2 * t), peak + (end - peak) * ease(2 * t - 1)
    )
    safe = np.r_[
        np.maximum.accumulate(required[:middle]),
        np.maximum.accumulate(required[middle:][::-1])[::-1],
    ]
    top = np.maximum(desired, safe)
    return np.column_stack((-below_reference * top, top)).tolist()


def render(
    out,
    video=True,
    stable=True,
    destination=None,
    seconds=None,
    fps=None,
    window="local",
    fixed_camera=False,
):
    out = Path(out)
    config = Config.read(out / "config.json")
    report = json.loads((out / "report.json").read_text())
    fingerprint = report["fingerprint"]
    snapshots, positions, dates, omitted, saved_arrays, saved_rows = [], [], [], [], [], []
    for summary in report["rows"]:
        data = out / "weeks" / fingerprint[:20] / (summary["date"] + ".npz")
        if sha_file(data) != summary["arrays_sha256"]:
            raise ValueError(f"Corrupt cached geometry: {data}")
        with np.load(data, allow_pickle=False) as z:
            arrays = {key: z[key] for key in z.files}
        if "heston_patch" not in arrays:
            omitted.append(summary["date"])
            continue
        saved_arrays.append(arrays)
        saved_rows.append(summary)
        if not stable:
            snapshots.append(project_snapshot(arrays, summary))
        positions.append(summary["panel_index"])
        dates.append(summary["date"])
    if len(saved_arrays) < 2:
        raise ValueError("At least two cached surfaces are required to render")
    geometry = None
    if stable:
        from .stable_geometry import fixed_snapshots

        with np.load(out / "panel.npz", allow_pickle=False) as z:
            panel = {key: z[key] for key in z.files}
        snapshots, geometry = fixed_snapshots(
            saved_arrays, saved_rows, panel, config, window=window
        )
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
        if "paired_rms" in a:
            frame["paired_rms"] = (1 - w) * a["paired_rms"] + w * b["paired_rms"]
        for name in ("heston", "market", "heston_alpha", "market_alpha"):
            frame[name] = (1 - w) * a[name] + w * b[name]
        frame["lines"] = []
        for x, y in zip(a["lines"], b["lines"], strict=True):
            line = {
                name: (1 - w) * np.asarray(x[name]) + w * np.asarray(y[name])
                for name in ("start", "end", "distance", "visible")
            }
            line["day"] = x["day"]
            frame["lines"].append(line)
        frame["warning"] = " · ".join(
            sorted(
                {
                    warning
                    for item in (a, b)
                    for warning in item.get("warning", "").split(" · ")
                    if warning
                }
            )
        )
        frames.append(frame)
        labels.append(str(pd.Timestamp(int((1 - w) * date_days[lo] + w * date_days[hi])).date()))
    if stable:
        frames = align_reference(frames)
    ranges = vertical_ranges(frames)
    partial = len(report["rows"]) < report["total_weeks"]
    flagged = any(r["flags"] for r in report["rows"])
    note = "SYNTHETIC TEST · " if report.get("synthetic") else ""
    note += "2-week smoothing"
    target = Path(destination) if destination is not None else out
    target.mkdir(parents=True, exist_ok=True)
    duration = config.seconds if seconds is None else seconds
    frame_rate = config.fps if fps is None else fps
    if duration <= 0 or frame_rate <= 0:
        raise ValueError("Positive movie duration and frame rate required")
    path = target / "animation.html"
    write_html(
        frames,
        path,
        dates,
        labels=labels,
        hold=0,
        move=duration / (len(frames) - 1),
        reveal=0.25,
        note=note,
        fixed_camera=fixed_camera,
        steady_surfaces=stable,
        vertical_ranges=ranges,
        compact_hud=True,
        top_sheet="heston" if stable else "market",
        show_warnings=False,
        tangent_planes=stable,
        axes=["slice direction 1", "slice direction 2", "gap"],
        axis_units=["section fraction" if stable and window == "local" else "USD/bbl"] * 2
        + ["USD/bbl"],
        local_line_note="Local labels: RMS (USD/bbl)" if stable else "",
    )
    # Offline assets are prepared before production; no CDN dependency while rendering.
    asset_root = Path(__file__).resolve().parents[2] / "outputs" / "web_assets"
    if (asset_root / "three.module.js").exists() and (asset_root / "CSS2DRenderer.js").exists():
        import shutil

        local = target / "assets"
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
        (target / "animation_manifest.json").resolve(),
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
            "stage": report.get("stage", "all"),
            "stage_complete": report.get("stage_complete", not partial),
            "accuracy_flags": flagged,
            "stable_geometry": geometry,
            "presentation": {
                "view": "projected_manifolds",
                "curvature_gain": 1,
                "common_market_linear_tilt_removed": stable,
                "reference_manifold": "market" if stable else None,
                "market_centre_height": 0 if stable else None,
                "common_vertical_direction_reversed": stable,
                "projected_vertical_separation_preserved": True,
                "market_surface_is_rms_offset_not_literal_projection": False,
                "numerical_error_labels_changed": False,
                "camera": "fixed" if fixed_camera else "orbit",
                "camera_distance_multiplier": 1,
                "error_reveal_start_seconds": duration * 0.25,
                "error_reveal_complete_seconds": duration * 0.31,
                "warnings_shown": False,
                "tangent_planes": 2 if stable else 0,
                "tangent_reveal_start_seconds": duration * 0.5,
                "tangent_reveal_complete_seconds": duration * 0.56,
                "tangent_statistic": "max absolute vertical departure of displayed slice from local finite-difference tangent",
                "tangent_statistic_units": "mUSD/bbl",
                "tangent_statistic_is_intrinsic_curvature": False,
                "zoom": "z-axis range widens first half, tightens second half",
                "vertical_ranges_usd_per_bbl": ranges,
            },
            "seconds": duration,
            "fps": frame_rate,
        },
    )
    if video:
        record_video(path, target / "animation.mp4", duration, frame_rate)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/weekly_cpp"))
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--raw-patches", action="store_true", help="original varying-footprint diagnostic view"
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("outputs"),
        help="export directory (default: outputs)",
    )
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--fixed-camera", action="store_true")
    parser.add_argument(
        "--window",
        choices=("local", "fixed"),
        default="local",
        help="wider normalized local sections or shared physical window",
    )
    args = parser.parse_args(argv)
    render(
        args.out,
        not args.no_video,
        not args.raw_patches,
        args.destination,
        args.seconds,
        args.fps,
        args.window,
        args.fixed_camera,
    )


if __name__ == "__main__":
    main()
