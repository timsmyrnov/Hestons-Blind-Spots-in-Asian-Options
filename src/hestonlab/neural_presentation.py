"""Render exact neural-manifold sections with the existing presentation style."""

from pathlib import Path
from zipfile import ZipFile, ZIP_STORED
import json
import shutil
import numpy as np
import pandas as pd

from .manifold_movie import smooth_snapshots, align_reference, vertical_ranges
from .movie_render import write_html, record_video
from .neural_training import NeuralSurface
from .neural_geometry import snapshot
from .stable_geometry import common_reference
from .weekly_store import sha_file, write_json, write_npz


def load_chart(path, date):
    with np.load(path, allow_pickle=False) as z:
        h, m = z["heston"], z["market"]
        lines = [
            dict(start=s, end=e, distance=float(d), day=bool(day), visible=1.0)
            for s, e, d, day in zip(z["starts"], z["ends"], z["distances"], z["days"], strict=True)
        ]
        return dict(
            slice=1,
            date=date,
            heston=h,
            market=m,
            heston_alpha=np.ones(h.shape[:2]),
            market_alpha=np.ones(m.shape[:2]),
            paired_rms=z["paired_rms"],
            lines=lines,
            warning="",
        )


def _ease(u):
    u = np.clip(u, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * u)


def steady_ranges(frames, duration, tight=(5.0, 17.0), ease=1.25, margin=1.06, below=0.15):
    """Vertical axis for the neural movie: tighter in the middle, still at the end.

    Before ``tight`` the original widening schedule is kept. Over ``ease`` seconds
    the axis eases down to one level that covers the highest sheet point from then
    until ``tight[1]``, and holds it, so the small gap stays readable. It then
    eases to one level that covers every later frame and holds it to the last
    frame: no closing zoom. Zero keeps its place in the box (``below`` of the top
    lies under it), so a rare downward bend of the market sheet may hang below
    the axis bottom; it is drawn, not clipped.
    """
    base = np.asarray(vertical_ranges(frames), float)[:, 1]
    top = np.array([max(float(f[k][..., 2].max()) for k in ("heston", "market")) for f in frames])
    t = duration * np.arange(len(frames)) / max(len(frames) - 1, 1)
    start, stop = tight
    middle = margin * top[(t >= start - ease) & (t <= stop)].max()
    end = margin * top[t >= stop].max()
    into = np.interp(start - ease, t, base)
    level = np.where(
        t < start - ease,
        base,
        np.where(
            t <= stop,
            into + (middle - into) * _ease((t - start + ease) / ease),
            middle + (end - middle) * _ease((t - stop) / ease),
        ),
    )
    return np.column_stack((-below * level, level)).tolist()


def render(out, config, destination=Path("outputs"), video=True, seconds=None, fps=None):
    out, destination = Path(out), Path(destination)
    report = json.loads((out / "report.json").read_text())
    rows = report["rows"]
    if len(rows) < 2:
        raise ValueError("At least two neural weeks required for a movie")
    model = NeuralSurface.load(out / "heston/model.pt")
    arrays = []
    for row in rows:
        path = out / "weeks" / (row["date"] + ".npz")
        if sha_file(path) != row["arrays_sha256"]:
            raise ValueError("Neural array checksum mismatch before rendering")
        with np.load(path, allow_pickle=False) as z:
            arrays.append({k: z[k] for k in z.files})
    reference_path = out / "reference_axes.npz"
    if reference_path.exists():
        with np.load(reference_path, allow_pickle=False) as z:
            reference = z["axes"]
    else:
        reference = common_reference(arrays)
        write_npz(reference_path, axes=reference)
    reference_hash = sha_file(reference_path)
    snapshots, details = [], []
    for row, data in zip(rows, arrays, strict=True):
        date = row["date"]
        chart_path = out / "weeks" / (date + ".chart.npz")
        meta_path = chart_path.with_suffix(".json")
        identity = dict(
            reference_sha256=reference_hash,
            arrays_sha256=row["arrays_sha256"],
            market_sha256=row["market_checkpoint_sha256"],
            heston_sha256=report["identity"]["heston_checkpoint_sha256"],
            grid_size=config.grid_size,
            chart_version=1,
        )
        if chart_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["identity"] != identity or sha_file(chart_path) != meta["sha256"]:
                raise ValueError("Neural chart cache does not match its source networks")
            item = load_chart(chart_path, date)
        else:
            market_path = out / "weeks" / (date + ".market.pt")
            if sha_file(market_path) != row["market_checkpoint_sha256"]:
                raise ValueError("Neural market checkpoint checksum mismatch")
            market = NeuralSurface.load(market_path).scaled(data["scale"])
            item, diagnostic = snapshot(
                model.scaled(data["scale"]), market, data, row, reference, config.grid_size
            )
            write_npz(
                chart_path,
                heston=item["heston"],
                market=item["market"],
                paired_rms=item["paired_rms"],
                starts=np.array([x["start"] for x in item["lines"]]),
                ends=np.array([x["end"] for x in item["lines"]]),
                distances=np.array([x["distance"] for x in item["lines"]]),
                days=np.array([x["day"] for x in item["lines"]]),
            )
            meta = dict(identity=identity, diagnostics=diagnostic, sha256=sha_file(chart_path))
            write_json(meta_path, meta)
            print(
                f"{date}: neural chart saved, physical half-width {diagnostic['half_width_usd_per_bbl']:.6g}",
                flush=True,
            )
        snapshots.append(item)
        details.append(meta["diagnostics"])
    positions = np.array([r["panel_index"] for r in rows])
    dates = [r["date"] for r in rows]
    smoothed = smooth_snapshots(snapshots, positions, 4.0)
    times = np.array([pd.Timestamp(d).value for d in dates], dtype=float)
    frames, labels = [], []
    for p in np.linspace(positions[0], positions[-1], len(positions)):
        hi = min(int(np.searchsorted(positions, p, side="right")), len(positions) - 1)
        lo = max(0, hi - 1)
        w = (p - positions[lo]) / max(positions[hi] - positions[lo], 1)
        a, b = smoothed[lo], smoothed[hi]
        frame = dict(slice=1, warning="", lines=[])
        for name in ("heston", "market", "heston_alpha", "market_alpha", "paired_rms"):
            frame[name] = (1 - w) * a[name] + w * b[name]
        for x, y in zip(a["lines"], b["lines"], strict=True):
            line = {
                name: (1 - w) * np.asarray(x[name]) + w * np.asarray(y[name])
                for name in ("start", "end", "distance", "visible")
            }
            line["day"] = x["day"]
            frame["lines"].append(line)
        frames.append(frame)
        labels.append(str(pd.Timestamp(int((1 - w) * times[lo] + w * times[hi])).date()))
    frames = align_reference(frames)
    duration, rate = seconds or config.seconds, fps or config.fps
    ranges = steady_ranges(frames, duration)
    destination.mkdir(parents=True, exist_ok=True)
    # Preserve the previous published visual in the existing archive, without new variant folders.
    if destination.resolve() == Path("outputs").resolve():
        with ZipFile(destination / "animation-history.zip", "a", compression=ZIP_STORED) as archive:
            for name in ("animation.mp4", "animation.html", "animation_manifest.json"):
                previous = destination / name
                entry = f"before-neural-{report['stage']}/{name}"
                if previous.exists() and entry not in archive.namelist():
                    archive.write(previous, entry)
    path = destination / "animation.html"
    write_html(
        frames,
        path,
        dates,
        labels=labels,
        hold=0,
        move=duration / (len(frames) - 1),
        reveal=0.25,
        note="2-week smoothing",
        fixed_camera=False,
        steady_surfaces=True,
        vertical_ranges=ranges,
        compact_hud=True,
        top_sheet="heston",
        show_warnings=False,
        tangent_planes=True,
        axes=["slice direction 1", "slice direction 2", "gap"],
        axis_units=["section fraction", "section fraction", "USD/bbl"],
        local_line_note="Local labels: RMS (USD/bbl)",
    )
    assets = Path("outputs/web_assets")
    (destination / "assets").mkdir(exist_ok=True)
    for name in ("three.module.js", "CSS2DRenderer.js", "LICENSE"):
        shutil.copyfile(assets / name, destination / "assets" / name)
    html = path.read_text().replace(
        "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
        "./assets/three.module.js",
    )
    html = html.replace("https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/", "./assets/")
    html = html.replace("three/addons/renderers/CSS2DRenderer.js", "three/addons/CSS2DRenderer.js")
    path.write_text(html)
    manifest = dict(
        method="neural manifolds for both sets",
        native_price_dimension=21,
        heston_latent_dimension=5,
        market_latent_dimension=5,
        heston_training_source="C++ Monte Carlo",
        heston_checkpoint_sha256=report["identity"]["heston_checkpoint_sha256"],
        heston_geometry="direct neural embedding evaluations on 2D latent sections",
        market_geometry="direct neural embedding evaluations on 2D latent sections",
        price_pca_compression=False,
        dates=dates,
        frames=len(frames),
        stage=report["stage"],
        completed_weeks=len(rows),
        total_weeks=report["total_weeks"],
        seconds=duration,
        fps=rate,
        smoothing_standard_deviation_trading_days=4,
        centered_smoothing_uses_future_snapshots=True,
        geometry_in_video_is_smoothed=True,
        unsmoothed_neural_sections_saved=True,
        common_market_tilt_removed=True,
        common_vertical_direction_reversed=True,
        reference_manifold="market",
        curvature_gain=1,
        camera="orbit",
        camera_distance_multiplier=1,
        error_reveal_start_seconds=duration * 0.25,
        tangent_reveal_start_seconds=duration * 0.5,
        numerical_error_labels="paired full 21-price RMS differences; central local projection gap",
        heston_cpp_validation=report["heston_validation"],
        vertical_ranges_usd_per_bbl=ranges,
        zoom="original widening until 3.75 s; one tighter level 5-17 s; one level held from 18.25 s to the end",
        weeks=details,
        video_complete=False,
    )
    write_json(destination / "animation_manifest.json", manifest)
    if video:
        temporary = destination / "animation.rendering.mp4"
        record_video(path, temporary, duration, rate)
        temporary.replace(destination / "animation.mp4")
        manifest["video_complete"] = True
        write_json(destination / "animation_manifest.json", manifest)
    print(f"Neural presentation saved to {destination}.", flush=True)
    return path
