"""All cached contract-coordinate errors over time; no calibration or simulation."""

from pathlib import Path
import argparse
import base64
import csv
import io
import json

import numpy as np

from .weekly_store import sha_file, write_json


def price_errors(arrays):
    """Convert normalized Heston prices once, then subtract dollar references."""
    heston = arrays["heston_prices"] * arrays["scale"]
    fitted = arrays["market_point"]
    original = arrays["raw_market"]
    se = arrays["heston_standard_errors"] * arrays["scale"]
    values = (heston, fitted, original, se)
    if len({v.shape for v in values}) != 1 or not all(np.isfinite(v).all() for v in values):
        raise ValueError("Finite, matching price coordinates required")
    return heston, fitted, original, se, heston - fitted, heston - original


def load_cache(folder):
    folder = Path(folder)
    report = json.loads((folder / "report.json").read_text())
    grid = json.loads((folder / "preflight.json").read_text())["grid"]
    names = ("heston", "fitted", "original", "se", "fitted_error", "original_error")
    data = {name: [] for name in names}
    rows = sorted(report["rows"], key=lambda row: row["date"])
    coordinates = len(grid["tenors"]) * len(grid["log_moneyness"])
    if not rows:
        raise ValueError("No completed weekly caches")
    for row in rows:
        path = folder / "weeks" / report["fingerprint"][:20] / (row["date"] + ".npz")
        if sha_file(path) != row["arrays_sha256"]:
            raise ValueError(f"Cache checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as arrays:
            values = price_errors(arrays)
        if values[0].shape != (coordinates,):
            raise ValueError("Cache does not match the saved contract grid")
        np.testing.assert_allclose(np.sqrt(np.mean(values[4] ** 2)), row["gap_usd"], rtol=1e-10, atol=1e-12)
        for name, value in zip(names, values, strict=True):
            data[name].append(value)
    data = {name: np.asarray(values) for name, values in data.items()}
    data.update(dates=[row["date"] for row in rows], grid=grid, rows=rows, fingerprint=report["fingerprint"])
    return data


def load_neural(folder, source=Path("outputs/weekly_cpp")):
    """Same price coordinates from the neural weekly run.

    Heston is the learned Heston manifold at the local projection and the fitted
    reference is the neural market manifold, so each week's RMS equals the
    reported learned-manifold gap (the number in the movie and the review).
    """
    folder, source = Path(folder), Path(source)
    report = json.loads((folder / "report.json").read_text())
    if json.loads((source / "report.json").read_text())["fingerprint"] != report["identity"]["source_fingerprint"]:
        raise ValueError("Neural run was built from a different weekly source")
    grid = json.loads((source / "preflight.json").read_text())["grid"]
    names = ("heston", "fitted", "original", "se", "fitted_error", "original_error")
    data = {name: [] for name in names}
    rows = []
    for row in sorted(report["rows"], key=lambda row: row["date"]):
        path = folder / "weeks" / (row["date"] + ".npz")
        if sha_file(path) != row["arrays_sha256"]:
            raise ValueError(f"Neural week checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as arrays:
            values = price_errors(arrays)
        if values[0].shape != (len(grid["tenors"]) * len(grid["log_moneyness"]),):
            raise ValueError("Neural week does not match the saved contract grid")
        np.testing.assert_allclose(np.sqrt(np.mean(values[4] ** 2)), row["gap_usd"], rtol=1e-10, atol=1e-12)
        for name, value in zip(names, values, strict=True):
            data[name].append(value)
        supported = "market_projection_at_latent_boundary" not in row["flags"]
        rows.append(dict(row, market_projection_supported=supported))
    if not rows:
        raise ValueError("No completed neural weeks")
    data = {name: np.asarray(values) for name, values in data.items()}
    data.update(dates=[row["date"] for row in rows], grid=grid, rows=rows,
                fingerprint=report["identity"]["heston_checkpoint_sha256"])
    return data


def date_edges(dates):
    import matplotlib.dates as mdates

    centers = mdates.date2num(np.asarray(dates, dtype="datetime64[D]"))
    if len(centers) < 2 or (np.diff(centers) <= 0).any():
        raise ValueError("At least two distinct increasing weekly dates required")
    edges = np.r_[centers[0] - (centers[1] - centers[0]) / 2,
                  (centers[:-1] + centers[1:]) / 2,
                  centers[-1] + (centers[-1] - centers[-2]) / 2]
    return centers, edges


def color_scale(data):
    from matplotlib.colors import FuncNorm, LinearSegmentedColormap

    maximum = max(float(np.abs(data[k]).max()) for k in ("fitted_error", "original_error"))
    limit = max(0.1, np.ceil(maximum * 10) / 10)
    linear = 0.04
    norm = FuncNorm((lambda x: np.arcsinh(x / linear), lambda y: linear * np.sinh(y)),
                    vmin=-limit, vmax=limit)
    cmap = LinearSegmentedColormap.from_list("signed_error", ["#175a96", "#f8f9f7", "#bb304a"], N=1024)
    return norm, cmap, float(limit), linear


def cm_font(name="cmr10.ttf"):
    import matplotlib

    return Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name


def figure(data, mode, norm, cmap):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.font_manager import FontProperties, fontManager
    from matplotlib.ticker import FixedLocator, FuncFormatter

    fontManager.addfont(cm_font())
    fontManager.addfont(cm_font("cmb10.ttf"))
    bold = FontProperties(fname=cm_font("cmb10.ttf"), size=24)
    plt.rcParams.update({"font.family": "cmr10", "font.size": 13,
                         "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
                         "axes.unicode_minus": False,
                         "axes.labelcolor": "#435164", "text.color": "#202b3b",
                         "xtick.color": "#576477", "ytick.color": "#576477",
                         "pdf.fonttype": 42})
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#fafbfc")
    fig.text(.13, .944, "Where Heston's prices differ", fontsize=24, fontproperties=bold)
    subtitle = ("Compared with prices estimated from market data" if mode == "fitted"
                else "Compared with market data")
    fig.text(.13, .906, subtitle, fontsize=14, color="#617084")
    centers, edges = date_edges(data["dates"])
    tenors, levels = data["grid"]["tenors"], data["grid"]["log_moneyness"]
    axes = []
    for panel, tenor in enumerate(tenors):
        ax = fig.add_axes([.13, .665 - .23 * panel, .80, .165])
        values = data[mode + "_error"][:, panel * len(levels):(panel + 1) * len(levels)].T
        mesh = ax.pcolormesh(edges, np.arange(len(levels) + 1), values, cmap=cmap, norm=norm,
                             shading="flat", antialiased=False, rasterized=True)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, len(levels))
        ax.set_yticks(np.arange(len(levels)) + .5,
                     [f"{100*np.exp(x):.1f}%" if x else "100%" for x in levels])
        ax.set_ylabel("Moneyness\n(strike / forward)", labelpad=16, fontsize=13)
        ax.set_title(f"Maturity: {tenor:g} years", loc="left", pad=13,
                     fontproperties=FontProperties(fname=cm_font("cmb10.ttf"), size=15))
        for y in range(1, len(levels)):
            ax.axhline(y, color="white", alpha=.45, linewidth=.6)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
        ax.tick_params(axis="both", which="major", length=0, pad=10)
        ax.tick_params(axis="x", which="minor", length=3, color="#c0c8d2")
        for year in range(int(data["dates"][0][:4]) + 1, int(data["dates"][-1][:4]) + 1):
            ax.axvline(mdates.datestr2num(f"{year}-01-01"), color="#27364c", alpha=.22, linewidth=.8)
        # Keep the first year visible even when the first observation is after Jan 1.
        year_ticks = [centers[0]] + [mdates.datestr2num(f"{year}-01-01")
                      for year in range(int(data["dates"][0][:4])+1, int(data["dates"][-1][:4])+1)]
        ax.xaxis.set_major_locator(FixedLocator(year_ticks))
        if panel < len(tenors) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        axes.append(ax)
    cax = fig.add_axes([.31, .074, .44, .018])
    bar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    ticks = [x for x in (-2, -.5, -.1, -.02, 0, .02, .1, .5, 2) if abs(x) <= norm.vmax]
    bar.set_ticks(ticks)
    bar.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    bar.ax.tick_params(labelsize=11, length=3, color="#a7b1be")
    bar.outline.set_visible(False)
    fig.text(.295, .080, "Heston lower", ha="right", va="center", fontsize=14, color="#175a96")
    fig.text(.765, .080, "Heston higher", ha="left", va="center", fontsize=14, color="#bb304a")
    fig.text(.53, .019, "Price difference ($/barrel)",
             ha="center", fontsize=13, color="#617084")
    fig.canvas.draw()
    layout = [[ax.get_position().x0, 1-ax.get_position().y1,
               ax.get_position().width, ax.get_position().height] for ax in axes]
    return fig, layout, centers, edges


def png_bytes(fig):
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=100, facecolor=fig.get_facecolor())
    return stream.getvalue()


def export_video(background, layout, centers, edges, dates, path, seconds, fps):
    from PIL import Image, ImageDraw, ImageFont
    import imageio_ffmpeg

    base = Image.open(io.BytesIO(background)).convert("RGB")
    width, height = base.size
    font = ImageFont.truetype(str(cm_font("cmb10.ttf")), 25)
    boxes = [(round(x*width), round(y*height), round((x+w)*width), round((y+h)*height)) for x,y,w,h in layout]
    future = Image.new("RGB", base.size, "#f0f3f6")
    moving_seconds = seconds - min(1.0, seconds / 10)
    frames = round(seconds * fps)
    temp = Path(path).with_name(Path(path).stem + ".rendering.mp4")
    writer = imageio_ffmpeg.write_frames(str(temp), (width, height), fps=fps, codec="libx264",
        quality=None, macro_block_size=1, output_params=["-crf", "18", "-preset", "fast", "-movflags", "+faststart"])
    writer.send(None)
    try:
        for frame in range(frames):
            fraction = min((frame / fps) / moving_seconds, 1)
            cursor = centers[0] + fraction * (centers[-1] - centers[0])
            index = min(len(dates)-1, max(0, int(np.searchsorted(edges, cursor, side="right")-1)))
            image = base.copy()
            draw = ImageDraw.Draw(image)
            for x0,y0,x1,y1 in boxes:
                reveal = round(x0 + (edges[index+1]-edges[0])/(edges[-1]-edges[0])*(x1-x0))
                if reveal < x1:
                    image.paste(future.crop((reveal,y0,x1,y1)), (reveal,y0))
                x = round(x0 + (cursor-edges[0])/(edges[-1]-edges[0])*(x1-x0))
                draw.line((x,y0,x,y1), fill="#25384d", width=2)
                draw.polygon([(x-5,y0-8),(x+5,y0-8),(x,y0-2)], fill="#25384d")
            draw.text((round(.93*width), round(.049*height)), dates[index], font=font,
                      anchor="rt", fill="#25384d")
            writer.send(np.asarray(image))
    finally:
        writer.close()
    temp.replace(path)


def export_csv(data, path):
    with Path(path).open("w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["date", "years_to_last_fixing", "log_strike_over_forward", "strike_over_forward",
                      "heston_usd_per_bbl", "fitted_market_usd_per_bbl", "original_market_usd_per_bbl",
                      "heston_minus_fitted_market_usd_per_bbl", "heston_minus_original_market_usd_per_bbl",
                      "heston_mc_standard_error_usd_per_bbl", "market_projection_supported", "flags"])
        for k, row in enumerate(data["rows"]):
            for t, tenor in enumerate(data["grid"]["tenors"]):
                for j, level in enumerate(data["grid"]["log_moneyness"]):
                    q = t * len(data["grid"]["log_moneyness"]) + j
                    out.writerow([row["date"], tenor, level, np.exp(level), data["heston"][k,q],
                        data["fitted"][k,q], data["original"][k,q], data["fitted_error"][k,q],
                        data["original_error"][k,q], data["se"][k,q], row["market_projection_supported"],
                        ";".join(row["flags"])])


def render(out, destination, seconds=15, fps=60, video=True, neural=False):
    import matplotlib.pyplot as plt

    if seconds <= 1 or fps <= 0:
        raise ValueError("Duration must exceed one second and frame rate must be positive")
    data = load_neural(out) if neural else load_cache(out)
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    norm, cmap, limit, linear = color_scale(data)
    backgrounds = {}
    for mode in ("fitted", "original"):
        fig, layout, centers, edges = figure(data, mode, norm, cmap)
        backgrounds[mode] = png_bytes(fig)
        if mode == "fitted":
            fig.savefig(target / "heatmap.png", dpi=200, facecolor=fig.get_facecolor())
            fig.savefig(target / "heatmap.pdf", dpi=200, facecolor=fig.get_facecolor())
        plt.close(fig)
    payload = {key: data[key].tolist() for key in ("heston", "fitted", "original", "se", "fitted_error", "original_error")}
    payload.update(dates=data["dates"], grid=data["grid"], layout=layout,
                   centers=centers.tolist(), edges=edges.tolist(), seconds=seconds,
                   images={key: "data:image/png;base64,"+base64.b64encode(value).decode() for key,value in backgrounds.items()})
    template = Path(__file__).with_name("weekly_heatmap.html").read_text()
    html = template.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    for marker, name in (("__CMR__", "cmr10.ttf"), ("__CMB__", "cmb10.ttf")):
        html = html.replace(marker, base64.b64encode(cm_font(name).read_bytes()).decode())
    (target / "heatmap.html").write_text(html)
    export_csv(data, target / "heatmap_data.csv")
    manifest = dict(fingerprint=data["fingerprint"], weeks=len(data["dates"]), coordinates=len(data["heston"][0]),
        first_date=data["dates"][0], last_date=data["dates"][-1], seconds=seconds, fps=fps,
        frames=round(seconds*fps), smoothing=False, residual_sign="Heston minus reference",
        price_units="USD/bbl", color_transform="asinh(error / 0.04)",
        color_limit_usd_per_bbl=limit, color_linear_scale_usd_per_bbl=linear,
        color_scale_shared_across_modes_panels_and_dates=True, clipped_values=0,
        font="Computer Modern", moneyness_display="100 * strike / average forward (%)",
        default_reference="fitted market manifold", alternative_reference="original interpolated market grid",
        same_heston_fit_in_both_views=True, projected_dimensions=0, coordinates_are_canonical_contract_prices=True,
        source_cache_checksums_verified=True, aggregate_errors_match_cached_gaps=True,
        simulation_calls=0, calibration_calls=0, video_complete=False)
    if neural:
        manifest.update(source="neural manifolds", heston_prices="learned Heston manifold at the local projection",
                        fitted_reference="learned neural market manifold (120 prior weeks)",
                        fingerprint_kind="neural Heston checkpoint sha256")
    write_json(target / "heatmap_manifest.json", manifest)
    print(f"Prepared {len(data['dates'])} weeks × {len(data['heston'][0])} coordinates; figure and interactive view saved.", flush=True)
    if video:
        export_video(backgrounds["fitted"], layout, centers, edges, data["dates"], target / "heatmap.mp4", seconds, fps)
        manifest["video_complete"] = True
        write_json(target / "heatmap_manifest.json", manifest)
        print(f"Saved {target / 'heatmap.mp4'} ({seconds:g}s, {fps} fps).", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--neural", action="store_true", help="use the neural weekly run")
    parser.add_argument("--destination", type=Path, default=Path("outputs"))
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    out = args.out or Path("outputs/neural_weekly" if args.neural else "outputs/weekly_cpp")
    render(out, args.destination, args.seconds, args.fps, not args.no_video, args.neural)


if __name__ == "__main__":
    main()
