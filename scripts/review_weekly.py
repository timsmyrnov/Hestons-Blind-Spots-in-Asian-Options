#!/usr/bin/env python3
"""Summarize a saved stage report; no fitting or simulation."""

from pathlib import Path
import argparse
from collections import Counter
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "mpl_cache"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folder", type=Path, default=ROOT / "outputs/weekly_cpp/stages/first-quarter"
    )
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = args.folder
    report = json.loads((folder / "report.json").read_text())
    rows = report["rows"]
    table = pd.DataFrame(
        [
            {
                "date": r["date"],
                "gap_usd_per_bbl": r["gap_usd"],
                "gap_delta_se_usd_per_bbl": r["gap_delta_se_usd"],
                "price_noise_rms_usd_per_bbl": r["price_noise_rms_usd"],
                "market_projection_rms_usd_per_bbl": r["market_projection_rmse_usd"],
                "market_validation_rms_usd_per_bbl": r["market"]["validation_rmse_usd"],
                "market_supported": r["market_projection_supported"],
                "market_model": r["market"]["method"],
                "fit_converged": r["fit"]["converged"],
                "fit_calls": r.get("total_fit_calls", r["fit"]["calls"]),
                "screened": r["status"] == "screened",
                "flags": "; ".join(r["flags"]),
            }
            for r in rows
        ]
    )
    numerical_root = folder if (folder / "weeks").exists() else folder.parents[1]
    energy, visibility = [], []
    for row in rows:
        path = numerical_root / "weeks" / report["fingerprint"][:20] / (row["date"] + ".npz")
        with np.load(path, allow_pickle=False) as data:
            if "singular_values" in data:
                singular = data["singular_values"]
                energy.append(float(np.sum(singular[:2] ** 2) / np.sum(singular**2)))
                visibility.append(
                    float(data["market_valid_fraction"])
                    if row["market_projection_supported"]
                    else 0.0
                )
            else:
                energy.append(float("nan"))
                visibility.append(0.0)
    table["slice_two_direction_jacobian_energy_share"] = energy
    table["market_patch_visible_fraction"] = visibility
    table.to_csv(folder / "weekly_diagnostics.csv", index=False)
    flags = Counter(f for r in rows for f in r["flags"])
    detailed = [r for r in rows if r["checks"]["detailed_checks_performed"]]
    daily = report["daily_checks"]
    supported = table[table.market_supported]
    summary = {
        "stage": report.get("stage"),
        "fingerprint": report["fingerprint"],
        "completed_weeks": len(table),
        "total_weeks": report["total_weeks"],
        "first_date": table.date.iloc[0],
        "last_date": table.date.iloc[-1],
        "wall_minutes": report["elapsed_seconds"] / 60,
        "screened_weeks": int(table.screened.sum()),
        "supported_market_weeks": int(table.market_supported.sum()),
        "converged_fits": int(table.fit_converged.sum()),
        "median_gap_all_usd": float(table.gap_usd_per_bbl.median()),
        "median_gap_supported_usd": float(supported.gap_usd_per_bbl.median())
        if len(supported)
        else None,
        "median_price_noise_usd": float(table.price_noise_rms_usd_per_bbl.median()),
        "market_model_counts": dict(Counter(table.market_model)),
        "median_slice_jacobian_energy_share": float(np.nanmedian(energy))
        if np.isfinite(energy).any()
        else None,
        "median_market_patch_visible_fraction": float(np.median(visibility)),
        "flags": dict(flags),
        "detailed_check_weeks": len(detailed),
        "daily_checks": len(daily),
        "daily_checks_passed": sum(r["passed"] for r in daily),
        "fully_validated": False,
    }
    (folder / "review_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    dates = pd.to_datetime(table.date)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
    axes[0].plot(
        dates,
        table.gap_usd_per_bbl,
        color="#4d3788",
        lw=1.4,
        label="Locally fitted gap (all weeks)",
    )
    axes[0].plot(
        dates,
        2 * table.price_noise_rms_usd_per_bbl,
        color="#737373",
        ls="--",
        label="2 × RMS pricing noise",
    )
    unsupported = ~table.market_supported
    axes[0].scatter(
        dates[unsupported],
        table.loc[unsupported, "gap_usd_per_bbl"],
        marker="x",
        s=35,
        color="#c53d3d",
        label="Unsupported market projection",
        zorder=5,
    )
    axes[0].set_ylabel("RMS USD/barrel")
    axes[0].set_title(
        f"First-quarter review: {table.date.iloc[0]} to {table.date.iloc[-1]} ({len(table)} weeks)"
    )
    axes[0].legend(fontsize=9, loc="upper right")
    axes[1].plot(
        dates,
        table.market_projection_rms_usd_per_bbl,
        color="#1671a2",
        label="Observed point → market surface",
    )
    axes[1].plot(
        dates,
        table.market_validation_rms_usd_per_bbl,
        color="#858585",
        ls="--",
        label="Trailing validation reconstruction error",
    )
    axes[1].set_ylabel("RMS USD/barrel")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(folder / "diagnostics.png", dpi=170)
    plt.close(fig)

    lines = [
        "# First-quarter review",
        "",
        f"Computed **{len(table)} of {report['total_weeks']} weekly snapshots**, {table.date.iloc[0]} through {table.date.iloc[-1]}.",
        f"Elapsed time including export: **{summary['wall_minutes']:.1f} minutes**. The remaining stage has not been started.",
        "",
        "## Quantitative results",
        "",
        "| Diagnostic | Result |",
        "|---|---:|",
        f"| Weeks passing implemented screens | {summary['screened_weeks']} / {len(table)} |",
        f"| Supported market projections | {summary['supported_market_weeks']} / {len(table)} |",
        f"| Locally converged Heston fits | {summary['converged_fits']} / {len(table)} |",
        f"| Median gap, all weeks | ${summary['median_gap_all_usd']:.4f}/barrel |",
        f"| Median RMS sampling noise in validated prices | ${summary['median_price_noise_usd']:.4f}/barrel |",
        f"| Detailed time-step / derivative / mesh checks | {len(detailed)} weeks |",
        f"| Intermediate daily checks passed | {summary['daily_checks_passed']} / {len(daily)} |",
        "",
        "The gap is the full-grid RMS distance from a projected market point to a locally fitted Heston point. It is not the minimum separation of two entire manifolds. Unsupported market projections are included in the all-week median for transparency, but are not evidence of a reliable market-surface gap. Sampling diagnostics do not cover model or convention error.",
        "",
        "## Flags",
        "",
    ]
    lines += [f"- `{name}`: {count} weeks." for name, count in flags.most_common()] or [
        "No weekly flags."
    ]
    lines += [
        "",
        "## Files",
        "",
        "- [Video](animation.mp4) and [interactive page](animation.html).",
        "- [Diagnostic plot](diagnostics.png), [per-week CSV](weekly_diagnostics.csv), and [full report](report.json).",
        "",
        "The video uses centered smoothing. Its last few weeks can change when later snapshots are added; the stored unsmoothed prices and gaps do not change. A hidden market sheet indicates unsupported geometry. Screened results are not certified global optima or fully validated model estimates.",
        "",
    ]
    (folder / "review.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
