"""End-to-end surrogate runs: contract-set sources -> data -> train -> evaluate -> report.

Contract-set sources (synthetic grid, June archive, later other periods) live here and
hand plain tuples of AsianContract to surrogate.py, which knows nothing about loaders.

    python -m hestonlab.surrogate_pipeline synthetic
    python -m hestonlab.surrogate_pipeline june --root /path/to/private/root

All outputs go under outputs/ (git-ignored). Nothing derived from real data is
written anywhere git would track.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import argparse
import json
import subprocess
import time
import numpy as np
import pandas as pd
from .calibration import load_june_training
from .heston import MonteCarloConfig
from .surrogate import (
    TrainConfig,
    contract_fingerprint,
    flat_direction_check,
    independent_benchmark,
    jacobian_check,
    learning_curve,
    plot_jacobian,
    plot_learning_curve,
    set_seeds,
)
from .surrogate_data import generate, save, synthetic_pilot_grid

DEFAULT_OUT = Path("outputs/surrogate")


@dataclass(frozen=True)
class PipelineConfig:
    # Powers of 2 keep Sobol point sets balanced; prefixes are nested.
    train_sizes: tuple[int, ...] = (2048, 4096, 8192, 16384)
    test_draws: int = 2048
    benchmark_points: int = 256
    benchmark_paths: int = 65536
    jacobian_points: int = 2048
    train_sobol_seed: int = 0
    test_sobol_seed: int = 1  # different Sobol scramble from training
    jacobian_sobol_seed: int = 2
    train: TrainConfig = field(default_factory=TrainConfig)


QUICK = PipelineConfig(
    train_sizes=(256, 512),
    test_draws=256,
    benchmark_points=16,
    benchmark_paths=4096,
    jacobian_points=256,
    train=TrainConfig(max_epochs=300, lr_patience=30, batch_size=32, learning_rate=1e-2),
)


# ---- contract-set sources -------------------------------------------------------------


def synthetic_contract_sets():
    return {"synthetic": synthetic_pilot_grid()}


def june_contract_sets(root):
    """{observation_date: contracts} from the private June archive. Missing prices are
    simply absent, so N can differ between dates; each date gets its own network."""
    try:
        snapshots = load_june_training(root)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"June contract sets unavailable: private data root {root} is incomplete. {exc}"
        ) from exc
    return {s.observation_date: s.contracts for s in snapshots}


def june_monte_carlo(observation_date, base=MonteCarloConfig()):
    """Same per-date stream calibrate_june uses, so the surrogate approximates exactly
    the Monte Carlo function Tim's calibrator evaluates on that date."""
    seed = base.seed + int(pd.Timestamp(observation_date).strftime("%Y%m%d"))
    return MonteCarloConfig(base.paths, base.max_step_years, seed)


# ---- pipeline -------------------------------------------------------------------------


def cached_generate(contracts, n, mc, sobol_seed, path, log=print):
    """generate(), reusing a saved file only if it was made with identical inputs."""
    path = Path(path)
    key = {
        "contract_fingerprint": contract_fingerprint(contracts),
        "monte_carlo": str(asdict(mc)),
        "sobol_seed": sobol_seed,
        "draws": n,
    }
    if path.is_file():
        data = dict(np.load(path))
        if all(str(data.get(k)) == str(v) for k, v in key.items()):
            log(f"reusing {path}")
            return data
    start = time.time()
    data = generate(contracts, n, mc, seed=sobol_seed, progress_every=max(1, n // 8))
    data.update({k: np.array(v) for k, v in key.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    save(data, path)
    log(f"generated {n} draws in {time.time() - start:.0f}s -> {path}")
    return data


def run_pipeline(contracts, out_dir, mc=MonteCarloConfig(), config=PipelineConfig(),
                 label="", log=print):
    """Generate, train (learning curve), evaluate, benchmark, Jacobian; save everything."""
    contracts, out_dir = tuple(contracts), Path(out_dir)
    set_seeds(config.train.seed)
    log(f"[{label}] {len(contracts)} contracts -> {out_dir}")
    train_data = cached_generate(
        contracts, max(config.train_sizes), mc, config.train_sobol_seed,
        out_dir / "data/train.npz", log,
    )
    test_data = cached_generate(
        contracts, config.test_draws, mc, config.test_sobol_seed, out_dir / "data/test.npz", log
    )
    log("training learning curve ...")
    curve, models = learning_curve(
        train_data, test_data, contracts, config.train_sizes, config.train
    )
    model = models[max(config.train_sizes)]
    model.metadata.update({"label": label, "sobol_seeds": {
        "train": config.train_sobol_seed, "test": config.test_sobol_seed}})
    model.save(out_dir / "model.pt")
    log("independent Monte Carlo benchmark ...")
    benchmark = independent_benchmark(
        model, contracts, test_data, config.benchmark_points, config.benchmark_paths
    )
    log("Jacobian check ...")
    jacobian, arrays = jacobian_check(model, config.jacobian_points, config.jacobian_sobol_seed)
    np.savez_compressed(out_dir / "jacobian.npz", **arrays)
    log("re-pricing along the flattest direction with Monte Carlo ...")
    flat = flat_direction_check(model, contracts, mc, paths=config.benchmark_paths)
    plot_learning_curve(curve, benchmark, out_dir / "learning_curve.png")
    plot_jacobian(arrays, out_dir / "jacobian.png")
    report = {
        "label": label,
        "status": "experimental; surrogate approximation error only, NOT Heston model error",
        "contracts": len(contracts),
        "coordinate_ids": list(model.coordinate_ids),
        "contract_fingerprint": contract_fingerprint(contracts),
        "training_monte_carlo": asdict(mc),
        "pipeline": asdict(config),
        "test_set": curve[-1],
        "learning_curve": curve,
        "independent_benchmark": benchmark,
        "jacobian": jacobian,
        "flat_direction_check": flat,
        "files": {"model": "model.pt", "data": "data/", "jacobian": "jacobian.npz",
                  "plots": ["learning_curve.png", "jacobian.png"]},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(summary(report))
    return report


def summary(report):
    t, b = report["test_set"], report["independent_benchmark"]
    j = report["jacobian"]["smallest_singular_value"]
    f = report["flat_direction_check"]["summary"]
    paths = b["benchmark_monte_carlo"][0]["paths"]

    def bench_row(name, block):
        return (
            f"  {name:<22} {block['rmse_noise_corrected_usd_per_bbl']:>10.4f} "
            f"{block['rmse_raw_usd_per_bbl']:>8.4f} {block['median_abs_error_usd_per_bbl']:>8.4f} "
            f"{block['max_abs_error_usd_per_bbl']:>8.4f}"
        )

    def decile_rows(block):
        return [
            f"    v0 [{d['v0_range'][0]:.3g}, {d['v0_range'][1]:.3g}]: "
            f"RMSE {d['rmse_noise_corrected_usd_per_bbl']:.4f}, "
            f"median {d['median_abs_error_usd_per_bbl']:.4f}"
            for d in block["by_v0_decile"]
        ]

    lines = [
        f"[{report['label']}] surrogate approximation error, USD/barrel (not model error)",
        f"  {'train draws':>11} {'test RMSE':>10} {'test max':>10}",
        *(
            f"  {r['training_draws']:>11} {r['rmse_usd_per_bbl']:>10.4f} "
            f"{r['max_abs_error_usd_per_bbl']:>10.4f}"
            for r in report["learning_curve"]
        ),
        f"  label MC SE: mean {t['label_mc_se_mean_usd_per_bbl']:.4f}, "
        f"max {t['label_mc_se_max_usd_per_bbl']:.4f}",
        f"  vs independent MC, 2 streams x {paths} paths, {b['points']} pts "
        f"(stream-to-stream RMSE {b['stream_to_stream_rmse_usd_per_bbl']:.4f}):",
        f"  {'':<22} {'RMSE corr':>10} {'raw':>8} {'median':>8} {'max':>8}",
        bench_row("surrogate", b["surrogate_vs_benchmark"]),
        bench_row("training-stream labels", b["training_stream_labels_vs_benchmark"]),
        "  surrogate by v0 decile (noise-corrected RMSE; median incl. benchmark noise):",
        *decile_rows(b["surrogate_vs_benchmark"]),
        f"  Jacobian smallest singular value: min {j['min']:.3g}, median {j['median']:.3g}",
        f"  flattest direction, step {f['step']}: median MC move {f['median_mc_move_flattest']:.4f} "
        f"(surrogate {f['median_surrogate_move_flattest']:.4f}) vs kappa-only "
        f"{f['median_mc_move_kappa_only']:.4f}, theta-only {f['median_mc_move_theta_only']:.4f}; "
        f"max stream disagreement {f['max_stream_disagreement_flattest']:.4f}",
    ]
    return "\n".join(lines)


def check_output_location(out):
    """Refuse a destination git would track, so real-data results never get committed."""
    out = Path(out).resolve()
    probe = out
    while not probe.exists():
        probe = probe.parent
    inside = subprocess.run(
        ["git", "-C", str(probe), "rev-parse", "--is-inside-work-tree"],
        capture_output=True, text=True,
    )
    if inside.returncode != 0:
        return out  # not in a repository
    ignored = subprocess.run(["git", "-C", str(probe), "check-ignore", "-q", str(out)])
    if ignored.returncode != 0:
        raise ValueError(f"{out} is tracked by git; write surrogate outputs under outputs/")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="source", required=True)
    syn = sub.add_parser("synthetic", help="SYNTHETIC development grid (no real data)")
    june = sub.add_parser("june", help="one network per June observation date")
    june.add_argument("--root", type=Path, required=True, help="private June data root")
    june.add_argument("--dates", nargs="*", help="subset of observation dates (YYYY-MM-DD)")
    for p in (syn, june):
        p.add_argument("--out", type=Path, default=None)
        p.add_argument("--quick", action="store_true", help="tiny sizes, smoke test only")
    args = parser.parse_args(argv)
    config = QUICK if args.quick else PipelineConfig()
    try:
        if args.source == "synthetic":
            out = check_output_location(args.out or DEFAULT_OUT / "synthetic")
            sets = {"synthetic": (synthetic_contract_sets()["synthetic"], MonteCarloConfig())}
        else:
            out = check_output_location(args.out or DEFAULT_OUT / "june")
            if args.root.resolve() in (out, *out.parents):
                raise ValueError("Write outputs outside the preserved data archive")
            found = june_contract_sets(args.root)
            missing = set(args.dates or ()) - set(found)
            if missing:
                raise ValueError(f"No June snapshot for {sorted(missing)}")
            sets = {d: (c, june_monte_carlo(d)) for d, c in found.items()
                    if not args.dates or d in args.dates}
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Surrogate pipeline unavailable: {exc}\n")
    reports = {}
    for name, (contracts, mc) in sets.items():
        target = out if args.source == "synthetic" else out / name
        reports[name] = run_pipeline(contracts, target, mc, config, label=name)
    return reports


if __name__ == "__main__":
    main()
