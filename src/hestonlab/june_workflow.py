"""Local June integration evidence around the existing, unchanged Heston engine."""

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from . import calibration, heston
from .calibration import load_june_training, calibrate_june
from .heston import HestonParameters, MonteCarloConfig, price_asian_options


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_output_root(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out == root or root in out.parents:
        raise ValueError("Experiment output must be outside the preserved data archive")
    return root, out


def price_snapshots(snapshots, parameters, config):
    rows = []
    for s in snapshots:
        seed = config.seed + int(pd.Timestamp(s.observation_date).strftime("%Y%m%d"))
        cfg = MonteCarloConfig(config.paths, config.max_step_years, seed)
        estimates = price_asian_options(s.contracts, parameters, cfg)
        for contract, observed, estimate in zip(s.contracts, s.market_prices, estimates):
            rows.append(
                dict(
                    observation_date=s.observation_date,
                    contract_id=contract.coordinate_id,
                    observed_price=observed,
                    model_price=estimate.price,
                    model_standard_error=estimate.standard_error,
                    residual=observed - estimate.price,
                    absolute_error=abs(observed - estimate.price),
                    relative_error=(observed - estimate.price) / observed,
                    daily_simulation_seed=seed,
                )
            )
    frame = pd.DataFrame(rows)
    if frame.duplicated(["observation_date", "contract_id"]).any():
        raise ValueError("Duplicate prediction identity")
    if not np.isfinite(frame[["model_price", "model_standard_error", "residual"]]).all().all():
        raise ValueError("Nonfinite model outputs")
    if (frame.model_price < 0).any():
        raise ValueError("Negative option premium")
    return frame


def error_metrics(frame):
    r = frame.residual.to_numpy()
    return dict(
        observations=len(r),
        mae=float(np.abs(r).mean()),
        rmse=float(np.sqrt(np.mean(r * r))),
        median_absolute_error=float(np.median(np.abs(r))),
        mean_signed_residual=float(r.mean()),
        max_absolute_error=float(np.max(np.abs(r))),
        absolute_error_p95=float(np.quantile(np.abs(r), 0.95)),
        max_price_standard_error=float(frame.model_standard_error.max()),
        fraction_abs_residual_gt_two_mc_se=float(
            (frame.absolute_error > 2 * frame.model_standard_error).mean()
        ),
    )


def enrich_metadata(frame, root):
    meta = pd.read_parquet(
        root / "pilot_june_july_2024/train_fixed_coordinate_observations.parquet"
    )
    meta["observation_date"] = pd.to_datetime(meta.observation_date).dt.strftime("%Y-%m-%d")
    cols = [
        "observation_date",
        "contract_id",
        "strike",
        "call_put",
        "contract_month",
        "expiration_date",
        "discount_factor",
        "known_fixing_sum",
        "known_fixing_count",
        "remaining_fixings",
        "total_fixings",
        "expected_average_from_cl",
        "sofr_percent",
        "rate_effective_date",
    ]
    data = frame.merge(
        meta[cols], on=["observation_date", "contract_id"], how="left", validate="one_to_one"
    )
    if data.strike.isna().any():
        raise ValueError("Prediction lacks contract metadata")
    # Only June metadata/quality rows are read, never July option labels.
    obs = pd.read_parquet(
        root / "ao_observations.parquet",
        columns=[
            "observation_date",
            "contract_id",
            "price_type",
            "stat_flags",
            "trade_volume",
            "open_interest",
            "nominal_indicator",
            "source_file",
            "source_row",
        ],
        filters=[
            ("observation_date", ">=", pd.Timestamp("2024-06-01").date()),
            ("observation_date", "<=", pd.Timestamp("2024-06-30").date()),
        ],
    )
    obs["observation_date"] = pd.to_datetime(obs.observation_date).dt.strftime("%Y-%m-%d")
    data = data.merge(
        obs, on=["observation_date", "contract_id"], how="left", validate="one_to_one"
    )
    if data.price_type.isna().any():
        raise ValueError("Prediction lacks source quality identity")
    data["volume_category"] = np.where(
        data.trade_volume.isna(), "missing", np.where(data.trade_volume.eq(0), "zero", "positive")
    )
    data["is_otm"] = np.where(
        data.call_put.eq("C"),
        data.strike > data.expected_average_from_cl,
        data.strike < data.expected_average_from_cl,
    )
    data["absolute_strike_distance_fraction"] = abs(data.strike / data.expected_average_from_cl - 1)
    return data


def export_comparison(root, out, snapshots, result):
    params = HestonParameters(**result["parameters"])
    cfg = MonteCarloConfig(**result["monte_carlo"])
    fitted = price_snapshots(snapshots, params, cfg)
    if not np.isclose(
        error_metrics(fitted)["rmse"], result["training_rmse"], atol=1e-12, rtol=1e-12
    ):
        raise ValueError("Repriced fitted RMSE differs from the saved calibration")
    fresh = price_snapshots(
        snapshots, params, MonteCarloConfig(cfg.paths, cfg.max_step_years, cfg.seed + 1000000)
    )
    if not np.isclose(
        error_metrics(fresh)["rmse"], result["independent_seed_rmse"], atol=1e-12, rtol=1e-12
    ):
        raise ValueError("Independent-seed RMSE differs from the saved calibration")
    data = enrich_metadata(fitted, root)
    additional = fresh[
        ["observation_date", "contract_id", "model_price", "model_standard_error", "residual"]
    ].rename(
        columns={
            "model_price": "independent_model_price",
            "model_standard_error": "independent_standard_error",
            "residual": "independent_residual",
        }
    )
    data = data.merge(additional, on=["observation_date", "contract_id"], validate="one_to_one")
    data.to_csv(out / "actual_vs_model.csv", index=False)
    metrics = dict(
        fitting_seed=error_metrics(fitted),
        independent_seed=error_metrics(fresh),
        residual_definition="observed settlement minus model price, USD/barrel",
        sampling_error_warning="Monte Carlo SE excludes time-step bias, parameter uncertainty and settlement uncertainty; prices share simulation paths.",
        in_sample=True,
        july_labels_used=False,
    )
    (out / "residual_summary.json").write_text(json.dumps(metrics, indent=2))
    for field in [
        "contract_month",
        "call_put",
        "price_type",
        "volume_category",
        "is_otm",
        "observation_date",
    ]:
        rows = [
            dict(category=str(k), **error_metrics(g)) for k, g in data.groupby(field, dropna=False)
        ]
        pd.DataFrame(rows).to_csv(out / f"errors_by_{field}.csv", index=False)
    return data, metrics


def sensitivity(snapshots, result, out):
    cfg = MonteCarloConfig(**result["monte_carlo"])
    params = HestonParameters(**result["parameters"])
    frames = []
    reports = []
    cases = [
        ("double_paths", MonteCarloConfig(cfg.paths * 2, cfg.max_step_years, cfg.seed)),
        ("half_time_step", MonteCarloConfig(cfg.paths, cfg.max_step_years / 2, cfg.seed)),
        (
            "second_independent_seed",
            MonteCarloConfig(cfg.paths, cfg.max_step_years, cfg.seed + 2000000),
        ),
    ]
    for name, case in cases:
        start = time.monotonic()
        f = price_snapshots(snapshots, params, case)
        f["scenario"] = name
        frames.append(f)
        reports.append(
            dict(
                scenario=name,
                **asdict(case),
                runtime_seconds=time.monotonic() - start,
                **error_metrics(f),
            )
        )
    pd.concat(frames, ignore_index=True).to_csv(
        out / "fixed_parameter_sensitivity_prices.csv", index=False
    )
    pd.DataFrame(reports).to_csv(out / "fixed_parameter_sensitivity_summary.csv", index=False)
    (out / "sensitivity_limitations.txt").write_text(
        "Fixed fitted parameters; no recalibration and no July scoring. Path/time-step changes alter random-number allocation, so observed differences combine sampling noise and discretization effects. No coupled Brownian-refinement bias estimate, multistart calibration or identifiability proof is claimed.\n"
    )
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=os.environ.get("HESTONLAB_DATA_ROOT"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--source-commit", help="Verified source SHA for a downloaded code snapshot without .git"
    )
    p.add_argument("--paths", type=int, default=4096)
    p.add_argument("--max-evaluations", type=int, default=100)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--max-step-days", type=float, default=1)
    p.add_argument("--sensitivity", action="store_true")
    args = p.parse_args()
    if args.root is None:
        p.error("Set HESTONLAB_DATA_ROOT or supply --root")
    root, out = validate_output_root(args.root, args.out)
    if out.exists() and any(out.iterdir()):
        p.error("Use a new experiment directory to preserve previous results")
    commit = args.source_commit
    if not commit:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
        )
        if proc.returncode == 0:
            commit = proc.stdout.strip()
    if not commit or not re.fullmatch("[0-9a-f]{40}", commit):
        p.error("Supply a verified full --source-commit for a non-Git snapshot")
    required = [
        root / "pilot_june_july_2024/train_fixed_coordinate_observations.parquet",
        root / "ao_model_fixing_inputs.parquet",
        root / "ao_contracts.parquet",
        root / "sofr_history.parquet",
        root / "ao_observations.parquet",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        p.error("Missing June inputs under the configured data root: " + ", ".join(missing))
    snapshots = load_june_training(root)
    config = MonteCarloConfig(args.paths, args.max_step_days / 365, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        started_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=commit,
        local_integration_extension=True,
        python=sys.version,
        executable=sys.executable,
        dependencies={
            n: importlib.metadata.version(n)
            for n in ["numpy", "pandas", "pyarrow", "duckdb", "databento"]
        },
        input_files=[dict(path=str(x), sha256=digest(x), bytes=x.stat().st_size) for x in required],
        pricing_module_sha256=digest(heston.__file__),
        calibration_module_sha256=digest(calibration.__file__),
        workflow_module_sha256=digest(__file__),
        initial_parameters=asdict(HestonParameters()),
        monte_carlo=asdict(config),
        max_evaluations=args.max_evaluations,
        training_dates=[s.observation_date for s in snapshots],
        observed_prices=sum(len(s.contracts) for s in snapshots),
        july_option_files_opened=False,
        output_directory=str(out),
        command=sys.argv,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    start = time.monotonic()
    print("June inputs verified; calibrating unchanged engine", flush=True)
    result = calibrate_june(snapshots, config=config, max_evaluations=args.max_evaluations)
    duration = time.monotonic() - start
    (out / "calibration.json").write_text(json.dumps(result, indent=2))
    print("Calibration saved; exporting exact seeded prices", flush=True)
    _, metrics = export_comparison(root, out, snapshots, result)
    diagnostics = sensitivity(snapshots, result, out) if args.sensitivity else []
    manifest.update(
        finished_utc=datetime.now(timezone.utc).isoformat(),
        calibration_runtime_seconds=duration,
        total_runtime_seconds=time.monotonic() - start,
        optimizer_stop_reason=result["stop_reason"],
        optimizer_step_threshold_reached=result["stop_reason"] == "step_threshold",
        fully_validated=False,
        bound_hits=[
            n
            for n, (lo, hi) in result["bounds"].items()
            if np.isclose(result["parameters"][n], lo, atol=1e-10, rtol=0)
            or np.isclose(result["parameters"][n], hi, atol=1e-10, rtol=0)
        ],
        residual_metrics=metrics,
        sensitivity_scenarios=diagnostics,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            dict(
                out=str(out),
                parameters=result["parameters"],
                training_rmse=result["training_rmse"],
                stop_reason=result["stop_reason"],
                calibration_runtime_seconds=duration,
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
