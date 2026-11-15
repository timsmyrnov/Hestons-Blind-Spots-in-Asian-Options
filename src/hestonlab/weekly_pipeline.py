"""Prepare, validate and execute the bounded weekly C++ Heston workflow.

`prepare` never calibrates the four-year archive. `run` is the explicit production
entry point. Completed numerical snapshots are atomic, resumable checkpoints.
"""

from dataclasses import asdict
from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from .cpp_pricer import CppPricer
from .direct_fit import Budget, DeadlineExceeded, encode, fit
from .heston import HestonParameters
from .market_grid import GridSpec, build_panel, grid_scale, heston_grid_contracts, load_archive_days
from .market_manifold import fit_market
from .weekly_geometry import gap_statistics, geometry_checks, independent_estimate, patches, rms
from .weekly_store import (
    Config,
    identity,
    output_root,
    sha_file,
    weekly_indices,
    write_json,
    write_npz,
)


def prepare(root, out, config):
    root, out = output_root(root, out)
    print("Preparing daily market grid and C++ benchmark; no production fits.", flush=True)
    first = str(
        (pd.Timestamp(config.start) - pd.Timedelta(days=max(400, config.history * 3))).date()
    )
    source_paths = [
        root / "ao_calibration_inputs_provisional.parquet",
        root / "ao_model_fixing_inputs.parquet",
    ]
    sources = {str(p): sha_file(p) for p in source_paths}
    grid = GridSpec()
    days = load_archive_days(root, first, config.end)
    panel = build_panel(days, grid)
    contracts = {date: c for date, c, _ in days}
    scales = np.array([grid_scale(contracts[d], grid) for d in panel.dates])
    indices = weekly_indices(panel.dates, config)
    if not len(indices):
        raise ValueError("No eligible weeks with enough strictly prior history")
    write_npz(
        out / "panel.npz",
        dates=np.array(panel.dates),
        vectors=panel.vectors,
        scales=scales,
        weekly_indices=indices,
    )
    write_json(out / "config.json", asdict(config))
    pricer = CppPricer(heston_grid_contracts(grid))
    benchmark = pricer.benchmark()
    fingerprint, contents = identity(config, sha_file(out / "panel.npz"), pricer.build_info)
    report = preparation_report(config, len(indices), benchmark)
    report.update(
        status="prepared_not_started",
        fingerprint=fingerprint,
        identity=contents,
        input_sources=sources,
        source_root=str(root),
        grid=asdict(grid),
        dates=[panel.dates[i] for i in indices],
        daily_observations=len(panel.dates),
        dropped_incomplete=list(panel.incomplete),
        synthetic=False,
    )
    write_json(out / "preflight.json", report)
    print(
        json.dumps(
            {k: report[k] for k in ("status", "weeks", "forecast_minutes", "within_budget")},
            indent=2,
        )
    )
    return report


def preparation_report(config, weeks, benchmark):
    timings = {r["paths"]: r["max_seconds"] for r in benchmark["timings"]}

    def seconds(paths):
        # The standard benchmark includes these exact production path counts.
        nearest = min(timings, key=lambda n: abs(n - paths))
        return timings[nearest] * paths / nearest

    fitting = weeks * config.fit_calls * seconds(config.fit_paths)
    surfaces = weeks * (config.surface_grid**2 + 10) * seconds(config.surface_paths)
    checking = weeks * config.validation_batches * seconds(config.validation_paths) + np.ceil(
        weeks / config.step_check_every
    ) * (
        config.validation_batches * 2 * seconds(config.validation_paths)
        + 13 * seconds(config.surface_paths)
    )
    refinements = min(24, weeks) * config.refinement_calls * seconds(config.refinement_paths)
    daily = min(config.daily_checks, weeks) * min(300, config.fit_calls) * seconds(config.fit_paths)
    # Conservative planning margin includes Python, market fitting, higher-path
    # acceptance checks, serialization, and sustained-load variation.
    forecast = (
        1.2 * (fitting + surfaces + checking + refinements + daily) + config.render_reserve_seconds
    )
    return {
        "weeks": weeks,
        "benchmark": benchmark,
        "forecast_minutes": float(forecast / 60),
        "within_budget": bool(forecast <= config.budget_seconds),
        "forecast_kind": "call-cap extrapolation with 20 percent margin; not a guarantee",
        "component_minutes": {
            "fitting": fitting / 60,
            "surfaces": surfaces / 60,
            "checking": checking / 60,
            "refinement_allowance": refinements / 60,
            "daily_checks": daily / 60,
            "render_reserve": config.render_reserve_seconds / 60,
        },
    }


def load_prepared(out):
    out = Path(out)
    preflight = json.loads((out / "preflight.json").read_text())
    config = Config.read(out / "config.json")
    pricer = CppPricer(heston_grid_contracts(GridSpec()))
    fingerprint, _ = identity(config, sha_file(out / "panel.npz"), pricer.build_info)
    if fingerprint != preflight["fingerprint"]:
        raise ValueError(
            "Prepared sources/configuration/data changed: rerun prepare before production"
        )
    for name, digest in preflight.get("input_sources", {}).items():
        if sha_file(name) != digest:
            raise ValueError(f"Archive input changed: rerun prepare ({name})")
    with np.load(out / "panel.npz", allow_pickle=False) as data:
        panel = {key: data[key] for key in data.files}
    return config, preflight, pricer, panel


def model_for_day(panel, index, config):
    prior = slice(index - config.history, index)
    scale = panel["scales"][index]
    dates = panel["dates"][prior].tolist()
    if len(dates) != config.history or not all(d < panel["dates"][index] for d in dates):
        raise ValueError("Market training must use the full strictly prior history window")
    model = fit_market(
        panel["vectors"][prior] * scale,
        dates,
        rank=config.market_rank,
        validation=config.market_validation,
    )
    projection = model.project(panel["vectors"][index] * scale)
    return model, projection


def starts_for(previous, config, date):
    seed = config.seed + int(str(date).replace("-", ""))
    rng = np.random.default_rng(seed)
    candidates = [previous if previous is not None else encode([0.04, 2, 0.04, 0.5, -0.5])]
    if config.fit_starts > 1:
        candidates.append(encode([0.09, 1, 0.06, 0.8, -0.3]))
    while len(candidates) < config.fit_starts:
        candidates.append(rng.uniform(0.15, 0.85, 5))
    return candidates


def compute_week(
    pricer,
    panel,
    index,
    config,
    deadline,
    previous=None,
    previous_axes=None,
    detailed=False,
    allow_refinement=True,
):
    started = time.monotonic()
    date = str(panel["dates"][index])
    print(f"{date}: fitting direct Monte Carlo", flush=True)
    model, projection = model_for_day(panel, index, config)
    scale = panel["scales"][index]
    target = projection["point"] / scale
    flags = []
    if not projection["supported"]:
        flags.append("market_target_outside_supported_interior")
    if not projection["success"]:
        flags.append("market_projection_not_converged")
    if (projection["point"] < 0).any():
        flags.append("market_projection_negative_price")
        projection["supported"] = False
    fitted = fit(
        pricer,
        target,
        scale,
        starts_for(previous, config, date),
        paths=config.fit_paths,
        seed=config.seed,
        max_step=config.max_step_days / 365,
        max_calls=config.fit_calls,
        deadline=deadline,
    )
    initial_fit_calls = fitted["calls"]
    date_seed = config.seed + int(date.replace("-", "")) * 10
    estimate = independent_estimate(pricer, fitted["unit"], config, date_seed + 100000, deadline)
    initial_shift = rms((fitted["prices"] - estimate.prices) * scale)
    tolerance = max(
        config.absolute_tolerance_usd,
        config.relative_tolerance * rms((target - estimate.prices) * scale),
    )
    comparison_noise = rms(np.hypot(fitted["standard_errors"], estimate.standard_errors) * scale)
    refined = False
    refine_report = None
    if initial_shift > tolerance + 2 * comparison_noise or not fitted["converged"] or detailed:
        if allow_refinement:
            candidate = fit(
                pricer,
                target,
                scale,
                [fitted["unit"]],
                paths=config.refinement_paths,
                seed=config.seed + 50000,
                max_step=config.max_step_days / 365,
                max_calls=config.refinement_calls,
                deadline=deadline,
            )
            # Compare on common independent draws; final reporting uses another seed.
            challenger = independent_estimate(
                pricer, candidate["unit"], config, date_seed + 100000, deadline
            )
            improved = rms((target - challenger.prices) * scale) < rms(
                (target - estimate.prices) * scale
            )
            refine_report = {
                "calls": candidate["calls"],
                "accepted": improved,
                "converged": candidate["converged"],
            }
            if improved:
                fitted = candidate
            refined = True
            estimate = independent_estimate(
                pricer, fitted["unit"], config, date_seed + 200000, deadline
            )
        else:
            flags.append("refinement_needed_budget_reserved_for_other_weeks")
    if not fitted["converged"]:
        flags.append("heston_local_search_not_converged")
    final_shift = rms((fitted["prices"] - estimate.prices) * scale)
    comparison_noise = rms(np.hypot(fitted["standard_errors"], estimate.standard_errors) * scale)
    if final_shift > tolerance + 3 * comparison_noise:
        flags.append("fit_prices_not_stable_at_validation_precision")
    statistics = gap_statistics(target, estimate, scale)
    if statistics["price_noise_rms_usd"] > config.absolute_tolerance_usd:
        flags.append("monte_carlo_noise_exceeds_tolerance")
    if not statistics["noise_resolved"]:
        flags.append("gap_not_resolved_above_sampling_noise")
    geometry, geometry_flags = patches(
        pricer, fitted["unit"], model, projection["unit"], scale, config, deadline, previous_axes
    )
    notes = [f for f in geometry_flags if f == "market_patch_reduced_to_supported_scale"]
    flags += [f for f in geometry_flags if f not in notes]
    checks = {"detailed_checks_performed": detailed}
    if detailed:
        half = independent_estimate(
            pricer, fitted["unit"], config, date_seed + 300000, deadline, half_step=True
        )
        discrepancy = rms((estimate.prices - half.prices) * scale)
        noise = rms(np.hypot(estimate.standard_errors, half.standard_errors) * scale)
        checks.update(
            half_step_difference_rms_usd=discrepancy, half_step_comparison_noise_rms_usd=noise
        )
        if discrepancy > config.absolute_tolerance_usd + 3 * noise:
            flags.append("time_discretization_check_failed")
        more, more_flags = geometry_checks(
            pricer, fitted["unit"], geometry, scale, config, deadline
        )
        checks.update(more)
        flags += more_flags
    summary = {
        "date": date,
        "panel_index": int(index),
        "status": "needs_review" if flags else "screened",
        "flags": flags,
        **statistics,
        "fit": fitted,
        "initial_fit_calls": initial_fit_calls,
        "total_fit_calls": initial_fit_calls + (refine_report["calls"] if refine_report else 0),
        "elapsed_seconds": time.monotonic() - started,
        "market": model.metadata,
        "market_projection_supported": projection["supported"],
        "market_projection_rmse_usd": projection["distance"],
        "raw_market_to_fitted_heston_rmse_usd": rms(
            (panel["vectors"][index] - estimate.prices) * scale
        ),
        "initial_fit_validation_shift_usd": initial_shift,
        "final_fit_validation_shift_usd": final_shift,
        "refined": refined,
        "fit_validation_comparison_noise_rms_usd": comparison_noise,
        "notes": notes,
        "refinement": refine_report,
        "checks": checks,
        "interpretation": "local point-to-Heston distance; not whole-manifold separation",
        "fully_validated": False,
    }
    arrays = {
        "unit": fitted["unit"],
        "parameters": fitted["parameters"],
        "heston_prices": estimate.prices,
        "heston_standard_errors": estimate.standard_errors,
        "heston_covariance": estimate.covariance,
        "market_point": projection["point"],
        "raw_market": panel["vectors"][index] * scale,
        "scale": scale,
        "gap_usd": np.array(statistics["gap_usd"]),
        "market_supported": np.array(projection["supported"]),
        **geometry,
    }
    return summary, arrays


def cache_paths(out, fingerprint, date):
    folder = Path(out) / "weeks" / fingerprint[:20]
    return folder / f"{date}.json", folder / f"{date}.npz"


def cached_week(out, fingerprint, date):
    meta, data = cache_paths(out, fingerprint, date)
    if not (meta.exists() and data.exists()):
        return None
    summary = json.loads(meta.read_text())
    if summary.get("fingerprint") != fingerprint or summary.get("arrays_sha256") != sha_file(data):
        return None
    with np.load(data, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return summary, arrays


def save_week(out, fingerprint, summary, arrays):
    meta, data = cache_paths(out, fingerprint, summary["date"])
    write_npz(data, **arrays)
    summary.update(fingerprint=fingerprint, arrays_sha256=sha_file(data))
    write_json(meta, summary)  # metadata is the commit marker, written last


def daily_spot_checks(out, preflight, panel, config, pricer, records, deadline):
    """Check intermediate days in the largest-move intervals against weekly interpolation."""
    if config.daily_checks == 0 or len(records) < 2:
        return []
    scored = []
    for a, b in zip(records, records[1:]):
        lo, hi = a["panel_index"], b["panel_index"]
        if hi - lo < 2:
            continue
        daily = panel["vectors"][lo : hi + 1] * panel["scales"][lo : hi + 1]
        changes = np.sqrt(np.mean(np.diff(daily, axis=0) ** 2, axis=1))
        index = min(hi - 1, lo + 1 + int(np.argmax(changes)))
        scored.append((float(changes.max()), a, b, index))
    selected = sorted(scored, key=lambda row: row[0], reverse=True)[: config.daily_checks]
    result = []
    for _, a, b, index in selected:
        Budget(deadline, 1).check()
        _, projection = model_for_day(panel, index, config)
        scale = panel["scales"][index]
        previous = np.asarray(a["fit"]["unit"])
        fitted = fit(
            pricer,
            projection["point"] / scale,
            scale,
            [previous],
            paths=config.fit_paths,
            seed=config.seed,
            max_calls=min(300, config.fit_calls),
            deadline=deadline,
            max_step=config.max_step_days / 365,
        )
        reference = independent_estimate(
            pricer, fitted["unit"], config, config.seed + 400000 + int(index) * 10, deadline
        )
        actual = gap_statistics(projection["point"] / scale, reference, scale)
        weight = (index - a["panel_index"]) / (b["panel_index"] - a["panel_index"])
        interpolated = (1 - weight) * a["gap_usd"] + weight * b["gap_usd"]
        difference = abs(actual["gap_usd"] - interpolated)
        tolerance = max(
            config.absolute_tolerance_usd, config.relative_tolerance * actual["gap_usd"]
        )
        result.append(
            {
                "date": str(panel["dates"][index]),
                "left": a["date"],
                "right": b["date"],
                "daily_gap_usd": actual["gap_usd"],
                "weekly_interpolated_gap_usd": interpolated,
                "difference_usd": difference,
                "sampling_se_usd": actual["gap_delta_se_usd"],
                "local_fit_converged": fitted["converged"],
                "market_supported": projection["supported"],
                "passed": bool(
                    difference <= tolerance + 3 * actual["gap_delta_se_usd"]
                    and fitted["converged"]
                    and projection["supported"]
                ),
            }
        )
        write_json(
            Path(out) / "daily_checks.json",
            {
                "fingerprint": preflight["fingerprint"],
                "rows": result,
                "selection": "largest daily moves between computed weekly anchors; retrospective",
                "kind": "unsmoothed spot check; not a proof of equivalence of smoothed movies",
            },
        )
    return result


def worker(out, no_video=False):
    out = Path(out)
    config, preflight, pricer, panel = load_prepared(out)
    started = time.monotonic()
    # Parent supplies remaining global time, including worker startup and input checks.
    available = float(os.environ.get("HESTON_RUN_REMAINING", config.budget_seconds))
    deadline = started + available - config.render_reserve_seconds
    records, previous, axes = [], None, None
    refinement_count = 0
    status = "running"
    error = None
    daily = []
    try:
        for number, index in enumerate(panel["weekly_indices"]):
            Budget(deadline, 1).check()
            date = str(panel["dates"][index])
            cached = cached_week(out, preflight["fingerprint"], date)
            if cached:
                summary, arrays = cached
                print(f"{date}: verified cache", flush=True)
            else:
                summary, arrays = compute_week(
                    pricer,
                    panel,
                    int(index),
                    config,
                    deadline,
                    previous,
                    axes,
                    detailed=number % config.step_check_every == 0,
                    allow_refinement=refinement_count < 24,
                )
                save_week(out, preflight["fingerprint"], summary, arrays)
            records.append(summary)
            previous = arrays["unit"]
            axes = arrays.get("axes", axes)
            refinement_count += bool(summary["refined"])
            write_json(
                out / "progress.json",
                {
                    "fingerprint": preflight["fingerprint"],
                    "completed_weeks": len(records),
                    "total_weeks": len(panel["weekly_indices"]),
                    "last_date": date,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
        daily = daily_spot_checks(out, preflight, panel, config, pricer, records, deadline)
        status = (
            "computed_with_flags"
            if any(r["flags"] for r in records) or any(not r["passed"] for r in daily)
            else "computed_screened"
        )
    except DeadlineExceeded as exc:
        status, error = "partial_deadline", str(exc)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    report = {
        "status": status,
        "error": error,
        "fingerprint": preflight["fingerprint"],
        "completed_weeks": len(records),
        "total_weeks": len(panel["weekly_indices"]),
        "elapsed_seconds": time.monotonic() - started,
        "rows": records,
        "daily_checks": daily,
        "fully_validated": False,
        "synthetic": preflight.get("synthetic", False),
        "metric": "RMS USD/barrel; projected market point to locally fitted Heston",
        "rendering": "pending",
    }
    write_json(out / "report.json", report)
    if len(records) >= 2 and status != "failed":
        try:
            from .manifold_movie import render

            render(out, video=not no_video)
            report["rendering"] = "html_only" if no_video else "complete"
        except Exception as exc:
            report["rendering"] = "failed"
            report["render_error"] = f"{type(exc).__name__}: {exc}"
    report["elapsed_seconds"] = time.monotonic() - started
    write_json(out / "report.json", report)
    return 0 if status.startswith("computed") and report["rendering"] != "failed" else 2


def run(out, no_video=False):
    """Explicit execution entry point; a parent watchdog bounds the entire worker group."""
    out = Path(out).resolve()
    config = Config.read(out / "config.json")
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Run lock exists: {lock}. Verify no active process before removing a stale lock."
        ) from exc
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    command = [sys.executable, "-m", "hestonlab.weekly_pipeline", "_worker", "--out", str(out)]
    if no_video:
        command.append("--no-video")
    environment = os.environ.copy()
    environment["HESTON_RUN_REMAINING"] = str(config.budget_seconds)
    started = time.monotonic()
    try:
        with (out / "run.log").open("a") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=config.budget_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = 124
                write_json(
                    out / "deadline.json",
                    {
                        "status": "hard_deadline_reached",
                        "elapsed_seconds": time.monotonic() - started,
                        "message": "Worker and render subprocesses stopped; completed week caches remain resumable",
                    },
                )
                # Never leave a previous successful report looking like this run
                # completed after the watchdog interrupted it.
                report_path = out / "report.json"
                report = (
                    json.loads(report_path.read_text())
                    if report_path.exists()
                    else {"rows": [], "completed_weeks": 0, "total_weeks": 0}
                )
                report.update(
                    status="partial_hard_deadline",
                    rendering="interrupted",
                    elapsed_seconds=time.monotonic() - started,
                    fully_validated=False,
                )
                write_json(report_path, report)
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        write_json(
            out / "execution.json",
            {
                "exit_code": code,
                "elapsed_seconds": time.monotonic() - started,
                "status": "deadline" if code == 124 else "finished",
            },
        )
        return code
    finally:
        lock.unlink(missing_ok=True)


def smoke(out, video=False):
    """Tiny synthetic end-to-end test; never opens the licensed market archive."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = Config(
        start="2024-03-01",
        end="2024-03-08",
        history=40,
        market_validation=10,
        fit_paths=128,
        fit_calls=36,
        surface_paths=256,
        surface_grid=3,
        validation_paths=512,
        validation_batches=2,
        refinement_paths=256,
        refinement_calls=24,
        step_check_every=1,
        daily_checks=1,
        seconds=1,
        fps=6,
        budget_seconds=120,
        render_reserve_seconds=45,
    )
    grid = GridSpec()
    dates = np.array([str(d.date()) for d in pd.bdate_range("2024-01-01", config.end)])
    pricer = CppPricer(heston_grid_contracts(grid))
    base = pricer.price(HestonParameters(), paths=4096).prices
    rng = np.random.default_rng(51)
    factors = rng.normal(size=(len(dates), 5)) * 0.0005
    basis = np.linalg.qr(rng.normal(size=(grid.size, 5)))[0]
    vectors = np.maximum(base + factors @ basis.T, 1e-8)
    scales = np.full_like(vectors, 80.0)
    write_npz(
        out / "panel.npz",
        dates=dates,
        vectors=vectors,
        scales=scales,
        weekly_indices=weekly_indices(dates, config),
    )
    write_json(out / "config.json", asdict(config))
    fingerprint, contents = identity(config, sha_file(out / "panel.npz"), pricer.build_info)
    write_json(
        out / "preflight.json",
        {
            "status": "synthetic_smoke_prepared",
            "fingerprint": fingerprint,
            "identity": contents,
            "synthetic": True,
            "input_sources": {},
            "grid": asdict(grid),
        },
    )
    return run(out, no_video=not video)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser(
        "prepare", help="compile, prepare inputs and forecast; no production fit"
    )
    prepare_parser.add_argument("--root", type=Path, required=True)
    prepare_parser.add_argument("--config", type=Path, default=Path("configs/weekly_cpp.json"))
    prepare_parser.add_argument("--out", type=Path, default=Path("outputs/weekly_cpp"))
    for name in ("run", "_worker"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, default=Path("outputs/weekly_cpp"))
        p.add_argument("--no-video", action="store_true")
    p = sub.add_parser("smoke", help="small synthetic pipeline, not the archive")
    p.add_argument("--out", type=Path, default=Path("outputs/weekly_cpp_smoke"))
    p.add_argument("--video", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args.root, args.out, Config.read(args.config))
            return 0
        if args.command == "smoke":
            return smoke(args.out, args.video)
        return (worker if args.command == "_worker" else run)(args.out, args.no_video)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
