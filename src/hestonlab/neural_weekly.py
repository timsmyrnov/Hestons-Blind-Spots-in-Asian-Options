"""Resumable neural manifolds trained in the full 21-contract price space."""

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import hashlib
import json
import time
import fcntl

import numpy as np
from scipy.stats import qmc

from .calibration import LOWER, UPPER
from .cpp_pricer import CppPricer
from .direct_fit import decode
from .market_grid import GridSpec, heston_grid_contracts
from .weekly_store import sha_file, write_json, write_npz


@dataclass(frozen=True)
class NeuralConfig:
    seed: int = 17012026
    train_global: int = 8192
    train_local: int = 2048
    validation_global: int = 256
    validation_local: int = 256
    test_global: int = 256
    test_local: int = 256
    train_paths: int = 16384
    check_paths: int = 65536
    max_step: float = 1 / 365
    width: int = 64
    depth: int = 4
    heston_epochs: int = 1800
    market_epochs: int = 1600
    batch_size: int = 512
    threads: int = 2
    history: int = 120
    market_validation: int = 20
    grid_size: int = 31
    seconds: float = 26
    fps: int = 60


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def scope(source):
    report = json.loads((Path(source) / "report.json").read_text())
    rows = sorted(report["rows"], key=lambda r: r["date"])
    half = (len(rows) + 1) // 2
    return report, rows, half


def cloud_units(count_global, count_local, anchors, seed):
    exponent = int(np.ceil(np.log2(max(count_global, 1))))
    global_units = qmc.Sobol(5, scramble=True, seed=seed).random_base2(exponent)[:count_global]
    rng = np.random.default_rng(seed + 1)
    local = anchors[rng.integers(len(anchors), size=count_local)] + rng.normal(
        0, 0.045, (count_local, 5)
    )
    # These are parameter samples, never dimension-reduced price targets.
    return np.vstack((global_units, np.clip(local, 0, 1)))


def prepare_cloud(source, out, config):
    source, out = Path(source), Path(out)
    report, rows, half = scope(source)
    anchors = []
    for row in rows[:half]:
        path = source / "weeks" / report["fingerprint"][:20] / (row["date"] + ".npz")
        if sha_file(path) != row["arrays_sha256"]:
            raise ValueError(f"Source checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as z:
            anchors.append(z["unit"])
    anchors = np.asarray(anchors)
    grid = GridSpec()
    pricer = CppPricer(heston_grid_contracts(grid))
    folder = out / "heston"
    folder.mkdir(parents=True, exist_ok=True)
    identity = dict(
        grid=asdict(grid),
        lower=LOWER.tolist(),
        upper=UPPER.tolist(),
        source=report["fingerprint"],
        anchor_dates=[r["date"] for r in rows[:half]],
        anchors_sha256=hashlib.sha256(anchors.tobytes()).hexdigest(),
        cpp_build=pricer.build_info,
        config=asdict(config),
        generator_version=1,
    )
    key = digest(identity)
    manifest = folder / "cloud_manifest.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["fingerprint"] != key:
            raise ValueError("Cloud configuration changed; use a new output directory")
        if saved["status"] == "complete":
            for name, expected in saved["arrays"].items():
                if sha_file(folder / (name + ".npz")) != expected:
                    raise ValueError("Completed C++ cloud checksum mismatch")
            return key
    write_json(manifest, {"fingerprint": key, **identity, "status": "generating"})
    for split, n_global, n_local, paths, offset in (
        ("train", config.train_global, config.train_local, config.train_paths, 0),
        (
            "validation",
            config.validation_global,
            config.validation_local,
            config.check_paths,
            100000,
        ),
        ("test", config.test_global, config.test_local, config.check_paths, 200000),
    ):
        units = cloud_units(n_global, n_local, anchors, config.seed + offset)
        target = folder / (split + ".npz")
        prices = np.full((len(units), grid.size), np.nan)
        errors = prices.copy()
        if target.exists():
            with np.load(target, allow_pickle=False) as z:
                np.testing.assert_array_equal(z["units"], units)
                prices, errors = z["prices"], z["standard_errors"]
        started = time.monotonic()
        count = 0
        for i, unit in enumerate(units):
            if np.isfinite(prices[i]).all():
                continue
            estimate = pricer.price(
                decode(unit), paths=paths, seed=config.seed + offset + i, max_step=config.max_step
            )
            prices[i], errors[i] = estimate.prices, estimate.standard_errors
            count += 1
            if count % 256 == 0 or i == len(units) - 1:
                write_npz(
                    target,
                    units=units,
                    prices=prices,
                    standard_errors=errors,
                    global_count=np.array(n_global),
                    paths=np.array(paths),
                )
                print(
                    f"C++ {split}: {np.isfinite(prices).all(axis=1).sum()}/{len(units)} vectors, "
                    f"{time.monotonic() - started:.0f}s this session",
                    flush=True,
                )
        if not np.isfinite(prices).all():
            raise ValueError("Incomplete C++ price cloud")
    write_json(
        manifest,
        {
            "fingerprint": key,
            **identity,
            "status": "complete",
            "arrays": {s: sha_file(folder / (s + ".npz")) for s in ("train", "validation", "test")},
        },
    )
    print("C++ clouds complete; independent validation/test streams saved.", flush=True)
    return key


def stage_indices(total, stage, completed=0, limit=None):
    half = (total + 1) // 2
    if stage == "remaining" and completed < half:
        raise ValueError("Complete the first half before requesting the remaining half")
    stop = total if stage == "remaining" else half
    if limit is not None:
        if stage == "remaining" or not 1 <= limit <= half:
            raise ValueError("Pilot limits must stay inside the first half")
        stop = limit
    return range(stop)


def compute_snapshot(source, out, config, panel, source_row, heston, pricer, previous):
    import torch
    from .cpp_pricer import Estimate
    from .neural_training import NeuralSurface, fit_market
    from .neural_embedding import geometry_diagnostics
    from .weekly_geometry import gap_statistics

    date, index = source_row["date"], source_row["panel_index"]
    started = time.monotonic()
    dates = panel["dates"][index - config.history : index].tolist()
    if len(dates) != config.history or not all(d < date for d in dates):
        raise ValueError("Market training would use non-prior observations")
    history = panel["vectors"][index - config.history : index]
    scale = panel["scales"][index]
    history_key = hashlib.sha256(history.tobytes() + json.dumps(dates).encode()).hexdigest()
    checkpoint = out / "weeks" / (date + ".market.pt")
    print(f"{date}: neural market manifold, then full-price projections", flush=True)
    if checkpoint.exists():
        market_normalized = NeuralSurface.load(checkpoint)
        if market_normalized.metadata["history_sha256"] != history_key:
            raise ValueError("Market checkpoint history mismatch")
    else:
        market_normalized = fit_market(history, dates, config)
        market_normalized.metadata["history_sha256"] = history_key
        market_normalized.metadata["geometry"] = geometry_diagnostics(
            market_normalized.scaled(scale), market_normalized.train_units[::8]
        )
        market_normalized.save(checkpoint)
    market = market_normalized.scaled(scale)
    h = heston.scaled(scale)
    observed = panel["vectors"][index] * scale
    projection = market.project(observed)
    starts = [np.asarray(source_row["fit"]["unit"])]
    if previous is not None:
        starts.append(previous)
    fitted = h.project(projection["point"], starts=starts)
    unit = fitted["unit"]
    seed = config.seed + int(date.replace("-", "")) * 10
    estimates = [
        pricer.price(
            decode(unit),
            paths=config.check_paths,
            seed=seed + b,
            max_step=config.max_step,
            covariance=True,
        )
        for b in range(4)
    ]
    covariance = sum(e.covariance for e in estimates) / 16
    estimate = Estimate(
        np.mean([e.prices for e in estimates], axis=0),
        np.sqrt(np.maximum(np.diag(covariance), 0)),
        covariance,
    )
    physical = gap_statistics(projection["point"] / scale, estimate, scale)
    approximation = float(np.sqrt(np.mean((fitted["point"] - estimate.prices * scale) ** 2)))
    jac = h.jacobian(unit)
    free = (unit > 0.002) & (unit < 0.998)
    if free.sum() < 2:
        raise ValueError(f"{date}: fewer than two interior Heston latent directions")
    left, singular, _ = np.linalg.svd(jac[:, free] / np.sqrt(21), full_matrices=False)
    axes = left[:, :2]
    flags = []
    if not projection["success"]:
        flags.append("market_projection_not_converged")
    if projection["at_bound"]:
        flags.append("market_projection_at_latent_boundary")
    if not fitted["success"]:
        flags.append("heston_neural_projection_not_converged")
    if (projection["point"] < 0).any():
        flags.append("market_neural_price_negative")
    if approximation > max(0.02, 0.2 * fitted["distance"]) + 2 * physical["price_noise_rms_usd"]:
        flags.append("neural_heston_cpp_discrepancy")
    derivative_check = dict(performed=False)
    week_number = int(np.flatnonzero(panel["weekly_indices"] == index)[0])
    if week_number % 13 == 0:
        from .direct_fit import price_jacobian

        derivatives = [
            price_jacobian(
                pricer,
                unit,
                paths=config.check_paths,
                seed=seed + 100 + b,
                max_step=config.max_step,
            )
            * scale[:, None]
            for b in range(2)
        ]
        physical_jac = (derivatives[0] + derivatives[1]) / 2
        denominator = max(float(np.linalg.norm(physical_jac)), 1e-12)
        error = float(np.linalg.norm(jac - physical_jac) / denominator)
        repeat = float(np.linalg.norm(derivatives[0] - derivatives[1]) / denominator)
        derivative_check = dict(performed=True, relative_error=error, mc_repeat_difference=repeat)
        if error > 0.35 + repeat:
            flags.append("neural_heston_derivative_discrepancy")
    with torch.no_grad():
        if not market.metadata["geometry"]["numerically_resolved"]:
            raise ValueError("Market neural embedding is not numerically resolved")
    arrays = dict(
        unit=unit,
        parameters=decode(unit),
        heston_prices=heston.prices(unit),
        heston_cpp_prices=estimate.prices,
        heston_standard_errors=estimate.standard_errors,
        heston_covariance=covariance,
        market_point=projection["point"],
        market_unit=projection["unit"],
        raw_market=observed,
        scale=scale,
        heston_jacobian=jac,
        axes=axes,
        singular_values=singular,
        gap_usd=np.array(fitted["distance"]),
    )
    row = dict(
        date=date,
        panel_index=index,
        gap_usd=fitted["distance"],
        cpp_gap_usd=physical["gap_usd"],
        cpp_gap_delta_se_usd=physical["gap_delta_se_usd"],
        neural_cpp_rms_usd=approximation,
        cpp_noise_rms_usd=physical["price_noise_rms_usd"],
        market_projection_rmse_usd=projection["distance"],
        raw_market_neural_heston_rmse_usd=float(
            np.sqrt(np.mean((observed - fitted["point"]) ** 2))
        ),
        market=market_normalized.metadata,
        flags=flags,
        derivative_check=derivative_check,
        market_checkpoint_sha256=sha_file(checkpoint),
        native_price_dimension=21,
        heston_latent_dimension=5,
        market_latent_dimension=5,
        gap_definition="local projected market point to locally nearest learned Heston point, RMS USD/bbl",
        cpp_independent_validation_batches=4,
        cpp_independent_validation_paths=config.check_paths,
        elapsed_seconds=time.monotonic() - started,
    )
    path = out / "weeks" / (date + ".npz")
    write_npz(path, **arrays)
    row["arrays_sha256"] = sha_file(path)
    write_json(out / "weeks" / (date + ".json"), row)
    print(
        f"{date}: gap {row['gap_usd']:.5f}, independent C++ gap {row['cpp_gap_usd']:.5f}, "
        f"neural/MC discrepancy {approximation:.5f}; {row['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return row, arrays


def write_review(out, report):
    import csv
    from collections import Counter

    rows = report["rows"]
    keys = [
        "date",
        "gap_usd",
        "cpp_gap_usd",
        "neural_cpp_rms_usd",
        "cpp_noise_rms_usd",
        "market_projection_rmse_usd",
        "raw_market_neural_heston_rmse_usd",
    ]
    with (out / "review.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(keys + ["flags"])
        for row in rows:
            writer.writerow([row[k] for k in keys] + [";".join(row["flags"])])
    flags = Counter(flag for row in rows for flag in row["flags"])
    validation = report["heston_validation"]
    text = [
        "# Neural manifold review",
        "",
        f"Completed {len(rows)} of {report['total_weeks']} weekly snapshots: {rows[0]['date']} to {rows[-1]['date']}.",
        "",
        "Both models use smooth injective neural embeddings with five latent coordinates and all 21 price outputs.",
        "Heston training labels are C++ Monte Carlo prices. Each market network uses only its 120 prior observations.",
        "",
        f"Median learned-manifold gap: ${np.median([r['gap_usd'] for r in rows]):.6f}/barrel.",
        f"Median independent C++ gap at the neural fit: ${np.median([r['cpp_gap_usd'] for r in rows]):.6f}/barrel.",
        f"Median neural/C++ price discrepancy: ${np.median([r['neural_cpp_rms_usd'] for r in rows]):.6f}/barrel.",
        f"Held-out local Heston test RMS: ${validation['local_rms_usd']:.6f}/barrel.",
        f"Independent global test/geometry/derivative screening passed: {validation['screen_passed']}.",
        "",
        "The reported gap is a local point-to-manifold comparison, not the minimum separation of two entire manifolds.",
        "Neural approximation error and Monte Carlo noise can be comparable to these gaps; small gaps are not precisely resolved by passing the screening checks.",
        "The animation smooths display geometry and labels only. Numerical rows in review.csv are unsmoothed.",
        "",
        "## Numerical review flags",
        "",
    ]
    text += [f"- {key}: {value} weeks" for key, value in sorted(flags.items())] or ["None."]
    if len(rows) == report["total_weeks"]:
        text += ["", "All weekly snapshots are complete. Existing first-half checkpoints were verified and reused."]
    else:
        text += [
            "",
            "## Continuation",
            "",
            "```bash",
            report["continuation_command"],
            "```",
            "",
            "The remaining half has not been computed by this stage. Existing checkpoints are verified and reused.",
        ]
    (out / "review.md").write_text("\n".join(text) + "\n")


def run(source, out, config, stage, limit=None, video=True):
    import torch
    from .neural_training import train_heston, validate_heston

    source, out = Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "weeks").mkdir(exist_ok=True)
    with (out / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(config.threads)
        source_report, source_rows, half = scope(source)
        completed = sum((out / "weeks" / (r["date"] + ".json")).exists() for r in source_rows)
        indices = stage_indices(len(source_rows), stage, completed, limit)
        cloud_key = prepare_cloud(source, out, config)
        model = train_heston(out / "heston", config)
        if stage == "train":
            return
        with np.load(source / "panel.npz", allow_pickle=False) as z:
            panel = {k: z[k] for k in z.files}
        identity = dict(
            source_panel_sha256=sha_file(source / "panel.npz"),
            source_fingerprint=source_report["fingerprint"],
            cloud_fingerprint=cloud_key,
            config=asdict(config),
            heston_checkpoint_sha256=sha_file(out / "heston/model.pt"),
            market_method="neural_injective_embedding",
            native_price_dimension=21,
        )
        identity_path = out / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Run identity changed; cannot mix neural results")
        write_json(identity_path, identity)
        pricer = CppPricer(heston_grid_contracts(GridSpec()))
        validation_path = out / "heston/validation.json"
        if validation_path.exists():
            validation = json.loads(validation_path.read_text())
        else:
            scales = np.array([panel["scales"][r["panel_index"]] for r in source_rows[:half]])
            validation = validate_heston(
                model, out / "heston", np.median(scales, axis=0), pricer, config
            )
        if not validation["geometry"]["numerically_resolved"]:
            raise ValueError("Heston neural embedding is not numerically resolved")
        print(
            f"Independent Heston test: local RMS ${validation['local_rms_usd']:.5f}/bbl; "
            f"screen passed: {validation['screen_passed']}",
            flush=True,
        )
        rows, previous = [], None
        start = time.monotonic()
        for index in indices:
            source_row = source_rows[index]
            path = out / "weeks" / (source_row["date"] + ".json")
            if path.exists():
                row = json.loads(path.read_text())
                data_path = path.with_suffix(".npz")
                if (
                    sha_file(data_path) != row["arrays_sha256"]
                    or sha_file(out / "weeks" / (source_row["date"] + ".market.pt"))
                    != row["market_checkpoint_sha256"]
                ):
                    raise ValueError("Completed neural week checksum mismatch")
                with np.load(data_path, allow_pickle=False) as z:
                    previous = z["unit"]
            else:
                row, arrays = compute_snapshot(
                    source, out, config, panel, source_row, model, pricer, previous
                )
                previous = arrays["unit"]
            rows.append(row)
            write_json(
                out / "report.json",
                dict(
                    status="computing",
                    stage=stage,
                    total_weeks=len(source_rows),
                    target_weeks=len(indices),
                    completed_weeks=len(rows),
                    first_half_weeks=half,
                    first_half_end=source_rows[half - 1]["date"],
                    next_half_start=source_rows[half]["date"],
                    heston_validation=validation,
                    identity=identity,
                    rows=rows,
                    current_session_elapsed_seconds=time.monotonic() - start,
                ),
            )
        report = json.loads((out / "report.json").read_text())
        report["status"] = "first_half_computed" if len(rows) == half else "computed"
        report["stage_complete"] = len(rows) == (len(source_rows) if stage == "remaining" else half)
        report["ready_to_continue"] = len(rows) == half
        report["continuation_command"] = (
            "PYTHONPATH=src python3 -m hestonlab.neural_weekly --stage remaining"
        )
        write_json(out / "report.json", report)
        write_review(out, report)
        if video:
            from .neural_presentation import render

            render(out, config)
            report["video_complete"] = True
            write_json(out / "report.json", report)
        print(f"Stopped at {len(rows)}/{len(source_rows)} weeks; stage {stage}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/weekly_cpp"))
    parser.add_argument("--out", type=Path, default=Path("outputs/neural_weekly"))
    parser.add_argument(
        "--stage",
        choices=("prepare", "train", "first-half", "remaining", "render"),
        default="first-half",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare_cloud(args.source, args.out, NeuralConfig())
    elif args.stage == "render":
        from .neural_presentation import render

        render(args.out, NeuralConfig())
        report = json.loads((args.out / "report.json").read_text())
        report["video_complete"] = True
        write_json(args.out / "report.json", report)
    else:
        run(args.source, args.out, NeuralConfig(), args.stage, args.limit, not args.no_video)


if __name__ == "__main__":
    main()
