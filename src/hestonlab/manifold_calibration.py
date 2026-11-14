"""Fit June prices on learned manifolds, then independently reprice with Monte Carlo.

python -m hestonlab.manifold_calibration --root /private/archive \
    --models outputs/manifold/june --out outputs/manifold/june/fit.json

Default: each date gets its own five parameters (--pooled shares one vector).
Residuals are divided by each contract's pricing error scale (--weighting usd
for equal USD weights). Every fit also reports a Heston control: Heston's own
prices fitted the same way, i.e. the distance that is pure approximation error.
"""

from dataclasses import asdict
from pathlib import Path
import argparse
import ast
import hashlib
import json
import numpy as np
import torch
from scipy.optimize import least_squares

from .calibration import LOWER, UPPER, PARAMETER_NAMES, load_june_training
from .heston import HestonParameters, MonteCarloConfig, price_asian_options
from .manifold import ManifoldNetwork, geometry_diagnostics
from .surrogate import DTYPE, InputScaler, Surrogate
from .surrogate_data import sample_parameters

WEIGHTINGS = {
    "noise": "residual / per-contract pricing error scale (noise units)",
    "usd": "equal-weight Euclidean prices in USD/barrel",
}
SCALE_KEY = "price_error_scale_usd_per_bbl"


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


def residual_geometry(jacobian, residual, metric=WEIGHTINGS["usd"]):
    """Euclidean metric (in the given weighting); normal component is a LOCAL diagnostic.

    This is not another global projection. At box boundaries the feasible tangent
    cone is smaller than the full tangent space used here.
    """
    left, singular, _ = np.linalg.svd(jacobian, full_matrices=False)
    tolerance = max(jacobian.shape) * np.finfo(float).eps * singular[0]
    rank = int((singular > tolerance).sum())
    basis = left[:, :rank]
    tangent = basis @ (basis.T @ residual)
    normal = residual - tangent
    return {
        "metric": metric,
        "numerical_rank": rank,
        "singular_values": singular.tolist(),
        "tangent_residual": tangent.tolist(),
        "normal_residual": normal.tolist(),
        "tangent_rmse": float(np.sqrt(np.mean(tangent**2))),
        "normal_rmse": float(np.sqrt(np.mean(normal**2))),
        "interpretation": "local linear decomposition, not a global distance or a "
                          "boundary tangent-cone projection",
    }


def fit_manifold(snapshots, models, initial=HestonParameters(), starts=8,
                 max_evaluations=300, seed=2024, verification_paths=65536,
                 weighting="noise"):
    """One Heston vector for the supplied June snapshots (one date: its own vector).

    With several dates this is the pooled fit: one shared vector, v0 included.
    Each date has its own fixed-contract manifold. Multiple bounded local searches
    reduce start sensitivity; their best result is not a certified global minimum.
    Numerical geometry is checked before fitting. Independent MC verification is
    mandatory, and never used by the optimizer.
    """
    snapshots = tuple(snapshots)
    if not snapshots or starts < 1 or max_evaluations < 1:
        raise ValueError("Snapshots, positive starts and evaluation budget required")
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {sorted(WEIGHTINGS)}")
    dates = [s.observation_date for s in snapshots]
    if len(set(dates)) != len(dates) or set(models) != set(dates):
        raise ValueError("Exactly one matching model per distinct observation date required")
    scaler = InputScaler()
    initial_unit = scaler.to_unit(initial)
    geometry = {}
    label_configs = {}
    scales = []
    check_units = np.vstack((np.zeros(5), np.ones(5), np.full(5, 0.5),
                             scaler.to_unit(sample_parameters(32, seed))))
    for snapshot in snapshots:
        snapshot.__post_init__()  # June-only scope, positive labels, no duplicates.
        model = models[snapshot.observation_date]
        if not isinstance(model.network, ManifoldNetwork):
            raise ValueError("Fitting requires a manifold model; retrain legacy MLP checkpoints")
        model.check_contracts(snapshot.contracts)
        if model.coordinate_ids != tuple(c.coordinate_id for c in snapshot.contracts):
            raise ValueError("Model price coordinates do not match snapshot order")
        try:
            cfg = MonteCarloConfig(**ast.literal_eval(model.metadata["monte_carlo"]))
        except (KeyError, TypeError, ValueError, SyntaxError) as exc:
            raise ValueError("Training Monte Carlo metadata required for independent verification") from exc
        label_configs[snapshot.observation_date] = cfg
        # Validate requested verification count before invoking the optimizer.
        MonteCarloConfig(verification_paths, cfg.max_step_years, cfg.seed + 1_000_000)
        diagnostic = geometry_diagnostics(model, check_units)
        if not diagnostic["numerically_resolved"]:
            raise ValueError("Manifold geometry is numerically unresolved; inspect conditioning")
        geometry[snapshot.observation_date] = diagnostic
        if weighting == "usd":
            scales.append(np.ones(model.n_outputs))
            continue
        scale = np.asarray(model.metadata.get(SCALE_KEY, []), dtype=float)
        if scale.shape != (model.n_outputs,):
            raise ValueError("Noise weighting needs per-contract error scales; retrain with "
                             "surrogate_pipeline or use weighting='usd'")
        if not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Per-contract error scales must be finite and positive")
        scales.append(scale)
    target = np.concatenate([s.market_prices for s in snapshots])
    scale = np.concatenate(scales)
    weight = 1 / scale

    def prediction(unit):
        return torch.cat([models[d].prices_from_unit(unit) for d in dates])

    def predict(unit):
        with torch.no_grad():
            return prediction(torch.as_tensor(unit, dtype=DTYPE)).numpy()

    def residual(unit, goal):
        return (predict(unit) - goal) * weight

    def derivative(unit, goal=None):
        jac = torch.func.jacrev(prediction)(torch.as_tensor(unit, dtype=DTYPE)).detach().numpy()
        return jac * weight[:, None]

    # Power-of-two Sobol generation, followed by a deterministic prefix.
    count = 1 << max(0, (starts - 2).bit_length())
    candidates = [initial_unit, *scaler.to_unit(sample_parameters(count, seed))[:starts - 1]]

    def solve(goal, points):
        runs, best = [], None
        for point in points:
            result = least_squares(residual, point, jac=derivative, args=(goal,),
                                   bounds=(0, 1), max_nfev=max_evaluations,
                                   ftol=1e-10, xtol=1e-10, gtol=1e-10)
            distance = rms(result.fun)
            if not np.isfinite(distance):
                raise ValueError("Nonfinite manifold fit")
            runs.append({"initial_unit": point.tolist(), "final_unit": result.x.tolist(),
                         "weighted_distance": distance,
                         "usd_rmse": rms(predict(result.x) - goal),
                         "success": bool(result.success),
                         "message": result.message, "evaluations": result.nfev,
                         "optimality": float(result.optimality)})
            if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
                best = result
        return best, runs

    best, runs = solve(target, candidates)
    unit = best.x
    parameters = HestonParameters(**dict(zip(PARAMETER_NAMES, scaler.from_unit(unit))))
    predicted = predict(unit)
    market_residual = target - predicted
    fitted_geometry = residual_geometry(derivative(unit), weight * market_residual,
                                        WEIGHTINGS[weighting])
    # Also test inversion/conditioning at the actual optimizer output.
    fitted_checks = {d: geometry_diagnostics(models[d], unit[None]) for d in dates}
    resolved = all(g["numerically_resolved"] for g in fitted_checks.values())
    streams, stream_configs, standard_errors, coordinates = [[], []], {}, [], []
    for snapshot in snapshots:
        day = snapshot.observation_date
        cfg = label_configs[day]
        configs = [MonteCarloConfig(verification_paths, cfg.max_step_years,
                                   cfg.seed + k * 1_000_000) for k in (1, 2)]
        stream_configs[day] = [asdict(c) for c in configs]
        errors = []
        for i, mc in enumerate(configs):
            estimates = price_asian_options(snapshot.contracts, parameters, mc)
            streams[i].extend(e.price for e in estimates)
            errors.append([e.standard_error for e in estimates])
        standard_errors.extend((np.sqrt(np.sum(np.asarray(errors)**2, axis=0)) / 2).tolist())
        coordinates.extend({"observation_date": day, "coordinate_id": c.coordinate_id}
                           for c in snapshot.contracts)
    a, b = np.asarray(streams)
    reference = (a + b) / 2
    # Heston control: prices Heston really produces (independent MC at the fitted
    # parameters), fitted exactly like the market. Its distance is the zero level:
    # pure approximation error. Starting from the market fit as well guarantees the
    # control is never worse than the surrogate's error at that point.
    control, control_runs = solve(reference, [unit, *candidates])
    market_distance, control_distance = rms(best.fun), rms(control.fun)
    rows = [{**identity, "observed": float(target[i]), "manifold_price": float(predicted[i]),
             "independent_mc_price": float(reference[i]),
             "independent_mc_standard_error": standard_errors[i],
             "error_scale_usd_per_bbl": float(scale[i]),
             "market_minus_manifold": float(market_residual[i]),
             "weighted_residual": float(weight[i] * market_residual[i]),
             "market_minus_mc": float(target[i] - reference[i])}
            for i, identity in enumerate(coordinates)]
    pooled = len(dates) > 1
    return {
        "status": "experimental_manifold_fit_with_independent_mc_verification",
        "fully_validated": False,
        "mode": "pooled" if pooled else "single_date",
        "weighting": weighting,
        "metric": WEIGHTINGS[weighting],
        "parameters": asdict(parameters),
        "bounds": dict(zip(PARAMETER_NAMES, zip(LOWER.tolist(), UPPER.tolist()))),
        "training_dates": dates, "july_labels_used": False,
        "conventions": ("one shared parameter vector, v0 included, for all dates; "
                        "one manifold per date") if pooled else
                       "one date with its own five parameters and manifold",
        "market_weighted_distance": market_distance,
        "heston_control": {
            "what": "independent-MC Heston prices at the fitted parameters, fitted the same "
                    "way; its distance is pure approximation error (the zero level)",
            "weighted_distance": control_distance,
            "usd_rmse": rms(predict(control.x) - reference),
            "final_unit": control.x.tolist(),
            "success": bool(control.success),
            "starts": control_runs,
        },
        "market_to_control_ratio": (market_distance / control_distance
                                    if control_distance > 0 else None),
        "initial_rmse": rms(predict(initial_unit) - target),
        "manifold_fit_rmse": rms(market_residual),
        "independent_mc_market_rmse": rms(target - reference),
        "surrogate_vs_independent_mc_rmse": rms(predicted - reference),
        "mc_stream_disagreement_rmse": rms(a - b),
        "verification_monte_carlo": stream_configs,
        "observed_prices": len(target), "rows": rows,
        "optimizer": {"starts": runs, "seed": seed, "success": bool(best.success),
                      "max_evaluations_per_start": max_evaluations,
                      "global_minimum_certified": False},
        "boundary_parameters": [name for name, value in zip(PARAMETER_NAMES, unit)
                                if min(value, 1 - value) < 1e-6],
        "geometry_before_fit": geometry, "geometry_at_fit": fitted_checks,
        "geometry_numerically_resolved_at_fit": resolved,
        "residual_geometry": fitted_geometry,
        "limitations": "Manifold guarantee applies to the learned map. Pricing fidelity, "
                       "discretization bias and economic constraints require separate validation. "
                       "Raw verification errors include MC noise; no accuracy threshold is implied.",
    }


def fit_per_date(snapshots, models, **kwargs):
    """Each date fitted on its own: its own five parameters on its own manifold."""
    snapshots = tuple(snapshots)
    dates = [s.observation_date for s in snapshots]
    if not snapshots or len(set(dates)) != len(dates) or set(models) != set(dates):
        raise ValueError("Exactly one matching model per distinct observation date required")
    reports = {s.observation_date: fit_manifold([s], {s.observation_date:
                                                      models[s.observation_date]}, **kwargs)
               for s in snapshots}
    summary = [{"observation_date": d, "observed_prices": r["observed_prices"],
                "market_weighted_distance": r["market_weighted_distance"],
                "heston_control_weighted_distance": r["heston_control"]["weighted_distance"],
                "market_to_control_ratio": r["market_to_control_ratio"],
                "manifold_fit_rmse_usd": r["manifold_fit_rmse"],
                "parameters": r["parameters"],
                "boundary_parameters": r["boundary_parameters"]}
               for d, r in reports.items()]
    return {
        "status": "experimental_manifold_fit_with_independent_mc_verification",
        "fully_validated": False, "mode": "per_date",
        "weighting": kwargs.get("weighting", "noise"),
        "training_dates": dates, "july_labels_used": False,
        "summary": summary, "dates": reports,
    }


def fit_saved_models(snapshots, model_dir, pooled=False, **kwargs):
    """Load the date-specific checkpoints written by surrogate_pipeline june."""
    snapshots = tuple(snapshots)
    paths = {s.observation_date: Path(model_dir) / s.observation_date / "model.pt"
             for s in snapshots}
    models = {d: Surrogate.load(p) for d, p in paths.items()}
    report = (fit_manifold if pooled else fit_per_date)(snapshots, models, **kwargs)
    report["model_sha256"] = {d: hashlib.sha256(p.read_bytes()).hexdigest()
                              for d, p in paths.items()}
    return report


def main(argv=None):
    from .surrogate_pipeline import check_output_location

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dates", nargs="+")
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--max-evaluations", type=int, default=300)
    parser.add_argument("--verification-paths", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--pooled", action="store_true",
                        help="one shared parameter vector for all dates (default: per date)")
    parser.add_argument("--weighting", choices=sorted(WEIGHTINGS), default="noise")
    args = parser.parse_args(argv)
    try:
        out = check_output_location(args.out)
        if args.root.resolve() in (out, *out.parents):
            raise ValueError("Write outputs outside the preserved data archive")
        if out.exists():
            raise ValueError("Use a fresh fit output path")
        snapshots = load_june_training(args.root)
        if args.dates:
            missing = set(args.dates) - {s.observation_date for s in snapshots}
            if missing:
                raise ValueError(f"No June snapshot for {sorted(missing)}")
            snapshots = [s for s in snapshots if s.observation_date in args.dates]
        report = fit_saved_models(snapshots, args.models, pooled=args.pooled,
                                  starts=args.starts, weighting=args.weighting,
                                  max_evaluations=args.max_evaluations, seed=args.seed,
                                  verification_paths=args.verification_paths)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Manifold fitting unavailable: {exc}\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"Saved manifold fit and independent Monte Carlo verification to {out}")
    return report


if __name__ == "__main__":
    main()
