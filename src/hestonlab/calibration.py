"""June-only experimental Monte Carlo calibration. July labels are never read."""

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from .heston import AsianContract, HestonParameters, MonteCarloConfig, price_asian_options


PARAMETER_NAMES = ("v0", "kappa", "theta", "sigma", "rho")
# Development search bounds, not inferred from the market or the July holdout.
LOWER = np.array([0.001, 0.05, 0.001, 0.01, -0.95])
UPPER = np.array([1.0, 10.0, 1.0, 3.0, 0.95])


@dataclass(frozen=True)
class CalibrationSnapshot:
    observation_date: str
    contracts: tuple[AsianContract, ...]
    market_prices: tuple[float, ...]

    def __post_init__(self):
        day = pd.Timestamp(self.observation_date)
        if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
            raise ValueError("A date without time or timezone is required")
        if not pd.Timestamp("2024-06-01") <= day <= pd.Timestamp("2024-06-30"):
            raise ValueError("Calibration accepts June 2024 only; July is held out")
        prices = np.asarray(self.market_prices)
        if not self.contracts or prices.shape != (len(self.contracts),):
            raise ValueError("One observed price required per contract")
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("Market labels must be positive finite observed settlements")
        if len({c.coordinate_id for c in self.contracts}) != len(self.contracts):
            raise ValueError("Duplicate coordinates in snapshot")


def load_june_training(root):
    """Use the existing June-selected fixed grid and provisional fixing handoff.

    Require unseasoned, final-forward pilot inputs. Raw public sample prices lack
    these conditioning inputs and cannot be substituted. No missing-price fill.
    """
    root = Path(root)
    option_file = root / "pilot_june_july_2024/train_fixed_coordinate_observations.parquet"
    fixing_file = root / "ao_model_fixing_inputs.parquet"
    for path in (option_file, fixing_file):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Full local archive required; run model_inputs.build "
                "and splits.build_pilot_split first. The public CSV is insufficient."
            )
    options = pd.read_parquet(option_file)
    if options.empty:
        raise ValueError("June training grid is empty")
    dates = pd.to_datetime(options.observation_date)
    if not dates.between("2024-06-01", "2024-06-30").all():
        raise ValueError("Training file contains dates outside June 2024")
    if not options.prototype_input_complete.eq(True).all():
        raise ValueError("Incomplete prototype inputs in training grid")
    for field in (
        "known_fixing_count",
        "provisional_or_missing_elapsed",
        "provisional_remaining_forwards",
        "missing_or_nonpositive_forwards",
    ):
        if not options[field].eq(0).all():
            raise ValueError(f"Pilot requires zero {field}")
    if not options.terminal_diagnostic_pass.eq(True).all():
        raise ValueError("Terminal diagnostic failed")
    fixings = pd.read_parquet(
        fixing_file,
        filters=[
            ("observation_date", ">=", pd.Timestamp("2024-06-01").date()),
            ("observation_date", "<=", pd.Timestamp("2024-06-30").date()),
        ],
    )
    snapshots = []
    for day, group in options.groupby("observation_date", sort=True):
        contracts, labels = [], []
        for row in group.sort_values("contract_id").itertuples():
            f = fixings.loc[
                (pd.to_datetime(fixings.observation_date) == pd.Timestamp(day))
                & (fixings.contract_month == row.contract_month)
            ].sort_values("fixing_date")
            if len(f) != row.total_fixings or len(f) != row.remaining_fixings:
                raise ValueError("Missing or inconsistent fixing schedule")
            if f.fixing_date.duplicated().any() or not f.forward_final_flag.eq(True).all():
                raise ValueError("Duplicate fixings or nonfinal forward inputs")
            expected_times = (pd.to_datetime(f.fixing_date) - pd.Timestamp(day)).dt.days / 365
            if not np.allclose(f.fixing_time_years, expected_times, atol=1e-12, rtol=0):
                raise ValueError("Fixing times do not match ACT/365F dates")
            if not f.known_fixing.isna().all() or row.known_fixing_sum != 0:
                raise ValueError("Pilot must have no accrued or realized future fixings")
            contracts.append(
                AsianContract(
                    row.contract_id,
                    tuple(f.fixing_time_years),
                    tuple(f.current_forward),
                    int(row.total_fixings),
                    float(row.known_fixing_sum),
                    float(row.strike),
                    row.call_put,
                    float(row.discount_factor),
                )
            )
            labels.append(float(row.settlement_price))
        snapshots.append(
            CalibrationSnapshot(str(pd.Timestamp(day).date()), tuple(contracts), tuple(labels))
        )
    return tuple(snapshots)


def _encode(parameters):
    values = np.array([getattr(parameters, name) for name in PARAMETER_NAMES])
    if (values < LOWER).any() or (values > UPPER).any():
        raise ValueError("Initial parameters outside development search bounds")
    values[:4] = np.log(values[:4])
    low, high = LOWER.copy(), UPPER.copy()
    low[:4], high[:4] = np.log(low[:4]), np.log(high[:4])
    return (values - low) / (high - low)


def _decode(point):
    low, high = LOWER.copy(), UPPER.copy()
    low[:4], high[:4] = np.log(low[:4]), np.log(high[:4])
    values = low + point * (high - low)
    values[:4] = np.exp(values[:4])
    return HestonParameters(**dict(zip(PARAMETER_NAMES, values)))


def calibrate_june(
    snapshots, initial=HestonParameters(), config=MonteCarloConfig(), max_evaluations=100
):
    """Bounded coordinate search with fixed common random numbers.

    Fits ONE shared parameter vector across June snapshots, resetting v0 daily.
    Equal weight per observed USD/barrel price. This starter local search does
    not establish global optimality or parameter identifiability.
    """
    snapshots = tuple(snapshots)
    if not snapshots or sum(len(s.contracts) for s in snapshots) < 6:
        raise ValueError("At least six observed option prices required")
    if len({s.observation_date for s in snapshots}) != len(snapshots):
        raise ValueError("Duplicate snapshot dates")
    if not isinstance(max_evaluations, int) or max_evaluations < 11:
        raise ValueError("max_evaluations must be an integer >= 11")
    # Recheck the split even for callers supplying their own snapshot objects.
    for s in snapshots:
        s.__post_init__()
    targets = np.concatenate([s.market_prices for s in snapshots])

    def evaluate(point, mc):
        estimates = []
        for s in snapshots:
            # Stable date-specific streams; identical across candidate parameters.
            seed = mc.seed + int(pd.Timestamp(s.observation_date).strftime("%Y%m%d"))
            daily = MonteCarloConfig(mc.paths, mc.max_step_years, seed)
            estimates.extend(price_asian_options(s.contracts, _decode(point), daily))
        prices = np.array([e.price for e in estimates])
        return float(np.mean((prices - targets) ** 2)), estimates

    point = _encode(initial)
    best, _ = evaluate(point, config)
    initial_rmse = float(np.sqrt(best))
    evaluations, step = 1, 0.15
    while evaluations < max_evaluations and step >= 0.005:
        improved = False
        for axis in range(5):
            for sign in (-1, 1):
                if evaluations >= max_evaluations:
                    break
                candidate = point.copy()
                candidate[axis] = np.clip(candidate[axis] + sign * step, 0, 1)
                loss, _ = evaluate(candidate, config)
                evaluations += 1
                if loss < best:
                    point, best, improved = candidate, loss, True
        if not improved:
            step /= 2
    # Fresh streams diagnose fitting the finite random sample; never uses July.
    independent = MonteCarloConfig(config.paths, config.max_step_years, config.seed + 1000000)
    independent_loss, estimates = evaluate(point, independent)
    return {
        "status": "experimental_local_search_not_validated",
        "parameters": asdict(_decode(point)),
        "feller_margin": _decode(point).feller_margin,
        "training_dates": [s.observation_date for s in snapshots],
        "held_out_month": "2024-07",
        "july_labels_used": False,
        "observed_prices": len(targets),
        "initial_rmse": initial_rmse,
        "training_rmse": float(np.sqrt(best)),
        "independent_seed_rmse": float(np.sqrt(independent_loss)),
        "max_independent_price_standard_error": max(e.standard_error for e in estimates),
        "evaluations": evaluations,
        "search_step": step,
        "stop_reason": "evaluation_budget" if evaluations >= max_evaluations else "step_threshold",
        "bounds": dict(zip(PARAMETER_NAMES, zip(LOWER.tolist(), UPPER.tolist()))),
        "monte_carlo": asdict(config),
        "fully_validated": False,
        "conventions": "shared June parameters; daily v0 reset; common-factor futures; "
        "discrete arithmetic average; equal price weights; USD/barrel",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=4096)
    parser.add_argument("--max-evaluations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max-step-days", type=float, default=1.0)
    args = parser.parse_args()
    try:
        config = MonteCarloConfig(args.paths, args.max_step_days / 365, args.seed)
        result = calibrate_june(
            load_june_training(args.root), config=config, max_evaluations=args.max_evaluations
        )
    except (ValueError, FileNotFoundError, FloatingPointError) as exc:
        parser.exit(2, f"Calibration unavailable: {exc}\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
