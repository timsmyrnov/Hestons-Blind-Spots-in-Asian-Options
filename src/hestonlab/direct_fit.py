"""Bounded direct-Monte-Carlo calibration with real-call budgets and deadlines."""

from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import least_squares
from .calibration import LOWER, UPPER

LOW = np.r_[np.log(LOWER[:4]), LOWER[4]]
HIGH = np.r_[np.log(UPPER[:4]), UPPER[4]]


def decode(unit):
    values = LOW + np.asarray(unit) * (HIGH - LOW)
    return np.r_[np.exp(values[:4]), values[4]]


def encode(parameters):
    p = np.asarray(parameters, dtype=float)
    return (np.r_[np.log(p[:4]), p[4]] - LOW) / (HIGH - LOW)


class DeadlineExceeded(RuntimeError):
    pass


class CallsExhausted(RuntimeError):
    pass


@dataclass
class Budget:
    deadline: float
    limit: int
    calls: int = 0

    def check(self):
        if time.monotonic() >= self.deadline:
            raise DeadlineExceeded("Computation deadline reached; completed checkpoints are saved")

    def consume(self):
        self.check()
        if self.calls >= self.limit:
            raise CallsExhausted("Actual pricing-call budget exhausted")
        self.calls += 1


def fit(
    pricer,
    target,
    scale,
    starts,
    *,
    paths=4096,
    seed=2024,
    max_step=1 / 365,
    max_calls=1200,
    deadline=float("inf"),
    difference_step=0.002,
):
    """Multi-start local fit; each C++ invocation, including Jacobians, counts.

    A fixed absolute step in the normalized parameter box is deliberate: machine-
    epsilon finite differences are inappropriate for finite-path Monte Carlo.
    """
    target, scale = np.asarray(target), np.asarray(scale)
    if not np.isfinite(target).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Finite target and positive finite scales required")
    if max_calls < len(starts) or not starts:
        raise ValueError("At least one call per starting point is required")
    budget = Budget(deadline, max_calls)
    best = {"loss": float("inf")}
    solved = {"loss": float("inf")}
    outcomes = []
    for start_index, start in enumerate(starts):
        # Reserve work for every restart rather than spending everything on the first.
        allowance = max(1, (max_calls - budget.calls) // (len(starts) - start_index))
        stop = budget.calls + allowance
        cached = {}
        cached_errors = {}

        def residual(unit):
            budget.check()
            key = np.asarray(unit, dtype=np.float64).tobytes()
            if key not in cached:
                if budget.calls >= stop:
                    raise CallsExhausted()
                budget.consume()
                estimate = pricer.price(decode(unit), paths=paths, seed=seed, max_step=max_step)
                price = estimate.prices
                r = (price - target) * scale / np.sqrt(len(scale))
                cached[key] = r
                cached_errors[key] = estimate.standard_errors
                loss = float(r @ r)
                if loss < best["loss"]:
                    best.update(
                        unit=np.array(unit), price=price, errors=estimate.standard_errors, loss=loss
                    )
            return cached[key]

        def jacobian(unit):
            columns = []
            base = residual(unit)
            for axis in range(5):
                shift = difference_step if unit[axis] + difference_step <= 1 else -difference_step
                other = np.array(unit)
                other[axis] += shift
                columns.append((residual(other) - base) / shift)
            return np.array(columns).T

        unit = np.clip(np.asarray(start), 0, 1)
        try:
            result = least_squares(
                residual,
                unit,
                jac=jacobian,
                bounds=(0, 1),
                ftol=1e-6,
                xtol=1e-5,
                gtol=1e-5,
                max_nfev=max_calls,
            )
            outcomes.append(
                {
                    "converged": bool(result.success),
                    "message": result.message,
                    "cost": float(result.fun @ result.fun),
                }
            )
            loss = float(result.fun @ result.fun)
            if result.success and loss < solved["loss"]:
                # Recover prices from the already evaluated residual. No uncounted call.
                price = target + result.fun * np.sqrt(len(scale)) / scale
                solved.update(
                    unit=np.array(result.x),
                    price=price,
                    loss=loss,
                    errors=cached_errors[np.asarray(result.x, dtype=np.float64).tobytes()],
                )
        except CallsExhausted:
            outcomes.append({"converged": False, "message": "actual_call_budget"})
    if "unit" not in best:
        raise RuntimeError("No finite Heston candidate evaluated")
    # Derivative probes can beat an actual optimizer iterate by tiny amounts.
    # Prefer the converged point when that difference is immaterial (1% of loss).
    converged = solved["loss"] <= best["loss"] + max(1e-8, 0.01 * best["loss"])
    if converged:
        best = solved
    return {
        "unit": best["unit"],
        "parameters": decode(best["unit"]),
        "prices": best["price"],
        "rmse_usd": np.sqrt(best["loss"]),
        "standard_errors": best["errors"],
        "calls": budget.calls,
        "starts": outcomes,
        "at_bound": bool(np.any((best["unit"] < 0.001) | (best["unit"] > 0.999))),
        "converged": converged,
    }


def price_jacobian(
    pricer, unit, *, paths=16384, seed=2024, step=0.004, max_step=1 / 365, budget=None
):
    def evaluate(x):
        if budget is not None:
            budget.consume()
        return pricer.price(decode(x), paths=paths, seed=seed, max_step=max_step).prices

    jac = []
    for axis in range(5):
        a, b = np.array(unit), np.array(unit)
        a[axis] = max(0, a[axis] - step)
        b[axis] = min(1, b[axis] + step)
        jac.append((evaluate(b) - evaluate(a)) / (b[axis] - a[axis]))
    return np.array(jac).T
