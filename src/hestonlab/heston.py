"""Experimental common-factor Heston pricing for discrete arithmetic AO options.

Full-truncation Euler variance, log-Euler futures, ACT/365F time. All CL
maturities share the same proportional shock. No neural surrogate is used.
"""

from dataclasses import dataclass, asdict
import math
import numpy as np


@dataclass(frozen=True)
class HestonParameters:
    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    sigma: float = 0.5
    rho: float = -0.5

    def __post_init__(self):
        values = asdict(self)
        if not all(math.isfinite(x) for x in values.values()):
            raise ValueError("Heston parameters must be finite")
        if min(self.v0, self.kappa, self.theta, self.sigma) <= 0 or not -1 < self.rho < 1:
            raise ValueError("Positive variance parameters and -1 < rho < 1 required")

    @property
    def feller_margin(self):
        return 2 * self.kappa * self.theta - self.sigma**2


@dataclass(frozen=True)
class MonteCarloConfig:
    paths: int = 4096
    max_step_years: float = 1 / 365
    seed: int = 2024

    def __post_init__(self):
        if not isinstance(self.paths, int) or self.paths < 4 or self.paths % 2:
            raise ValueError("paths must be an even integer >= 4")
        if not math.isfinite(self.max_step_years) or self.max_step_years <= 0:
            raise ValueError("max_step_years must be finite and positive")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


@dataclass(frozen=True)
class AsianContract:
    coordinate_id: str
    fixing_times: tuple[float, ...]
    forwards: tuple[float, ...]
    total_fixings: int
    known_fixing_sum: float
    strike: float
    side: str
    discount_factor: float

    def __post_init__(self):
        times = np.asarray(self.fixing_times, dtype=float)
        forwards = np.asarray(self.forwards, dtype=float)
        if times.ndim != 1 or forwards.shape != times.shape:
            raise ValueError("One forward is required for each remaining fixing")
        if not np.isfinite(times).all() or (times <= 0).any() or (np.diff(times) <= 0).any():
            raise ValueError("Remaining fixing times must be positive and strictly increasing")
        if not np.isfinite(forwards).all() or (forwards <= 0).any():
            raise ValueError("Positive finite forwards required by multiplicative baseline")
        if not isinstance(self.total_fixings, int) or self.total_fixings < max(1, len(times)):
            raise ValueError("Invalid total fixing count")
        if self.side not in ("C", "P"):
            raise ValueError("side must be C or P")
        if not all(
            math.isfinite(x) for x in (self.known_fixing_sum, self.strike, self.discount_factor)
        ):
            raise ValueError("Contract inputs must be finite")
        if self.discount_factor <= 0:
            raise ValueError("discount_factor must be positive")


@dataclass(frozen=True)
class PriceEstimate:
    coordinate_id: str
    price: float
    standard_error: float


def price_asian_options(contracts, parameters, config=MonteCarloConfig()):
    """Price one valuation snapshot jointly; SE uses independent antithetic pairs.

    SE measures sampling uncertainty only, not time-discretization bias. Calling
    with the same config reuses random numbers across parameter candidates.
    """
    contracts = tuple(contracts)
    if not contracts:
        raise ValueError("At least one contract is required")
    times = sorted({t for c in contracts for t in c.fixing_times})
    rng = np.random.default_rng(config.seed)
    variance = np.full(config.paths, parameters.v0, dtype=float)
    factor = np.ones(config.paths)
    sums = np.array([np.full(config.paths, c.known_fixing_sum, dtype=float) for c in contracts])
    events = {}
    for i, c in enumerate(contracts):
        for t, forward in zip(c.fixing_times, c.forwards):
            events.setdefault(t, []).append((i, forward))
    previous = 0.0
    for target in times:
        steps = math.ceil((target - previous) / config.max_step_years)
        dt = (target - previous) / steps
        for _ in range(steps):
            z = rng.standard_normal((2, config.paths // 2))
            z = np.concatenate((z, -z), axis=1)
            positive_v = np.maximum(variance, 0)
            diffusion = np.sqrt(positive_v * dt)
            price_shock = parameters.rho * z[0] + math.sqrt(1 - parameters.rho**2) * z[1]
            with np.errstate(over="raise", invalid="raise"):
                factor *= np.exp(-0.5 * positive_v * dt + diffusion * price_shock)
                # Retain negative auxiliary variance; truncate in drift/diffusion only.
                variance += parameters.kappa * (parameters.theta - positive_v) * dt
                variance += parameters.sigma * diffusion * z[0]
        for i, forward in events[target]:
            sums[i] += forward * factor
        previous = target
    estimates = []
    for c, total in zip(contracts, sums):
        signed = (total / c.total_fixings - c.strike) * (1 if c.side == "C" else -1)
        payoff = c.discount_factor * np.maximum(signed, 0)
        paired = (payoff[: config.paths // 2] + payoff[config.paths // 2 :]) / 2
        if not np.isfinite(paired).all():
            raise FloatingPointError("Nonfinite simulated payoff")
        estimates.append(
            PriceEstimate(
                c.coordinate_id,
                float(paired.mean()),
                float(paired.std(ddof=1) / math.sqrt(len(paired))),
            )
        )
    return estimates
