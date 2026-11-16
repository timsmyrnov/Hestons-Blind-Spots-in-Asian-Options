"""Custom parameter boxes and parallel Monte Carlo data for the neural surrogate.

Everything defaults to the calibration box (calibration.LOWER/UPPER). A narrower box
is used for experiments only. The box travels with the data and with the saved model,
and models, data and boxes that do not match are refused.

Deliberately free of torch at import time: worker processes for parallel Monte Carlo
import this module, and should not pay for loading torch.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
import os
import time
import numpy as np
from scipy.stats import qmc
from .calibration import LOWER, PARAMETER_NAMES, UPPER
from .heston import HestonParameters, MonteCarloConfig, price_asian_options
from .surrogate_data import LOG_SCALED, save


@dataclass(frozen=True)
class ParameterBox:
    """Bounds for (v0, kappa, theta, sigma, rho). v0, kappa, theta, sigma are sampled and
    scaled on a log scale; rho linearly (same scheme as surrogate_data)."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]
    name: str = "calibration"

    def __post_init__(self):
        lower = tuple(float(x) for x in self.lower)
        upper = tuple(float(x) for x in self.upper)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        if len(lower) != 5 or len(upper) != 5:
            raise ValueError(f"Need 5 bounds for {PARAMETER_NAMES}")
        if not all(np.isfinite(lower + upper)) or not all(a < b for a, b in zip(lower, upper)):
            raise ValueError("Each lower bound must be finite and below its upper bound")
        if min(lower[:LOG_SCALED]) <= 0 or not -1 < lower[4] < upper[4] < 1:
            raise ValueError("Positive v0, kappa, theta, sigma and -1 < rho < 1 required")

    @classmethod
    def calibration(cls):
        return cls(tuple(LOWER), tuple(UPPER), "calibration")

    @property
    def bounds(self):
        return np.array(self.lower), np.array(self.upper)

    def sample(self, n, seed=0):
        """Scrambled Sobol draws inside the box, (n, 5). Identical to
        surrogate_data.sample_parameters for the calibration box."""
        unit = qmc.Sobol(d=5, scramble=True, seed=seed).random(n)
        low, high = self.bounds
        low[:LOG_SCALED], high[:LOG_SCALED] = np.log(low[:LOG_SCALED]), np.log(high[:LOG_SCALED])
        values = low + unit * (high - low)
        values[:, :LOG_SCALED] = np.exp(values[:, :LOG_SCALED])
        return values

    def contains(self, params):
        params = np.asarray(params, dtype=float)
        low, high = self.bounds
        return bool(((params >= low) & (params <= high)).all())

    def input_scaler(self):
        from .surrogate import InputScaler

        return InputScaler(*self.bounds)

    def as_dict(self):
        return {
            "name": self.name,
            **{n: [lo, hi] for n, lo, hi in zip(PARAMETER_NAMES, self.lower, self.upper)},
        }

    def matches(self, lower, upper):
        return np.allclose(lower, self.lower) and np.allclose(upper, self.upper)


# Surrogate-value experiment box: v0 and theta up to 0.36 (60% volatility).
VALUE_EXPERIMENT_BOX = ParameterBox(
    lower=(0.01, 0.1, 0.01, 0.05, -0.95),
    upper=(0.36, 5.0, 0.36, 1.5, 0.5),
    name="value_experiment",
)


# ---- parallel Monte Carlo ---------------------------------------------------------------


def _price_chunk(contracts, rows, config):
    """Worker: price parameter rows; return prices, SEs and this process's CPU seconds."""
    start = time.process_time()
    prices, errors = [], []
    for row in rows:
        estimates = price_asian_options(
            contracts, HestonParameters(**dict(zip(PARAMETER_NAMES, row))), config
        )
        prices.append([e.price for e in estimates])
        errors.append([e.standard_error for e in estimates])
    return np.array(prices), np.array(errors), time.process_time() - start


def default_workers():
    return max(1, (os.cpu_count() or 2) - 1)


def price_points_parallel(contracts, params, config, workers=None, chunk_size=32):
    """Monte Carlo prices at given parameter rows, spread across processes.

    Every row starts its own random stream from config.seed, so results are identical
    to serial pricing whatever the number of workers. Returns (prices, standard_errors,
    timing) where timing has wall-clock and core-seconds (summed worker CPU time).
    """
    contracts = tuple(contracts)
    params = np.asarray(params, dtype=float).reshape(-1, 5)
    workers = default_workers() if workers is None else workers
    chunks = [params[i:i + chunk_size] for i in range(0, len(params), chunk_size)]
    start = time.perf_counter()
    if workers == 1 or len(chunks) == 1:
        results = [_price_chunk(contracts, c, config) for c in chunks]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(chunks))) as pool:
            results = list(pool.map(_price_chunk, [contracts] * len(chunks), chunks,
                                    [config] * len(chunks)))
    wall = time.perf_counter() - start
    prices = np.concatenate([r[0] for r in results])
    errors = np.concatenate([r[1] for r in results])
    core = float(sum(r[2] for r in results))
    timing = {
        "workers": int(min(workers, len(chunks))),
        "wall_seconds": wall,
        "core_seconds": core,
        "core_seconds_per_pricing": core / len(params),
    }
    return prices, errors, timing


def generate_in_box(contracts, n, config=MonteCarloConfig(), sobol_seed=0,
                    box=VALUE_EXPERIMENT_BOX, workers=None):
    """surrogate_data.generate() for a custom box, in parallel. Same output keys, plus
    the box and the compute it took."""
    contracts = tuple(contracts)
    params = box.sample(n, sobol_seed)
    prices, errors, timing = price_points_parallel(contracts, params, config, workers)
    return {
        "parameters": params,
        "parameter_names": np.array(PARAMETER_NAMES),
        "prices": prices,
        "standard_errors": errors,
        "coordinate_ids": np.array([c.coordinate_id for c in contracts]),
        "monte_carlo": np.array(str(asdict(config))),
        "bounds_lower": np.array(box.lower),
        "bounds_upper": np.array(box.upper),
        "box_name": np.array(box.name),
        "sobol_seed": np.array(sobol_seed),
        "core_seconds": np.array(timing["core_seconds"]),
        "wall_seconds": np.array(timing["wall_seconds"]),
    }


def cached_generate_in_box(path, contracts, n, config, sobol_seed, box, workers=None,
                           log=print):
    """Reuse a saved file only if every input matches (contracts, MC, seed, size, box)."""
    from .surrogate import contract_fingerprint

    path = Path(path)
    fingerprint = contract_fingerprint(contracts)
    if path.is_file():
        data = dict(np.load(path))
        same = (
            str(data.get("contract_fingerprint")) == fingerprint
            and str(data.get("monte_carlo")) == str(asdict(config))
            and int(data.get("sobol_seed", -1)) == sobol_seed
            and len(data["parameters"]) == n
            and box.matches(data["bounds_lower"], data["bounds_upper"])
        )
        if same:
            log(f"reusing {path}")
            return data
    data = generate_in_box(contracts, n, config, sobol_seed, box, workers)
    data["contract_fingerprint"] = np.array(fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    save(data, path)
    log(f"generated {n} draws: {float(data['wall_seconds']):.0f}s wall, "
        f"{float(data['core_seconds']):.0f} core-s -> {path}")
    return data


# ---- training and loading in a box ----------------------------------------------------


def check_data_box(data, box):
    if "bounds_lower" not in data:
        raise ValueError("Data carries no box; generate it with generate_in_box")
    if not box.matches(data["bounds_lower"], data["bounds_upper"]):
        raise ValueError(f"Data was generated in a different parameter box than {box.name}")


def train_in_box(data, contracts, box, config=None, metadata=None, verbose=False):
    """Train the default surrogate on box data, scaling inputs with the box's bounds."""
    from .surrogate import TrainConfig, train

    check_data_box(data, box)
    meta = {"parameter_box": box.as_dict(), **(metadata or {})}
    return train(data, contracts, config or TrainConfig(), meta, verbose,
                 input_scaler=box.input_scaler())


def load_in_box(path, box):
    """Load a surrogate, refusing it unless it was trained on exactly this box."""
    from .surrogate import Surrogate

    return Surrogate.load(path, bounds=box.bounds)
