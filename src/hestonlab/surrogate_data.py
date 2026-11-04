"""Training data for the neural surrogate: Heston parameters -> Monte Carlo AO prices.

Each row is one parameter draw priced on ONE fixed contract set. Every draw reuses
the same random numbers (common random numbers), so prices change smoothly with the
parameters instead of jittering from sampling noise. Draws cover the calibration
search box, so the surrogate is never asked about parameters it was not trained on.
"""

from dataclasses import asdict
import time
import numpy as np
import pandas as pd
from scipy.stats import qmc
from .calibration import LOWER, PARAMETER_NAMES, UPPER
from .heston import AsianContract, HestonParameters, MonteCarloConfig, price_asian_options

LOG_SCALED = 4  # v0, kappa, theta, sigma are sampled on a log scale; rho is linear.


def sample_parameters(n, seed=0):
    """Scrambled Sobol draws over the calibration bounds. Returns an (n, 5) array."""
    unit = qmc.Sobol(d=5, scramble=True, seed=seed).random(n)
    low, high = LOWER.copy(), UPPER.copy()
    low[:LOG_SCALED], high[:LOG_SCALED] = np.log(low[:LOG_SCALED]), np.log(high[:LOG_SCALED])
    values = low + unit * (high - low)
    values[:, :LOG_SCALED] = np.exp(values[:, :LOG_SCALED])
    return values


def generate(contracts, n, config=MonteCarloConfig(), seed=0, progress_every=0):
    """Price every parameter draw on the same contracts with the same random numbers."""
    contracts = tuple(contracts)
    params = sample_parameters(n, seed)
    prices = np.empty((n, len(contracts)))
    errors = np.empty_like(prices)
    start = time.time()
    for i, row in enumerate(params):
        estimates = price_asian_options(
            contracts, HestonParameters(**dict(zip(PARAMETER_NAMES, row))), config
        )
        prices[i] = [e.price for e in estimates]
        errors[i] = [e.standard_error for e in estimates]
        if progress_every and (i + 1) % progress_every == 0:
            print(f"{i + 1}/{n} draws, {time.time() - start:.0f}s")
    return {
        "parameters": params,
        "parameter_names": np.array(PARAMETER_NAMES),
        "prices": prices,
        "standard_errors": errors,
        "coordinate_ids": np.array([c.coordinate_id for c in contracts]),
        "monte_carlo": np.array(str(asdict(config))),
    }


def save(data, path):
    np.savez_compressed(path, **data)


def synthetic_pilot_grid(
    valuation_date="2024-06-14",
    months=("2024-08", "2024-09", "2024-10"),
    strikes=range(60, 101, 5),
    forward=80.0,
    rate=0.053,
):
    """SYNTHETIC development grid shaped like the June pilot: unseasoned calls on the
    Aug/Sep/Oct averaging months, $60-100 strikes, flat $80 curve, weekday fixings.
    For testing the pipeline only. Real runs use the archive's contracts per date."""
    day = pd.Timestamp(valuation_date)
    contracts = []
    for month in months:
        fixings = pd.bdate_range(month + "-01", pd.Timestamp(month + "-01") + pd.offsets.MonthEnd())
        times = tuple((fixings - day).days / 365)
        discount = (1 + rate / 360) ** -(fixings[-1] - day).days
        for k in strikes:
            contracts.append(
                AsianContract(
                    f"synthetic:{month}:C:{k}",
                    times,
                    (forward,) * len(times),
                    len(times),
                    0.0,
                    float(k),
                    "C",
                    discount,
                )
            )
    return tuple(contracts)
