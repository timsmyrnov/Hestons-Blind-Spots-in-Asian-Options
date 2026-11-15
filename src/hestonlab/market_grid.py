"""Canonical non-neural market grid, interpolation and preserved archive loading."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from .heston import AsianContract


@dataclass(frozen=True)
class GridSpec:
    """Fixed coordinates: years to last fixing x log(strike / average forward)."""

    tenors: tuple[float, ...] = (0.2, 0.3, 0.4)
    log_moneyness: tuple[float, ...] = (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2)
    fixings: int = 21
    averaging_years: float = 29 / 365  # first to last fixing of a calendar month

    def __post_init__(self):
        for name in ("tenors", "log_moneyness"):
            values = np.asarray(getattr(self, name), dtype=float)
            if (
                values.ndim != 1
                or len(values) == 0
                or not np.isfinite(values).all()
                or (np.diff(values) <= 0).any()
            ):
                raise ValueError(f"{name} must be finite and strictly increasing")
        if not isinstance(self.fixings, int) or self.fixings < 2:
            raise ValueError("At least two fixings per canonical contract required")
        if not 0 < self.averaging_years < min(self.tenors):
            raise ValueError("Every tenor must exceed the averaging window (unseasoned)")

    @property
    def coordinate_ids(self):
        return tuple(f"tau={t:g}:x={x:+g}" for t in self.tenors for x in self.log_moneyness)

    @property
    def size(self):
        return len(self.tenors) * len(self.log_moneyness)


def heston_grid_contracts(grid):
    """Canonical unseasoned contracts: forward 1, discount 1, out-of-the-money side.

    Their prices are directly comparable with the normalized market grid vectors.
    """
    contracts = []
    for tau in grid.tenors:
        times = tuple(np.linspace(tau - grid.averaging_years, tau, grid.fixings).tolist())
        for x, cid in zip(grid.log_moneyness, grid.coordinate_ids[len(contracts) :]):
            contracts.append(
                AsianContract(
                    cid,
                    times,
                    (1.0,) * grid.fixings,
                    grid.fixings,
                    0.0,
                    float(np.exp(x)),
                    "P" if x < 0 else "C",
                    1.0,
                )
            )
    return tuple(contracts)


def black_otm(x, vol, tau):
    """Normalized out-of-the-money Black-76 price: forward 1, strike e^x, discount 1.

    Puts below the forward (x < 0), calls at or above it.
    """
    x, vol, tau = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (x, vol, tau)))
    from scipy.special import ndtr

    s = vol * np.sqrt(tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (-x + 0.5 * s * s) / s
    d2 = d1 - s
    call = ndtr(d1) - np.exp(x) * ndtr(d2)
    put = np.exp(x) * ndtr(-d2) - ndtr(-d1)
    return np.where(x < 0, put, call)


def implied_vol(x, price, tau, low=1e-4, high=8.0, steps=80):
    """Black-76 volatility that reproduces a normalized out-of-the-money price (NaN if
    no volatility in [low, high] does). Only a smooth coordinate for interpolation."""
    x, price, tau = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (x, price, tau)))
    lo, hi = np.full(x.shape, low), np.full(x.shape, high)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        above = black_otm(x, mid, tau) > price
        lo, hi = np.where(above, lo, mid), np.where(above, mid, hi)
    vol = 0.5 * (lo + hi)
    ok = (black_otm(x, low, tau) < price) & (price < black_otm(x, high, tau))
    return np.where(ok, vol, np.nan)


def day_vector(contracts, prices, grid):
    """One day's quotes -> grid vector, NaN where the grid is not covered.

    Unseasoned contracts only. Out-of-the-money side only: puts below the average
    forward, calls at or above it. Normalized price = price / (discount * average
    forward). Interpolation runs in Black-76 implied volatility (time = time to the
    last fixing), which is smooth across the money; prices are not (out-of-the-money
    prices peak at the forward, so straight lines between listed strikes cut under
    the peak). Within a month: linear in log-moneyness on volatility. Across months:
    linear in time on total variance (vol^2 x time). Never extrapolates; nothing is
    imputed.
    """
    months = {}
    for c, p in zip(contracts, prices):
        if c.known_fixing_sum != 0 or len(c.fixing_times) != c.total_fixings or not p > 0:
            continue
        forward = float(np.mean(c.forwards))
        if (c.side == "P") != (c.strike < forward):
            continue
        months.setdefault((c.fixing_times, c.forwards), []).append(
            (np.log(c.strike / forward), p / (c.discount_factor * forward))
        )
    grid_x = np.asarray(grid.log_moneyness)
    curves = []
    for (times, _), quotes in months.items():
        tau = float(times[-1])
        x, price = np.asarray(quotes).T
        vol = implied_vol(x, price, tau)
        keep = np.isfinite(vol)
        x, vol = x[keep], vol[keep]
        x, inverse = np.unique(x, return_inverse=True)
        vol = np.bincount(inverse, weights=vol) / np.bincount(inverse)
        if len(x) < 2:
            continue
        values = np.full(len(grid_x), np.nan)
        inside = (grid_x >= x[0]) & (grid_x <= x[-1])
        values[inside] = np.interp(grid_x[inside], x, vol)
        curves.append((tau, values))
    curves.sort(key=lambda curve: curve[0])
    tenors = np.array([t for t, _ in curves])
    out = np.full((len(grid.tenors), len(grid_x)), np.nan)
    for i, tau in enumerate(grid.tenors):
        if not len(tenors) or tau < tenors[0] or tau > tenors[-1]:
            continue
        hi = int(np.searchsorted(tenors, tau))
        if np.isclose(tenors[hi], tau, rtol=0, atol=1e-12):
            vol = curves[hi][1]
        else:
            lo, w = hi - 1, (tau - tenors[hi - 1]) / (tenors[hi] - tenors[hi - 1])
            total = (1 - w) * curves[lo][1] ** 2 * tenors[lo] + w * curves[hi][1] ** 2 * tenors[hi]
            vol = np.sqrt(total / tau)
        out[i] = black_otm(grid_x, vol, tau)
    return out.ravel()


@dataclass(frozen=True)
class MarketPanel:
    """Complete grid vectors, one row per day, sorted by date."""

    dates: tuple[str, ...]
    vectors: np.ndarray
    grid: GridSpec
    incomplete: tuple[str, ...] = ()


def build_panel(days, grid):
    """(date, contracts, prices) per day -> panel of days that cover the whole grid."""
    complete, incomplete = {}, []
    for day, contracts, prices in days:
        day = str(pd.Timestamp(day).date())
        if day in complete or day in incomplete:
            raise ValueError(f"Duplicate day {day}")
        vector = day_vector(contracts, prices, grid)
        if np.isfinite(vector).all():
            complete[day] = vector
        else:
            incomplete.append(day)
    if not complete:
        raise ValueError("No day covers the whole grid; narrow the grid or add data")
    dates = tuple(sorted(complete))
    return MarketPanel(
        dates, np.array([complete[d] for d in dates]), grid, tuple(sorted(incomplete))
    )


def split_months(dates, test_months, embargo_days=0):
    """Boolean masks (train, test, embargoed). Test months never reach training."""
    days = pd.to_datetime(pd.Series(list(dates)))
    months = days.dt.strftime("%Y-%m")
    test_months = tuple(test_months)
    if not test_months or len(set(test_months)) != len(test_months):
        raise ValueError("Distinct test months required")
    for month in test_months:
        if str(pd.Period(month, "M")) != month:
            raise ValueError(f"Test month {month!r} must be YYYY-MM")
    test = months.isin(test_months).to_numpy()
    missing = set(test_months) - set(months[test])
    if missing:
        raise ValueError(f"No days in test months {sorted(missing)}")
    if embargo_days < 0:
        raise ValueError("embargo_days must be nonnegative")
    embargo = np.zeros(len(days), dtype=bool)
    if embargo_days:
        gaps = np.abs(days.to_numpy()[:, None] - days[test].to_numpy()[None, :]).min(axis=1)
        embargo = ~test & (gaps <= np.timedelta64(embargo_days, "D"))
    train_mask = ~test & ~embargo
    if not train_mask.any():
        raise ValueError("No training days left")
    return train_mask, test, embargo


OPTION_COLUMNS = [
    "observation_date",
    "contract_id",
    "contract_month",
    "strike",
    "call_put",
    "settlement_price",
    "discount_factor",
    "prototype_input_complete",
    "elapsed_fixings",
    "known_fixing_sum",
    "total_fixings",
    "remaining_fixings",
    "provisional_remaining_forwards",
]
FIXING_COLUMNS = [
    "observation_date",
    "contract_month",
    "fixing_date",
    "current_forward",
    "forward_final_flag",
    "fixing_time_years",
]


def load_archive_days(root, start=None, end=None):
    """(date, contracts, prices) per day from the local archive. Nothing is imputed.

    Keeps prototype-complete, unseasoned options whose every fixing has a final,
    positive forward. Seasoned months stay out of the fixed grid entirely.
    """
    root = Path(root)
    option_file = root / "ao_calibration_inputs_provisional.parquet"
    fixing_file = root / "ao_model_fixing_inputs.parquet"
    for path in (option_file, fixing_file):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; the full local archive is required")
    filters = []
    if start:
        filters.append(("observation_date", ">=", pd.Timestamp(start).date()))
    if end:
        filters.append(("observation_date", "<=", pd.Timestamp(end).date()))
    options = pd.read_parquet(option_file, columns=OPTION_COLUMNS, filters=filters or None)
    options = options[
        options.prototype_input_complete.eq(True)
        & options.elapsed_fixings.eq(0)
        & options.known_fixing_sum.eq(0)
        & options.provisional_remaining_forwards.eq(0)
        & options.remaining_fixings.eq(options.total_fixings)
        & (options.settlement_price > 0)
    ]
    fixings = pd.read_parquet(fixing_file, columns=FIXING_COLUMNS, filters=filters or None)
    schedules = {}
    for (day, month), f in fixings.groupby(["observation_date", "contract_month"], sort=False):
        f = f.sort_values("fixing_date")
        times = f.fixing_time_years.to_numpy(dtype=float)
        forwards = f.current_forward.to_numpy(dtype=float)
        if (
            not f.forward_final_flag.eq(True).all()
            or not np.isfinite(forwards).all()
            or (forwards <= 0).any()
            or (times <= 0).any()
            or (np.diff(times) <= 0).any()
        ):
            continue
        schedules[(pd.Timestamp(day), month)] = (tuple(times.tolist()), tuple(forwards.tolist()))
    days = []
    for day, group in options.groupby("observation_date", sort=True):
        day = pd.Timestamp(day)
        contracts, prices = [], []
        for row in group.itertuples():
            schedule = schedules.get((day, row.contract_month))
            if schedule is None or len(schedule[0]) != row.total_fixings:
                continue
            contracts.append(
                AsianContract(
                    str(row.contract_id),
                    schedule[0],
                    schedule[1],
                    int(row.total_fixings),
                    0.0,
                    float(row.strike),
                    row.call_put,
                    float(row.discount_factor),
                )
            )
            prices.append(float(row.settlement_price))
        if contracts:
            days.append((str(day.date()), tuple(contracts), tuple(prices)))
    return days


def grid_scale(contracts, grid):
    """Discount x average forward at each grid tenor for one day: normalized -> USD/bbl."""
    months = {}
    for c in contracts:
        if c.known_fixing_sum == 0 and len(c.fixing_times) == c.total_fixings:
            months[c.fixing_times[-1]] = c.discount_factor * float(np.mean(c.forwards))
    if len(months) < 2:
        raise ValueError("At least two unseasoned months are needed for a day's scale")
    tenors = np.array(sorted(months))
    taus = np.asarray(grid.tenors, dtype=float)
    if taus.min() < tenors[0] or taus.max() > tenors[-1]:
        raise ValueError("Grid tenors fall outside this day's averaging months")
    per_tenor = np.interp(taus, tenors, [months[t] for t in tenors])
    return np.repeat(per_tenor, len(grid.log_moneyness))
