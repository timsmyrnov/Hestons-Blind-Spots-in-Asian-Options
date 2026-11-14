"""Market manifold vs Heston manifold on one fixed price grid.

Every day becomes one vector: forward-normalized out-of-the-money AO prices on a
fixed (time-to-last-fixing x log-moneyness) grid, interpolated from that day's
quotes. On this grid Heston gives ONE manifold (unit forward and discount, flat
curve within the averaging month). The market manifold is learned from the
training days with the same injective architecture. For each day:

    market prices        -> nearest point on the market manifold
    that manifold point  -> nearest point on the Heston manifold = manifold distance

The chosen test months are excluded from everything that is fitted (market
manifold, its scaling and initialization) and scored afterwards.

    python -m hestonlab.market_manifold synthetic --quick
    python -m hestonlab.market_manifold archive --root R --test-months 2022-03 2023-11 2025-06
"""

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares

from .calibration import PARAMETER_NAMES
from .heston import AsianContract, MonteCarloConfig
from .manifold import ManifoldNetwork, geometry_diagnostics
from .manifold_calibration import rms
from .surrogate import (
    DTYPE, OutputScaler, TrainConfig, evaluate, price_error_scale, set_seeds, train,
)
from .surrogate_data import sample_parameters

DEFAULT_OUT = Path("outputs/market_manifold")
SCALE_KEY = "price_error_scale_normalized"


# ---- fixed coordinates -----------------------------------------------------------------


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
            if (values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all()
                    or (np.diff(values) <= 0).any()):
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
        for x, cid in zip(grid.log_moneyness, grid.coordinate_ids[len(contracts):]):
            contracts.append(AsianContract(cid, times, (1.0,) * grid.fixings, grid.fixings,
                                           0.0, float(np.exp(x)), "P" if x < 0 else "C", 1.0))
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
            (np.log(c.strike / forward), p / (c.discount_factor * forward)))
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
    return MarketPanel(dates, np.array([complete[d] for d in dates]), grid,
                       tuple(sorted(incomplete)))


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


# ---- learned market manifold -----------------------------------------------------------


@dataclass(frozen=True)
class MarketConfig:
    latent_dim: int = 5
    width: int = 64
    depth: int = 4
    epochs: int = 4000
    learning_rate: float = 2e-3
    lr_patience: int = 100
    min_learning_rate: float = 1e-5
    seed: int = 0


class MarketManifold:
    """Latent unit box [0, 1]^d -> normalized grid prices, an embedded d-manifold."""

    def __init__(self, network, output_scaler, coordinate_ids, train_units, metadata=None):
        self.network = network.eval()
        self.output_scaler = output_scaler
        self.coordinate_ids = tuple(coordinate_ids)
        self.train_units = np.asarray(train_units, dtype=float)
        self.metadata = dict(metadata or {})

    @property
    def latent_dim(self):
        return self.network.latent_dim

    def prices_from_unit(self, unit):
        mean = torch.as_tensor(self.output_scaler.mean, dtype=DTYPE)
        std = torch.as_tensor(self.output_scaler.std, dtype=DTYPE)
        return self.network(unit) * std + mean

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        net = self.network
        torch.save({
            "format": "market_manifold_v1", "state_dict": net.state_dict(),
            "n_outputs": net.n_outputs, "width": net.width, "depth": net.depth,
            "latent_dim": net.latent_dim,
            "output_mean": self.output_scaler.mean.tolist(),
            "output_std": self.output_scaler.std.tolist(),
            "coordinate_ids": list(self.coordinate_ids),
            "train_units": torch.as_tensor(self.train_units, dtype=DTYPE),
            "metadata": json.dumps(self.metadata),
        }, path)

    @classmethod
    def load(cls, path):
        blob = torch.load(Path(path), weights_only=True)
        if blob.get("format") != "market_manifold_v1":
            raise ValueError("Not a market manifold checkpoint")
        network = ManifoldNetwork(blob["n_outputs"], blob["width"], blob["depth"],
                                  blob["latent_dim"])
        network.load_state_dict(blob["state_dict"])
        return cls(network, OutputScaler(blob["output_mean"], blob["output_std"]),
                   blob["coordinate_ids"], blob["train_units"].numpy(),
                   json.loads(blob["metadata"]))


def train_market_manifold(vectors, coordinate_ids, config=MarketConfig()):
    """Auto-decoder: one latent point per training day, learned with the decoder.

    PCA of the TRAINING days initializes both the latent points and the decoder's
    linear map. The decoder is injective by construction, so its image is an
    embedded latent_dim-manifold. Only the vectors passed in are ever seen.
    """
    y = np.asarray(vectors, dtype=float)
    d = config.latent_dim
    if y.ndim != 2 or y.shape[1] != len(coordinate_ids) or not np.isfinite(y).all():
        raise ValueError("Finite (days, grid size) training vectors required")
    if len(y) < d + 2:
        raise ValueError(f"At least {d + 2} training days required for a {d}D manifold")
    if config.epochs < 1:
        raise ValueError("Positive epoch count required")
    set_seeds(config.seed)
    scaler = OutputScaler.fit(y)
    ys = scaler.transform(y)
    centered = ys - ys.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ vt[:d].T
    low, high = scores.min(axis=0), scores.max(axis=0)
    # Training days fill the middle half of the box: room for more extreme test days.
    units = 0.25 + 0.5 * (scores - low) / np.where(high > low, high - low, 1.0)
    network = ManifoldNetwork(y.shape[1], config.width, config.depth, latent_dim=d)
    network.initialize_from_data(units, ys)
    raw = torch.nn.Parameter(torch.logit(torch.as_tensor(units, dtype=DTYPE)))
    target = torch.as_tensor(ys, dtype=DTYPE)
    optimizer = torch.optim.Adam([*network.parameters(), raw], lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=config.lr_patience, min_lr=config.min_learning_rate)
    best, stale, best_state = np.inf, 0, None
    network.train()
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        loss = torch.mean((network(torch.sigmoid(raw)) - target) ** 2)
        value = loss.item()
        if not np.isfinite(value):
            raise ValueError("Nonfinite market manifold loss")
        if value < best:
            best, stale = value, 0
            best_state = ({k: v.clone() for k, v in network.state_dict().items()},
                          raw.detach().clone())
        else:
            stale += 1
        loss.backward()
        optimizer.step()
        scheduler.step(value)
        lr = optimizer.param_groups[0]["lr"]
        if lr <= config.min_learning_rate * 1.0001 and stale > 2 * config.lr_patience:
            break
    network.load_state_dict(best_state[0])
    metadata = {"config": asdict(config), "epochs": epoch + 1, "training_days": len(y),
                "best_train_mse_standardized": best}
    return MarketManifold(network, scaler, coordinate_ids,
                          torch.sigmoid(best_state[1]).numpy(), metadata)


# ---- distances -------------------------------------------------------------------------


def evaluate_units(model, units):
    with torch.no_grad():
        return model.prices_from_unit(torch.as_tensor(np.asarray(units, dtype=float),
                                                      dtype=DTYPE)).numpy()


def nearest_point(model, target, weight, cloud_units, cloud_prices, starts=4,
                  max_evaluations=200):
    """Closest point on a learned manifold in the weighted Euclidean metric.

    Bounded least squares over the manifold's unit box, started from the `starts`
    closest points of a precomputed cloud on that manifold. Local, not certified global.
    """
    target, weight = np.asarray(target, dtype=float), np.asarray(weight, dtype=float)
    gaps = (((np.asarray(cloud_prices) - target) * weight) ** 2).sum(axis=1)

    def residual(unit):
        return (evaluate_units(model, unit) - target) * weight

    def jacobian(unit):
        jac = torch.func.jacrev(model.prices_from_unit)(torch.as_tensor(unit, dtype=DTYPE))
        return jac.detach().numpy() * weight[:, None]

    best = None
    for i in np.argsort(gaps)[:max(1, starts)]:
        result = least_squares(residual, np.clip(cloud_units[i], 0, 1), jac=jacobian,
                               bounds=(0, 1), max_nfev=max_evaluations,
                               ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if best is None or result.cost < best.cost:
            best = result
    distance = rms(best.fun)
    if not np.isfinite(distance):
        raise ValueError("Nonfinite nearest-point distance")
    return {"unit": best.x, "point": evaluate_units(model, best.x), "distance": distance,
            "success": bool(best.success),
            "boundary": bool((np.minimum(best.x, 1 - best.x) < 1e-6).any())}


def compare_manifolds(panel, masks, heston, market, weight, heston_cloud, starts=4,
                      max_evaluations=200):
    """Per day: market -> market manifold point -> Heston manifold (manifold distance)."""
    _, test, embargo = masks
    market_cloud = (market.train_units, evaluate_units(market, market.train_units))
    rows = []
    for i, day in enumerate(panel.dates):
        if embargo[i]:
            continue
        y = panel.vectors[i]
        on_market = nearest_point(market, y, weight, *market_cloud, starts, max_evaluations)
        to_heston = nearest_point(heston, on_market["point"], weight, *heston_cloud, starts,
                                  max_evaluations)
        direct = nearest_point(heston, y, weight, *heston_cloud, starts, max_evaluations)
        params = heston.input_scaler.from_unit(to_heston["unit"])
        rows.append({
            "date": day, "month": day[:7], "split": "test" if test[i] else "train",
            "manifold_distance": to_heston["distance"],
            "market_to_market_manifold": on_market["distance"],
            "market_to_heston": direct["distance"],
            "heston_parameters": dict(zip(PARAMETER_NAMES, params.tolist())),
            "heston_boundary": to_heston["boundary"],
            "market_latent": on_market["unit"].tolist(),
            "market_latent_boundary": on_market["boundary"],
        })
    return rows


DISTANCES = ("manifold_distance", "market_to_market_manifold", "market_to_heston")


def summarize(rows):
    def block(selected):
        out = {"days": len(selected)}
        for key in DISTANCES:
            values = np.array([r[key] for r in selected])
            out[key] = ({"median": float(np.median(values)), "mean": float(values.mean()),
                         "p90": float(np.quantile(values, 0.9)), "max": float(values.max())}
                        if len(values) else None)
        return out

    test_months = sorted({r["month"] for r in rows if r["split"] == "test"})
    return {
        "train": block([r for r in rows if r["split"] == "train"]),
        "test": block([r for r in rows if r["split"] == "test"]),
        "by_test_month": {m: block([r for r in rows if r["split"] == "test" and r["month"] == m])
                          for m in test_months},
    }


# ---- pipeline --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    grid: GridSpec = field(default_factory=GridSpec)
    heston_draws: int = 8192
    heston_test_draws: int = 1024
    heston_mc: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    heston_train: TrainConfig = field(default_factory=TrainConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    weighting: str = "noise"  # residual / Heston-grid error scale; or "equal"
    scale_floor: float = 1e-4  # about one $0.01 tick at $80/bbl, forward-normalized
    heston_cloud: int = 4096
    starts: int = 4
    max_evaluations: int = 200


QUICK = RunConfig(
    heston_draws=512, heston_test_draws=128,
    heston_train=TrainConfig(max_epochs=300, lr_patience=30, batch_size=32, learning_rate=1e-2),
    market=MarketConfig(epochs=800, lr_patience=50),
    heston_cloud=1024, starts=2, max_evaluations=100,
)


def train_heston_grid(config, out_dir, log=print):
    """ONE Heston manifold on the canonical grid; uses no market data at all."""
    from .surrogate_pipeline import cached_generate

    contracts = heston_grid_contracts(config.grid)
    folder = Path(out_dir) / "heston"
    train_data = cached_generate(contracts, config.heston_draws, config.heston_mc, 0,
                                 folder / "data/train.npz", log)
    test_data = cached_generate(contracts, config.heston_test_draws, config.heston_mc, 1,
                                folder / "data/test.npz", log)
    log("training Heston manifold on the grid ...")
    model, _ = train(train_data, contracts, config.heston_train)
    scale = price_error_scale(model, test_data, config.scale_floor)
    model.metadata[SCALE_KEY] = scale.tolist()
    model.save(folder / "model.pt")
    return model, scale, evaluate(model, test_data)


def run(panel, test_months, out_dir, config=RunConfig(), embargo_days=0, label="",
        heston=None, log=print):
    """Split by month, learn the market manifold on training days, score every day."""
    if config.weighting not in ("noise", "equal"):
        raise ValueError("weighting must be noise or equal")
    if panel.grid != config.grid:
        raise ValueError("Panel and run configuration use different grids")
    out_dir = Path(out_dir)
    masks = split_months(panel.dates, test_months, embargo_days)
    train_mask, test_mask, embargo = masks
    log(f"[{label}] {int(train_mask.sum())} training days, {int(test_mask.sum())} test days "
        f"in {list(test_months)}, {int(embargo.sum())} embargoed")
    heston, scale, heston_accuracy = heston or train_heston_grid(config, out_dir, log)
    weight = 1 / scale if config.weighting == "noise" else np.ones(config.grid.size)
    log("training market manifold on training days only ...")
    market = train_market_manifold(panel.vectors[train_mask], config.grid.coordinate_ids,
                                   config.market)
    training_dates = [d for d, keep in zip(panel.dates, train_mask) if keep]
    market.metadata.update({"training_dates": training_dates, "test_months": list(test_months)})
    market.save(out_dir / "market_model.pt")
    geometry = geometry_diagnostics(market, market.train_units)
    units = heston.input_scaler.to_unit(sample_parameters(config.heston_cloud, 7))
    log("projecting every day onto both manifolds ...")
    rows = compare_manifolds(panel, masks, heston, market, weight,
                             (units, evaluate_units(heston, units)),
                             config.starts, config.max_evaluations)
    report = {
        "label": label,
        "status": "experimental; not validated",
        "primary": "manifold_distance: per day, distance from the day's point on the market "
                   "manifold to the nearest point on the Heston manifold",
        "metric": ("residual / Heston-grid pricing error scale (noise units)"
                   if config.weighting == "noise" else "forward-normalized price units"),
        "test_months": list(test_months),
        "embargo_days": embargo_days,
        "days": {"train": int(train_mask.sum()), "test": int(test_mask.sum()),
                 "embargoed": int(embargo.sum()), "incomplete_dropped": len(panel.incomplete)},
        "test_days_in_training": bool(set(training_dates)
                                      & {d for d, t in zip(panel.dates, test_mask) if t}),
        "summary": summarize(rows),
        "grid": asdict(config.grid),
        "coordinate_ids": list(config.grid.coordinate_ids),
        "heston_manifold": {"test_accuracy_normalized": heston_accuracy,
                            "error_scale_normalized": np.asarray(scale).tolist()},
        "market_manifold": {"latent_dim": market.latent_dim, "geometry": geometry,
                            **{k: v for k, v in market.metadata.items()
                               if k != "training_dates"}},
        "config": asdict(config),
        "rows": rows,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    flat = [{k: v for k, v in r.items() if k not in ("heston_parameters", "market_latent")}
            | r["heston_parameters"] for r in rows]
    pd.DataFrame(flat).to_csv(out_dir / "days.csv", index=False)
    log(summary_text(report))
    return report


def summary_text(report):
    s = report["summary"]
    lines = [f"[{report['label']}] manifold distance per day ({report['metric']})",
             f"  {'days':<10} {'n':>5} {'median':>10} {'p90':>10}   to market manifold (median)"]
    blocks = [("train", s["train"]), *s["by_test_month"].items()]
    for name, block in blocks:
        md, mm = block["manifold_distance"], block["market_to_market_manifold"]
        lines.append(f"  {name:<10} {block['days']:>5} {md['median']:>10.4g} {md['p90']:>10.4g}"
                     f"   {mm['median']:.4g}")
    return "\n".join(lines)


# ---- sources ---------------------------------------------------------------------------


def synthetic_panel(heston, grid, days=500, start="2021-01-04", blind_spot=0.0, noise=0.0,
                    seed=0):
    """SYNTHETIC market days for testing the method only, never presented as market data.

    Heston parameters wander smoothly through the box; prices come from the Heston
    manifold. `blind_spot` adds a time-varying lift of the shortest tenor's wings
    only, a shape Heston cannot produce on its own.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=days)
    units = np.empty((days, 5))
    units[0] = rng.uniform(0.3, 0.7, 5)
    for t in range(1, days):
        step = units[t - 1] + rng.normal(0, 0.02, 5)
        units[t] = 0.95 - np.abs((step - 0.05) % 1.8 - 0.9)  # reflect into [0.05, 0.95]
    y = evaluate_units(heston, units)
    if blind_spot:
        x = np.asarray(grid.log_moneyness)
        shape = np.zeros((len(grid.tenors), len(x)))
        shape[0] = (x / np.abs(x).max()) ** 2
        amplitude = blind_spot * (0.5 + 0.5 * np.sin(2 * np.pi * np.arange(days) / 63))
        y = y + amplitude[:, None] * shape.ravel()[None, :]
    if noise:
        y = y * (1 + noise * rng.standard_normal(y.shape))
    return MarketPanel(tuple(str(d.date()) for d in dates), y, grid)


OPTION_COLUMNS = [
    "observation_date", "contract_id", "contract_month", "strike", "call_put",
    "settlement_price", "discount_factor", "prototype_input_complete", "elapsed_fixings",
    "known_fixing_sum", "total_fixings", "remaining_fixings", "provisional_remaining_forwards",
]
FIXING_COLUMNS = [
    "observation_date", "contract_month", "fixing_date", "current_forward",
    "forward_final_flag", "fixing_time_years",
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
        options.prototype_input_complete.eq(True) & options.elapsed_fixings.eq(0)
        & options.known_fixing_sum.eq(0) & options.provisional_remaining_forwards.eq(0)
        & options.remaining_fixings.eq(options.total_fixings) & (options.settlement_price > 0)
    ]
    fixings = pd.read_parquet(fixing_file, columns=FIXING_COLUMNS, filters=filters or None)
    schedules = {}
    for (day, month), f in fixings.groupby(["observation_date", "contract_month"], sort=False):
        f = f.sort_values("fixing_date")
        times = f.fixing_time_years.to_numpy(dtype=float)
        forwards = f.current_forward.to_numpy(dtype=float)
        if (not f.forward_final_flag.eq(True).all() or not np.isfinite(forwards).all()
                or (forwards <= 0).any() or (times <= 0).any() or (np.diff(times) <= 0).any()):
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
            contracts.append(AsianContract(
                str(row.contract_id), schedule[0], schedule[1], int(row.total_fixings), 0.0,
                float(row.strike), row.call_put, float(row.discount_factor)))
            prices.append(float(row.settlement_price))
        if contracts:
            days.append((str(day.date()), tuple(contracts), tuple(prices)))
    return days


def main(argv=None):
    from .surrogate_pipeline import check_output_location

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="source", required=True)
    syn = sub.add_parser("synthetic", help="SYNTHETIC market days; tests the method only")
    syn.add_argument("--days", type=int, default=500)
    syn.add_argument("--blind-spot", type=float, default=0.003,
                     help="size of a non-Heston short-tenor wing lift, normalized units")
    syn.add_argument("--noise", type=float, default=0.0, help="relative price noise")
    syn.add_argument("--test-months", nargs="+", default=["2021-06", "2021-10", "2022-02"])
    arc = sub.add_parser("archive", help="real AO days from the private archive")
    arc.add_argument("--root", type=Path, required=True)
    arc.add_argument("--test-months", nargs="+", required=True, metavar="YYYY-MM")
    arc.add_argument("--start", help="first observation date (default: all)")
    arc.add_argument("--end", help="last observation date (default: all)")
    for p in (syn, arc):
        p.add_argument("--out", type=Path)
        p.add_argument("--quick", action="store_true", help="tiny sizes, smoke test only")
        p.add_argument("--latent-dim", type=int, default=5, help="market manifold dimension")
        p.add_argument("--embargo-days", type=int, default=0,
                       help="also drop training days this close to a test day")
        p.add_argument("--weighting", choices=("noise", "equal"), default="noise")
    args = parser.parse_args(argv)
    config = QUICK if args.quick else RunConfig()
    config = replace(config, weighting=args.weighting,
                     market=replace(config.market, latent_dim=args.latent_dim))
    try:
        out = check_output_location(args.out or DEFAULT_OUT / args.source)
        if args.source == "archive":
            if args.root.resolve() in (out, *out.parents):
                raise ValueError("Write outputs outside the preserved data archive")
            panel = build_panel(load_archive_days(args.root, args.start, args.end), config.grid)
            return run(panel, args.test_months, out, config, args.embargo_days, "archive")
        heston = train_heston_grid(config, out)
        panel = synthetic_panel(heston[0], config.grid, args.days, blind_spot=args.blind_spot,
                                noise=args.noise)
        return run(panel, args.test_months, out, config, args.embargo_days, "synthetic", heston)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Market manifold run unavailable: {exc}\n")


if __name__ == "__main__":
    main()
