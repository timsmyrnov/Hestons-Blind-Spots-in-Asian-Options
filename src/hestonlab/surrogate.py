"""Neural surrogate: 5 Heston parameters -> Monte Carlo AO prices on ONE fixed contract set.

The network approximates the tested Monte Carlo pricer (heston.py) at a fixed random
stream, so the calibrator can call it instead of simulating. Its error is SURROGATE
APPROXIMATION ERROR against Monte Carlo; it says nothing about how well Heston fits
the market and must never be reported as Heston model error.

One network per contract set (per observation date). Everything here takes a plain
tuple of AsianContract; data loaders are separate sources of contract sets.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
import ast
import hashlib
import json
import numpy as np
import torch
from torch import nn
from .calibration import LOWER, PARAMETER_NAMES, UPPER
from .heston import HestonParameters, MonteCarloConfig, price_asian_options
from .surrogate_data import LOG_SCALED, sample_parameters
from .manifold import ManifoldNetwork

DTYPE = torch.float64
FORMAT_VERSION = 2
# Same convention as calibrate_june's independent-stream check.
INDEPENDENT_SEED_OFFSET = 1_000_000
_BOUND_TOLERANCE = 1e-9  # Absorbs exp(log(x)) round-off at the box edges only.


def contract_fingerprint(contracts):
    """Stable hash of a contract set, so a model is never used on the wrong grid."""
    payload = json.dumps([asdict(c) for c in contracts], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def as_parameter_array(params):
    """HestonParameters, a dict, a (5,) or an (M, 5) array -> float array, columns in
    PARAMETER_NAMES order."""
    if isinstance(params, dict):
        params = [params[name] for name in PARAMETER_NAMES]
    elif hasattr(params, "rho") and hasattr(params, "v0"):
        params = [getattr(params, name) for name in PARAMETER_NAMES]
    values = np.asarray(params, dtype=float)
    if values.shape[-1:] != (5,) or values.ndim > 2:
        raise ValueError(f"Expected 5 parameters {PARAMETER_NAMES}, got shape {values.shape}")
    return values


class InputScaler:
    """log(v0, kappa, theta, sigma), then map every parameter to [0, 1] using the
    calibration bounds. Fixed by the bounds, so nothing is fitted to data."""

    def __init__(self, lower=LOWER, upper=UPPER):
        self.lower = np.asarray(lower, dtype=float).copy()
        self.upper = np.asarray(upper, dtype=float).copy()
        self._low, self._high = self.lower.copy(), self.upper.copy()
        self._low[:LOG_SCALED] = np.log(self._low[:LOG_SCALED])
        self._high[:LOG_SCALED] = np.log(self._high[:LOG_SCALED])

    def to_unit(self, params):
        values = as_parameter_array(params)
        if not np.isfinite(values).all():
            raise ValueError("Parameters must be finite")
        span = self.upper - self.lower
        if (values < self.lower - _BOUND_TOLERANCE * span).any() or (
            values > self.upper + _BOUND_TOLERANCE * span
        ).any():
            raise ValueError(
                "Parameters outside the calibration bounds the surrogate was trained on: "
                f"lower={self.lower.tolist()}, upper={self.upper.tolist()}"
            )
        values = np.clip(values, self.lower, self.upper)
        values[..., :LOG_SCALED] = np.log(values[..., :LOG_SCALED])
        return (values - self._low) / (self._high - self._low)

    def from_unit(self, unit):
        values = self._low + np.asarray(unit, dtype=float) * (self._high - self._low)
        values[..., :LOG_SCALED] = np.exp(values[..., :LOG_SCALED])
        return values


class OutputScaler:
    """Per-contract standardization, fitted on TRAINING prices only."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)
        if (self.mean.ndim != 1 or self.std.shape != self.mean.shape
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or (self.std <= 0).any()):
            raise ValueError("Finite output means and strictly positive scales required")

    @classmethod
    def fit(cls, prices):
        prices = np.asarray(prices, dtype=float)
        # Floor guards a contract whose price barely moves (e.g. far out of the money).
        return cls(prices.mean(axis=0), np.maximum(prices.std(axis=0), 1e-8))

    def transform(self, prices):
        return (np.asarray(prices, dtype=float) - self.mean) / self.std

    def inverse(self, standardized):
        return np.asarray(standardized, dtype=float) * self.std + self.mean


class SurrogateMLP(nn.Module):
    """5 scaled parameters -> N standardized prices. Softplus keeps it smooth, so
    gradients and the Jacobian are well defined everywhere."""

    def __init__(self, n_outputs, width=64, depth=4):
        super().__init__()
        self.architecture, self.width, self.depth = "mlp", width, depth
        layers, size = [], 5
        for _ in range(depth):
            layers += [nn.Linear(size, width, dtype=DTYPE), nn.Softplus()]
            size = width
        layers.append(nn.Linear(size, n_outputs, dtype=DTYPE))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Surrogate:
    """A trained network plus everything needed to use it: both scalers, the contract
    IDs (column order of the output) and a fingerprint of the contract set."""

    def __init__(self, network, output_scaler, coordinate_ids, metadata=None, input_scaler=None):
        self.network = network.eval()
        # Default: the calibration box. A custom box comes from surrogate_box.ParameterBox.
        self.input_scaler = InputScaler() if input_scaler is None else input_scaler
        self.output_scaler = output_scaler
        self.coordinate_ids = tuple(coordinate_ids)
        self.metadata = dict(metadata or {})
        if len(self.coordinate_ids) != len(output_scaler.mean):
            raise ValueError("One output per contract required")

    @property
    def n_outputs(self):
        return len(self.coordinate_ids)

    def prices_from_unit(self, unit):
        """Differentiable: scaled-parameter tensor (..., 5) -> USD/barrel tensor (..., N)."""
        mean = torch.as_tensor(self.output_scaler.mean, dtype=DTYPE)
        std = torch.as_tensor(self.output_scaler.std, dtype=DTYPE)
        return self.network(unit) * std + mean

    def __call__(self, params):
        """Parameters ((5,) or (M, 5)) -> prices in USD/barrel ((N,) or (M, N))."""
        unit = torch.as_tensor(self.input_scaler.to_unit(params), dtype=DTYPE)
        with torch.no_grad():
            return self.prices_from_unit(unit).numpy()

    def check_contracts(self, contracts):
        if self.metadata.get("contract_fingerprint") != contract_fingerprint(contracts):
            raise ValueError("This surrogate was trained on a different contract set")

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": FORMAT_VERSION,
                "state_dict": self.network.state_dict(),
                "architecture": self.network.architecture,
                "width": self.network.width,
                "depth": self.network.depth,
                "output_mean": self.output_scaler.mean.tolist(),
                "output_std": self.output_scaler.std.tolist(),
                "input_lower": self.input_scaler.lower.tolist(),
                "input_upper": self.input_scaler.upper.tolist(),
                "coordinate_ids": list(self.coordinate_ids),
                "metadata": json.dumps(self.metadata),
            },
            path,
        )

    @classmethod
    def load(cls, path, bounds=None):
        """bounds=None expects the calibration box. Pass (lower, upper) to load a model
        trained on a custom box; any mismatch with the saved bounds is refused."""
        blob = torch.load(Path(path), weights_only=True)
        if blob["format_version"] not in (1, FORMAT_VERSION):
            raise ValueError(f"Unsupported surrogate file version {blob['format_version']}")
        lower, upper = (LOWER, UPPER) if bounds is None else bounds
        saved_lower, saved_upper = np.asarray(blob["input_lower"]), np.asarray(blob["input_upper"])
        if not (np.allclose(saved_lower, lower) and np.allclose(saved_upper, upper)):
            if bounds is None:
                raise ValueError("Saved surrogate used different calibration bounds")
            raise ValueError(
                "Saved surrogate used different parameter bounds: saved "
                f"{saved_lower.tolist()}..{saved_upper.tolist()}, requested "
                f"{np.asarray(lower).tolist()}..{np.asarray(upper).tolist()}"
            )
        architecture = "mlp" if blob["format_version"] == 1 else blob["architecture"]
        network = make_network(architecture, len(blob["coordinate_ids"]),
                               blob["width"], blob["depth"])
        network.load_state_dict(blob["state_dict"])
        if not all(torch.isfinite(v).all() for v in network.state_dict().values()):
            raise ValueError("Nonfinite network state")
        if isinstance(network, ManifoldNetwork):
            if not torch.linalg.svdvals(network.affine)[-1].item() > 0:
                raise ValueError("Singular ambient affine map")
        scaler = OutputScaler(blob["output_mean"], blob["output_std"])
        input_scaler = None if bounds is None else InputScaler(saved_lower, saved_upper)
        return cls(network, scaler, blob["coordinate_ids"], json.loads(blob["metadata"]),
                   input_scaler=input_scaler)


@dataclass(frozen=True)
class TrainConfig:
    width: int = 64
    depth: int = 4
    batch_size: int = 256
    learning_rate: float = 3e-3
    min_learning_rate: float = 1e-5
    lr_patience: int = 40  # epochs without validation improvement before halving the LR
    max_epochs: int = 3000
    validation_fraction: float = 0.1  # carved from TRAINING data, for early stopping only
    seed: int = 0
    architecture: str = "manifold"


def make_network(architecture, n_outputs, width, depth):
    if architecture not in ("manifold", "mlp"):
        raise ValueError("Architecture must be manifold or mlp")
    cls = ManifoldNetwork if architecture == "manifold" else SurrogateMLP
    return cls(n_outputs, width, depth)


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(data, contracts, config=TrainConfig(), metadata=None, verbose=False,
          input_scaler=None):
    """Fit one network to generate() output for these contracts. The test set is never
    seen here; a validation slice of the training rows drives LR decay and stopping.
    input_scaler=None uses the calibration box (see surrogate_box for custom boxes)."""
    contracts = tuple(contracts)
    ids = tuple(str(i) for i in data["coordinate_ids"])
    if ids != tuple(c.coordinate_id for c in contracts):
        raise ValueError("Training data was generated for a different contract set")
    set_seeds(config.seed)
    input_scaler = InputScaler() if input_scaler is None else input_scaler
    x = input_scaler.to_unit(data["parameters"])
    prices = np.asarray(data["prices"], dtype=float)
    if (prices.shape != (len(x), len(contracts)) or len(x) < 3
            or not np.isfinite(prices).all()):
        raise ValueError("At least three finite, aligned parameter/price rows required")
    if (config.max_epochs < 1 or config.batch_size < 1
            or not 0 < config.validation_fraction < 1):
        raise ValueError("Positive epochs/batch size and validation fraction in (0, 1) required")
    order = np.random.default_rng(config.seed).permutation(len(x))
    n_val = max(1, int(round(config.validation_fraction * len(x))))
    val, fit = order[:n_val], order[n_val:]
    if len(fit) < 2:
        raise ValueError("At least two fitting rows required after validation split")
    output_scaler = OutputScaler.fit(prices[fit])
    x_fit, x_val = (torch.as_tensor(x[i], dtype=DTYPE) for i in (fit, val))
    y_fit, y_val = (
        torch.as_tensor(output_scaler.transform(prices[i]), dtype=DTYPE) for i in (fit, val)
    )
    network = make_network(config.architecture, len(contracts), config.width, config.depth)
    if isinstance(network, ManifoldNetwork):
        network.initialize_from_data(x[fit], output_scaler.transform(prices[fit]))
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=config.lr_patience, min_lr=config.min_learning_rate
    )
    generator = torch.Generator().manual_seed(config.seed)
    best, best_state, stale, history = np.inf, None, 0, []
    for epoch in range(config.max_epochs):
        network.train()
        for batch in torch.randperm(len(x_fit), generator=generator).split(config.batch_size):
            optimizer.zero_grad()
            loss = torch.mean((network(x_fit[batch]) - y_fit[batch]) ** 2)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.no_grad():
            val_loss = float(torch.mean((network(x_val) - y_val) ** 2))
        if not np.isfinite(val_loss):
            raise ValueError("Nonfinite validation loss")
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "validation_mse_standardized": val_loss, "lr": lr})
        if val_loss < best:
            best, stale = val_loss, 0
            best_state = {k: v.clone() for k, v in network.state_dict().items()}
        else:
            stale += 1
        if verbose and epoch % 100 == 0:
            print(f"epoch {epoch}: validation MSE {val_loss:.3e}, lr {lr:.1e}")
        # Stop once the LR has bottomed out and validation has stopped improving.
        if lr <= config.min_learning_rate * 1.0001 and stale > 2 * config.lr_patience:
            break
    network.load_state_dict(best_state)
    meta = {
        "contract_fingerprint": contract_fingerprint(contracts),
        "training_points": len(fit),
        "validation_points": len(val),
        "epochs": len(history),
        "best_validation_mse_standardized": best,
        "train_config": asdict(config),
        "monte_carlo": str(data.get("monte_carlo", "")),
        **(metadata or {}),
    }
    return Surrogate(network, output_scaler, ids, meta, input_scaler=input_scaler), history


def price_points(contracts, parameters, config):
    """Monte Carlo prices and SEs at GIVEN parameter rows (generate() samples its own)."""
    prices, errors = [], []
    for row in as_parameter_array(parameters).reshape(-1, 5):
        estimates = price_asian_options(
            contracts, HestonParameters(**dict(zip(PARAMETER_NAMES, row))), config
        )
        prices.append([e.price for e in estimates])
        errors.append([e.standard_error for e in estimates])
    return np.array(prices), np.array(errors)


def error_summary(predicted, reference):
    """Surrogate approximation error in USD/barrel (never Heston model error)."""
    diff = np.asarray(predicted) - np.asarray(reference)
    return {
        "rmse_usd_per_bbl": float(np.sqrt(np.mean(diff**2))),
        "max_abs_error_usd_per_bbl": float(np.abs(diff).max()),
        "mean_abs_error_usd_per_bbl": float(np.abs(diff).mean()),
        "worst_contract_rmse_usd_per_bbl": float(np.sqrt(np.mean(diff**2, axis=0)).max()),
    }


def evaluate(model, data):
    """Surrogate vs the test labels, beside the labels' own MC standard errors."""
    if tuple(str(i) for i in data["coordinate_ids"]) != model.coordinate_ids:
        raise ValueError("Test data was generated for a different contract set")
    se = np.asarray(data["standard_errors"])
    return {
        "what": "surrogate approximation error vs Monte Carlo test labels (not model error)",
        "points": len(se),
        **error_summary(model(data["parameters"]), data["prices"]),
        "label_mc_se_mean_usd_per_bbl": float(se.mean()),
        "label_mc_se_max_usd_per_bbl": float(se.max()),
    }


PRICE_TICK = 0.01  # AO minimum price fluctuation, USD/barrel; floor for error scales.


def price_error_scale(model, data, floor=PRICE_TICK):
    """How precisely each contract can be priced, in USD/barrel, from held-out draws.

    sqrt(surrogate MSE vs labels + mean label MC variance), averaged over the whole
    parameter box and floored at one price tick (pass `floor` for other price units).
    A fit residual divided by this is in noise units: about 1 means "within our own
    pricing error" for that contract.
    """
    if tuple(str(i) for i in data["coordinate_ids"]) != model.coordinate_ids:
        raise ValueError("Data was generated for a different contract set")
    diff = model(data["parameters"]) - np.asarray(data["prices"], dtype=float)
    se = np.asarray(data["standard_errors"], dtype=float)
    return np.maximum(np.sqrt(np.mean(diff**2, axis=0) + np.mean(se**2, axis=0)), floor)


def compare_to_benchmark(predicted, stream_a, stream_b, v0):
    """Error of `predicted` against the average of two independent benchmark streams.

    a - b contains only benchmark noise, so mean((a - b)^2) / 4 estimates the noise
    variance of their average. Subtracting it from the raw MSE estimates the error
    against the noise-free Monte Carlo price ("truth" for the pricer, not the market).
    Medians and maxima cannot be corrected this way and still include benchmark noise.
    """
    predicted, a, b = (np.asarray(x, dtype=float) for x in (predicted, stream_a, stream_b))
    v0 = np.asarray(v0, dtype=float)
    average = (a + b) / 2

    def block(rows):
        diff = predicted[rows] - average[rows]
        mse = float(np.mean(diff**2))
        noise = float(np.mean((a[rows] - b[rows]) ** 2) / 4)
        return {
            "points": int(rows.sum()),
            "rmse_noise_corrected_usd_per_bbl": float(np.sqrt(max(mse - noise, 0.0))),
            "rmse_raw_usd_per_bbl": float(np.sqrt(mse)),
            "median_abs_error_usd_per_bbl": float(np.median(np.abs(diff))),
            "max_abs_error_usd_per_bbl": float(np.abs(diff).max()),
            "benchmark_noise_rms_usd_per_bbl": float(np.sqrt(noise)),
        }

    edges = np.quantile(v0, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, v0, side="right") - 1, 0, 9)
    return {
        **block(np.ones(len(v0), dtype=bool)),
        "by_v0_decile": [
            {"v0_range": [float(edges[i]), float(edges[i + 1])], **block(bins == i)}
            for i in range(10)
            if (bins == i).any()
        ],
    }


def independent_benchmark(model, contracts, test_data, n_points=256, paths=65536, seeds=None):
    """Re-price test points with TWO independent random streams and many more paths.

    Neither stream is the labels' stream. Comparing the two streams measures the
    benchmark's own noise, which is then subtracted (see compare_to_benchmark).
    Surrogate-vs-benchmark is surrogate approximation error; labels-vs-benchmark shows
    how much of it is the training stream's own Monte Carlo noise.
    """
    model.check_contracts(contracts)
    label_config = _config_from_data(test_data)
    if seeds is None:
        seeds = (
            label_config.seed + INDEPENDENT_SEED_OFFSET,
            label_config.seed + 2 * INDEPENDENT_SEED_OFFSET,
        )
    if len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError("Benchmark needs two different random streams")
    if label_config.seed in seeds:
        raise ValueError("Benchmark must use a different random stream from the labels")
    configs = [MonteCarloConfig(paths, label_config.max_step_years, s) for s in seeds]
    params = np.asarray(test_data["parameters"])[:n_points]
    (a, se_a), (b, se_b) = (price_points(contracts, params, c) for c in configs)
    labels = np.asarray(test_data["prices"])[:n_points]
    se = np.concatenate([se_a, se_b])
    return {
        "what": "surrogate approximation error vs two-stream independent Monte Carlo, "
        "benchmark noise subtracted (not model error)",
        "points": len(params),
        "benchmark_monte_carlo": [asdict(c) for c in configs],
        "stream_to_stream_rmse_usd_per_bbl": float(np.sqrt(np.mean((a - b) ** 2))),
        "surrogate_vs_benchmark": compare_to_benchmark(model(params), a, b, params[:, 0]),
        "training_stream_labels_vs_benchmark": compare_to_benchmark(labels, a, b, params[:, 0]),
        "benchmark_mc_se_mean_usd_per_bbl": float(se.mean()),
        "benchmark_mc_se_max_usd_per_bbl": float(se.max()),
    }


def subset(data, n):
    """First n draws. Sobol prefixes are nested, so learning-curve sets grow by adding
    points rather than redrawing."""
    rows = ("parameters", "prices", "standard_errors")
    if n > len(data["parameters"]):
        raise ValueError(f"Asked for {n} draws but only {len(data['parameters'])} exist")
    return {k: (np.asarray(v)[:n] if k in rows else v) for k, v in data.items()}


def learning_curve(train_data, test_data, contracts, sizes, config=TrainConfig()):
    """Train one network per training-set size; score each on the SAME test set.
    Returns (rows, models) with models keyed by size."""
    rows, models = [], {}
    for n in sorted(sizes):
        model, history = train(subset(train_data, n), contracts, config)
        rows.append({"training_draws": n, "epochs": len(history), **evaluate(model, test_data)})
        models[n] = model
    return rows, models


def jacobian_check(model, n_points=2048, seed=2):
    """N x 5 Jacobian d(price, USD/bbl) / d(scaled parameter) at fresh Sobol points.

    Scaled coordinates are the calibrator's search coordinates, so a small singular
    value is a direction in which the SURROGATE's prices barely move. This describes
    the network; near-flat directions should be cross-checked against Monte Carlo
    before being read as a property of Heston itself.
    """
    params = sample_parameters(n_points, seed)
    unit = torch.as_tensor(model.input_scaler.to_unit(params), dtype=DTYPE)
    jac = torch.func.vmap(torch.func.jacrev(model.prices_from_unit))(unit).detach()
    _, singular, vh = torch.linalg.svd(jac, full_matrices=False)
    singular, weakest = singular.numpy(), vh[:, -1, :].abs().numpy()
    # Per-parameter sensitivity: how far all N prices move per unit of ONE parameter.
    sensitivity = torch.linalg.vector_norm(jac, dim=1).numpy()
    smallest = singular[:, -1]
    report = {
        "what": "singular values of the SURROGATE Jacobian, USD/bbl per unit scaled "
        "parameter (log scale for v0, kappa, theta, sigma)",
        "points": n_points,
        "sobol_seed": seed,
        "smallest_singular_value": _quantiles(smallest),
        "largest_singular_value": _quantiles(singular[:, 0]),
        "weakest_direction_mean_abs_loading": dict(
            zip(PARAMETER_NAMES, weakest.mean(axis=0).round(4).tolist())
        ),
        "parameter_sensitivity": {
            name: _quantiles(sensitivity[:, j]) for j, name in enumerate(PARAMETER_NAMES)
        },
        "by_parameter_decile": {
            name: _decile_table(params[:, j], smallest, weakest, sensitivity)
            for j, name in enumerate(PARAMETER_NAMES)
        },
    }
    arrays = {
        "parameters": params,
        "singular_values": singular,
        "weakest_direction": weakest,
        "parameter_sensitivity": sensitivity,
    }
    return report, arrays


def flat_direction_check(
    model, contracts, label_config, n_points=6, steps=(0.1, 0.2), paths=65536, seeds=None,
    sobol_seed=3,
):
    """Is the surrogate's flattest Jacobian direction flat in Monte Carlo too?

    At interior points, step along the surrogate's flattest direction, and along kappa
    alone and theta alone as controls, by the same distance in scaled coordinates. Each
    step is re-priced by the surrogate and by two independent high-path Monte Carlo
    streams. Within a stream every step reuses the same random numbers; kappa and theta
    do not change how those numbers enter the price, so these differences are smooth
    and precise. Agreement between the two streams shows the Monte Carlo moves are
    not noise; agreement with the surrogate shows the network's flatness is real.
    """
    model.check_contracts(contracts)
    if seeds is None:
        seeds = tuple(label_config.seed + k * INDEPENDENT_SEED_OFFSET for k in (3, 4))
    if len(set(seeds)) != 2 or label_config.seed in seeds:
        raise ValueError("Need two different random streams, neither the labels' stream")
    configs = [MonteCarloConfig(paths, label_config.max_step_years, s) for s in seeds]
    scaler = model.input_scaler
    candidates = scaler.to_unit(sample_parameters(256, sobol_seed))
    reach = max(steps)
    inside = ((candidates > reach) & (candidates < 1 - reach)).all(axis=1)
    if inside.sum() < n_points:
        raise ValueError("Steps too large to stay inside the parameter box")
    points = candidates[inside][:n_points]
    times = np.array(sorted({t for c in contracts for t in c.fixing_times}))
    k_axis, t_axis = PARAMETER_NAMES.index("kappa"), PARAMETER_NAMES.index("theta")
    results = []
    for u in points:
        jac = torch.func.jacrev(model.prices_from_unit)(torch.as_tensor(u, dtype=DTYPE))
        _, singular, vh = torch.linalg.svd(jac.detach(), full_matrices=False)
        flat = vh[-1].numpy()
        flat = flat if flat[k_axis] >= 0 else -flat  # fix the sign: kappa up
        directions = {"flattest": flat, "kappa_only": np.eye(5)[k_axis],
                      "theta_only": np.eye(5)[t_axis]}
        moves = [(name, s * h) for name in directions for h in steps for s in (-1, 1)]
        units = np.array([u] + [u + h * directions[name] for name, h in moves])
        params = scaler.from_unit(units)
        surrogate = model(params)
        mc = [price_points(contracts, params, c)[0] for c in configs]
        variance = _mean_expected_variance(params, times)

        def rms_move(prices, i):
            return float(np.sqrt(np.mean((prices[i] - prices[0]) ** 2)))

        rows = []
        for i, (name, h) in enumerate(moves, start=1):
            rows.append({
                "direction": name,
                "step": float(h),
                "parameters": dict(zip(PARAMETER_NAMES, params[i].round(6).tolist())),
                "surrogate_rms_move_usd_per_bbl": rms_move(surrogate, i),
                "mc_rms_move_usd_per_bbl": [rms_move(m, i) for m in mc],
                "mean_expected_variance_relative_change": float(variance[i] / variance[0] - 1),
            })
        results.append({
            "parameters": dict(zip(PARAMETER_NAMES, params[0].round(6).tolist())),
            "flattest_direction": dict(zip(PARAMETER_NAMES, flat.round(4).tolist())),
            "surrogate_singular_values": singular.numpy().tolist(),
            # Level noise of one stream, for scale (CRN step differences are far smaller).
            "mc_level_noise_rms_usd_per_bbl": float(
                np.sqrt(np.mean((mc[0][0] - mc[1][0]) ** 2) / 2)
            ),
            "moves": rows,
        })
    return {
        "what": "Monte Carlo re-pricing along the SURROGATE's flattest direction, with "
        "kappa-only and theta-only steps of equal length as controls; RMS price move "
        "per contract in USD/bbl",
        "monte_carlo": [asdict(c) for c in configs],
        "steps_scaled_units": list(steps),
        "points": results,
        "summary": _flat_summary(results, max(steps)),
    }


def _mean_expected_variance(params, times):
    """Heston E[average variance over (0, t)], averaged over the fixing times. Explains
    the flat direction: prices of short averages depend mostly on this one number."""
    v0, kappa, theta = params[:, :1], params[:, 1:2], params[:, 2:3]
    decay = (1 - np.exp(-kappa * times)) / (kappa * times)
    return (theta + (v0 - theta) * decay).mean(axis=1)


def _flat_summary(results, step):
    def at(direction, key):
        out = []
        for r in results:
            for m in r["moves"]:
                if m["direction"] == direction and m["step"] == step:
                    out.append(m[key])
        return np.array(out)

    flat_mc, kappa_mc = at("flattest", "mc_rms_move_usd_per_bbl"), at(
        "kappa_only", "mc_rms_move_usd_per_bbl")
    theta_mc = at("theta_only", "mc_rms_move_usd_per_bbl")
    flat_sur = at("flattest", "surrogate_rms_move_usd_per_bbl")
    flat_mean = flat_mc.mean(axis=1)
    return {
        "step": step,
        "median_mc_move_flattest": float(np.median(flat_mean)),
        "median_mc_move_kappa_only": float(np.median(kappa_mc.mean(axis=1))),
        "median_mc_move_theta_only": float(np.median(theta_mc.mean(axis=1))),
        "median_surrogate_move_flattest": float(np.median(flat_sur)),
        "max_stream_disagreement_flattest": float(np.abs(flat_mc[:, 0] - flat_mc[:, 1]).max()),
        "median_surrogate_minus_mc_flattest": float(np.median(flat_sur - flat_mean)),
    }


def _quantiles(values):
    q = np.quantile(values, [0, 0.01, 0.1, 0.5, 0.9, 1])
    return dict(zip(("min", "p01", "p10", "median", "p90", "max"), q.tolist()))


def _decile_table(column, smallest, weakest, sensitivity):
    edges = np.quantile(column, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, column, side="right") - 1, 0, 9)
    table = []
    for b in range(10):
        hit = bins == b
        table.append(
            {
                "range": [float(edges[b]), float(edges[b + 1])],
                "median_smallest_singular_value": float(np.median(smallest[hit])),
                "min_smallest_singular_value": float(smallest[hit].min()),
                "weakest_direction_mean_abs_loading": dict(
                    zip(PARAMETER_NAMES, weakest[hit].mean(axis=0).round(4).tolist())
                ),
                "median_parameter_sensitivity": dict(
                    zip(PARAMETER_NAMES, np.median(sensitivity[hit], axis=0).tolist())
                ),
            }
        )
    return table


def plot_learning_curve(rows, benchmark, path):
    plt = _pyplot()
    n = [r["training_draws"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(n, [r["rmse_usd_per_bbl"] for r in rows], "o-", label="surrogate RMSE vs test labels")
    ax.loglog(n, [r["max_abs_error_usd_per_bbl"] for r in rows], "s--", label="surrogate max error")
    ax.axhline(rows[-1]["label_mc_se_mean_usd_per_bbl"], color="grey", ls=":",
               label="mean MC SE of labels")
    if benchmark:
        ax.axhline(
            benchmark["surrogate_vs_benchmark"]["rmse_noise_corrected_usd_per_bbl"],
            color="C3", ls="-.",
            label=f"final surrogate vs independent MC, noise-corrected ({benchmark['points']} pts)",
        )
    ax.set_xlabel("training draws")
    ax.set_ylabel("USD/barrel")
    ax.set_title("Surrogate approximation error (not Heston model error)")
    ax.legend(fontsize=7)
    _save(fig, path)


def plot_jacobian(arrays, path):
    plt = _pyplot()
    params, smallest = arrays["parameters"], arrays["singular_values"][:, -1]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].scatter(params[:, 3], smallest, s=4, c=params[:, 4], cmap="coolwarm")
    axes[0].set(xscale="log", yscale="log", xlabel="sigma (vol of variance)",
                ylabel="smallest singular value", title="Smallest singular value vs sigma")
    axes[1].scatter(params[:, 3], arrays["parameter_sensitivity"][:, 4], s=4, alpha=0.5)
    axes[1].set(xscale="log", yscale="log", xlabel="sigma",
                ylabel="||d prices / d rho||", title="Rho sensitivity vs sigma")
    axes[2].scatter(params[:, 0], smallest, s=4, alpha=0.5)
    axes[2].set(xscale="log", yscale="log", xlabel="v0", ylabel="smallest singular value",
                title="Smallest singular value vs v0")
    axes[3].bar(PARAMETER_NAMES, arrays["weakest_direction"].mean(axis=0))
    axes[3].set(ylabel="mean |loading|", title="Which parameters form the flattest direction")
    fig.suptitle("SURROGATE Jacobian w.r.t. scaled parameters (colour in panel 1 = rho)")
    _save(fig, path)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    fig.clf()


def _config_from_data(data):
    return MonteCarloConfig(**ast.literal_eval(str(data["monte_carlo"])))


@lru_cache(maxsize=8)
def _cached_load(path, modified):
    return Surrogate.load(path)


def surface(params, model_path):
    """Tim's drop-in for Monte Carlo: Heston parameters -> N AO prices (USD/barrel),
    in the order of Surrogate.load(model_path).coordinate_ids. Loads once, then caches."""
    path = Path(model_path).resolve()
    return _cached_load(path, path.stat().st_mtime_ns)(params)
