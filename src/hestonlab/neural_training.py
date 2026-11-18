"""Full-price neural embeddings, training, projection and auditable checkpoints."""

from pathlib import Path
import os
import time
import numpy as np
import torch
from scipy.optimize import least_squares

from .neural_embedding import DTYPE, ManifoldNetwork, geometry_diagnostics
from .weekly_store import sha_file, write_json


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, temp)
    os.replace(temp, path)


class NeuralSurface:
    """An injective 5-to-21 network and invertible per-coordinate output scaling."""

    def __init__(self, network, mean, std, metadata=None, train_units=None):
        self.network = network.eval()
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)
        if self.mean.shape != (21,) or self.std.shape != (21,) or not (self.std > 0).all():
            raise ValueError("All 21 coordinates and positive output scales are required")
        self.metadata = metadata or {}
        self.train_units = None if train_units is None else np.asarray(train_units)
        self._mean = torch.as_tensor(self.mean, dtype=DTYPE)
        self._std = torch.as_tensor(self.std, dtype=DTYPE)
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)

    def prices_from_unit(self, unit):
        return self.network(unit) * self._std + self._mean

    def prices(self, units):
        with torch.no_grad():
            return self.prices_from_unit(torch.as_tensor(np.asarray(units), dtype=DTYPE)).numpy()

    def jacobian(self, unit):
        return (
            torch.func.jacrev(self.prices_from_unit)(torch.as_tensor(unit, dtype=DTYPE))
            .detach()
            .numpy()
        )

    def scaled(self, scale):
        return NeuralSurface(
            self.network, self.mean * scale, self.std * scale, self.metadata, self.train_units
        )

    def initial_unit(self, target):
        with torch.no_grad():
            standardized = (torch.as_tensor(target, dtype=DTYPE) - self._mean) / self._std
            units = self.network.ambient_inverse(standardized)[..., :5] + 0.5
        return np.clip(units.numpy(), 0.00001, 0.99999)

    def project(self, target, starts=()):
        target = np.asarray(target)
        choices = [self.initial_unit(target), *starts, np.full(5, 0.5)]
        best = None
        for start in choices:
            result = least_squares(
                lambda u: (self.prices(u) - target) / np.sqrt(21),
                np.clip(start, 0.00001, 0.99999),
                jac=lambda u: self.jacobian(u) / np.sqrt(21),
                bounds=(0.00001, 0.99999),
                max_nfev=180,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
            )
            loss = float(result.fun @ result.fun)
            if best is None or loss < best["loss"]:
                best = dict(
                    unit=result.x,
                    point=self.prices(result.x),
                    loss=loss,
                    distance=np.sqrt(loss),
                    success=bool(result.success),
                    at_bound=bool(np.any((result.x < 0.0001) | (result.x > 0.9999))),
                )
        return best

    def reconstruct_batch(self, values, steps=300):
        """Held-out reconstruction: optimize latent points with frozen weights."""
        targets = torch.as_tensor(np.asarray(values), dtype=DTYPE)
        raw = torch.nn.Parameter(
            torch.logit(torch.as_tensor(self.initial_unit(values), dtype=DTYPE))
        )
        optimizer = torch.optim.Adam([raw], lr=0.03)
        best_loss = torch.full((len(values),), float("inf"), dtype=DTYPE)
        best_points = torch.zeros_like(targets)
        for _ in range(steps):
            optimizer.zero_grad()
            points = self.prices_from_unit(torch.sigmoid(raw))
            losses = (points - targets).square().mean(1)
            with torch.no_grad():
                improved = losses < best_loss
                best_loss[improved] = losses[improved]
                best_points[improved] = points[improved]
            losses.mean().backward()
            optimizer.step()
        return best_points.numpy()

    def save(self, path):
        save_torch(
            path,
            dict(
                format_version=1,
                state=self.network.state_dict(),
                width=self.network.width,
                depth=self.network.depth,
                mean=self.mean.tolist(),
                std=self.std.tolist(),
                metadata=self.metadata,
                train_units=None if self.train_units is None else self.train_units.tolist(),
            ),
        )

    @classmethod
    def load(cls, path):
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if blob["format_version"] != 1:
            raise ValueError("Unknown neural checkpoint format")
        network = ManifoldNetwork(21, blob["width"], blob["depth"])
        network.load_state_dict(blob["state"])
        if not all(torch.isfinite(v).all() for v in network.state_dict().values()):
            raise ValueError("Nonfinite neural checkpoint")
        if torch.linalg.svdvals(network.affine)[-1] <= 0:
            raise ValueError("Singular neural ambient affine map")
        return cls(network, blob["mean"], blob["std"], blob["metadata"], blob["train_units"])


def train_heston(folder, config):
    folder = Path(folder)
    target = folder / "model.pt"
    sources = {name: sha_file(folder / (name + ".npz")) for name in ("train", "validation")}
    recipe = {
        name: sha_file(Path(__file__).with_name(name))
        for name in ("neural_training.py", "neural_embedding.py")
    }
    if target.exists():
        model = NeuralSurface.load(target)
        if model.metadata["source_arrays"] != sources:
            raise ValueError("Heston network and training cloud differ")
        if model.metadata["training_recipe"] != recipe:
            raise ValueError(
                "Neural training implementation changed; inspect before reusing this model"
            )
        return model
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    data = {}
    for name in sources:
        with np.load(folder / (name + ".npz"), allow_pickle=False) as z:
            data[name] = {key: z[key] for key in z.files}
    mean = data["train"]["prices"].mean(axis=0)
    std = np.maximum(data["train"]["prices"].std(axis=0), 1e-8)
    x = torch.as_tensor(data["train"]["units"], dtype=DTYPE)
    y = torch.as_tensor((data["train"]["prices"] - mean) / std, dtype=DTYPE)
    vx = torch.as_tensor(data["validation"]["units"], dtype=DTYPE)
    vy = torch.as_tensor((data["validation"]["prices"] - mean) / std, dtype=DTYPE)
    network = ManifoldNetwork(21, config.width, config.depth)
    network.initialize_from_data(x.numpy(), y.numpy())
    optimizer = torch.optim.Adam(network.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=60, min_lr=1e-5
    )
    generator = torch.Generator().manual_seed(config.seed)
    best, stale, start_epoch = float("inf"), 0, 0
    best_state = None
    checkpoint = folder / "training.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=True)
        if saved["source_arrays"] != sources:
            raise ValueError("Training resume data differ")
        if saved["training_recipe"] != recipe:
            raise ValueError("Training resume implementation differs")
        network.load_state_dict(saved["state"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        generator.set_state(saved["generator"])
        best, stale, start_epoch = saved["best"], saved["stale"], saved["epoch"] + 1
        best_state = saved["best_state"]
    started = time.monotonic()
    epoch = start_epoch - 1
    for epoch in range(start_epoch, config.heston_epochs):
        network.train()
        for batch in torch.randperm(len(x), generator=generator).split(config.batch_size):
            optimizer.zero_grad()
            loss = (network(x[batch]) - y[batch]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite Heston training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 10)
            optimizer.step()
        network.eval()
        with torch.no_grad():
            value = float((network(vx) - vy).square().mean())
        scheduler.step(value)
        if value < best:
            best, stale = value, 0
            best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
        else:
            stale += 1
        done = optimizer.param_groups[0]["lr"] <= 1.0001e-5 and stale > 120
        if epoch % 100 == 0 or done or epoch == config.heston_epochs - 1:
            save_torch(
                checkpoint,
                dict(
                    source_arrays=sources,
                    training_recipe=recipe,
                    epoch=epoch,
                    state=network.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    generator=generator.get_state(),
                    best=best,
                    stale=stale,
                    best_state=best_state,
                ),
            )
            print(
                f"Heston neural epoch {epoch + 1}: validation MSE {value:.6g}, "
                f"best {best:.6g}, {time.monotonic() - started:.0f}s",
                flush=True,
            )
        if done:
            break
    network.load_state_dict(best_state)
    model = NeuralSurface(
        network,
        mean,
        std,
        dict(
            source_arrays=sources,
            training_recipe=recipe,
            epochs=epoch + 1,
            best_validation_mse_standardized=best,
            training_points=len(x),
            validation_points=len(vx),
            native_price_dimension=21,
            latent_dimension=5,
            input="five Heston parameters",
            training_price_source="C++ Monte Carlo",
            validation_stream_independent=True,
            test_used_in_training=False,
            pca_output_compression=False,
        ),
    )
    model.save(target)
    return model


def fit_market(history, dates, config):
    """Train a neural auto-decoder. PCA only initializes latent coordinates."""
    y = np.asarray(history)
    if y.shape != (config.history, 21) or len(dates) != config.history:
        raise ValueError("Exactly the specified prior history and all 21 prices are required")
    validation = config.market_validation
    train = y[:-validation]

    def initialize(values):
        mean, std = values.mean(0), np.maximum(values.std(0), 1e-8)
        standardized = (values - mean) / std
        _, _, vt = np.linalg.svd(standardized - standardized.mean(0), full_matrices=False)
        scores = standardized @ vt[:5].T
        span = np.maximum(np.ptp(scores, axis=0), 1e-8)
        units = 0.25 + 0.5 * (scores - scores.min(0)) / span
        torch.manual_seed(config.seed)
        network = ManifoldNetwork(21, config.width, config.depth)
        network.initialize_from_data(units, standardized)
        raw = torch.nn.Parameter(torch.logit(torch.as_tensor(units, dtype=DTYPE)))
        return network, raw, mean, std, torch.as_tensor(standardized, dtype=DTYPE)

    def optimize(values, epochs, check=None):
        network, raw, mean, std, target = initialize(values)
        optimizer = torch.optim.Adam([*network.parameters(), raw], lr=0.002)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=120, min_lr=1e-5
        )
        best_loss, best_state, best_raw = float("inf"), None, None
        validation_curve = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = (network(torch.sigmoid(raw)) - target).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite market neural loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*network.parameters(), raw], 10)
            optimizer.step()
            scheduler.step(float(loss.detach()))
            if float(loss.detach()) < best_loss:
                best_loss = float(loss.detach())
                best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
                best_raw = raw.detach().clone()
            if check is not None and (epoch + 1) % 400 == 0:
                # Held-out dates optimize their latent locations only; no weight updates.
                model = NeuralSurface(network, mean, std)
                reconstructed = model.reconstruct_batch(check)
                error = float(np.sqrt(np.mean((reconstructed - check) ** 2)))
                validation_curve.append(
                    dict(epochs=epoch + 1, reconstruction_rmse_normalized=error)
                )
                for parameter in network.parameters():
                    parameter.requires_grad_(True)
        network.load_state_dict(best_state)
        return NeuralSurface(
            network, mean, std, train_units=torch.sigmoid(best_raw).numpy()
        ), validation_curve

    _, curve = optimize(train, config.market_epochs, y[-validation:])
    chosen = min(curve, key=lambda point: point["reconstruction_rmse_normalized"])
    model, _ = optimize(y, chosen["epochs"])
    model.metadata = dict(
        method="neural_injective_embedding",
        native_price_dimension=21,
        latent_dimension=5,
        pca_initialization_only=True,
        pca_output_compression=False,
        training_dates=dates,
        history_days=len(dates),
        validation_days=validation,
        selected_epochs=chosen["epochs"],
        validation_curve=curve,
        training_reconstruction_rmse_normalized=float(
            np.sqrt(np.mean((model.prices(model.train_units) - y) ** 2))
        ),
    )
    return model


def validate_heston(model, folder, scale, pricer, config):
    from .direct_fit import price_jacobian

    folder = Path(folder)
    with np.load(folder / "test.npz", allow_pickle=False) as z:
        units, truth, errors = z["units"], z["prices"], z["standard_errors"]
        global_count = int(z["global_count"])
    residual = (model.prices(units) - truth) * scale
    result = dict(
        native_price_dimension=21,
        test_points=len(units),
        independent_test_not_used_in_training=True,
        rms_usd=float(np.sqrt(np.mean(residual**2))),
        local_rms_usd=float(np.sqrt(np.mean(residual[global_count:] ** 2))),
        max_abs_usd=float(np.abs(residual).max()),
        mc_noise_rms_usd=float(np.sqrt(np.mean((errors * scale) ** 2))),
        scale_convention="median first-half forward/discount scale by contract",
        scale_usd=scale.tolist(),
    )
    result["geometry"] = geometry_diagnostics(model.scaled(scale), units[::8])
    derivative = []
    for index in (global_count, global_count + 32, global_count + 64, global_count + 96):
        j = model.jacobian(units[index]) * scale[:, None]
        a = (
            price_jacobian(
                pricer,
                units[index],
                paths=config.check_paths,
                seed=config.seed + 900000,
                step=0.004,
                max_step=config.max_step,
            )
            * scale[:, None]
        )
        b = (
            price_jacobian(
                pricer,
                units[index],
                paths=config.check_paths,
                seed=config.seed + 900001,
                step=0.004,
                max_step=config.max_step,
            )
            * scale[:, None]
        )
        denominator = max(float(np.linalg.norm((a + b) / 2)), 1e-12)
        derivative.append(
            dict(
                test_index=index,
                relative_error=float(np.linalg.norm(j - (a + b) / 2) / denominator),
                mc_repeat_difference=float(np.linalg.norm(a - b) / denominator),
            )
        )
    result["derivatives"] = derivative
    result["screen_passed"] = bool(
        result["geometry"]["numerically_resolved"]
        and result["local_rms_usd"] <= max(0.025, 2 * result["mc_noise_rms_usd"])
        and max(d["relative_error"] for d in derivative) < 0.35
    )
    write_json(folder / "validation.json", result)
    return result
