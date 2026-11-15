"""Non-neural local market manifold: PCA coordinates with a regularized quadratic graph."""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares


def features(z):
    z = np.atleast_2d(z)
    return np.column_stack(
        [
            np.ones(len(z)),
            z,
            *[z[:, i] * z[:, j] for i in range(z.shape[1]) for j in range(i, z.shape[1])],
        ]
    )


@dataclass
class MarketSurface:
    mean: np.ndarray
    basis: np.ndarray
    complement: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray
    low: np.ndarray
    high: np.ndarray
    radius: float
    metadata: dict

    @property
    def rank(self):
        return len(self.scales)

    def coordinates(self, values):
        return (np.asarray(values) - self.mean) @ self.basis.T / self.scales

    def prices(self, z):
        z = np.asarray(z)
        value = self.mean + (np.atleast_2d(z) * self.scales) @ self.basis
        value += features(z) @ self.coefficients @ self.complement
        return value[0] if z.ndim == 1 else value

    def supported(self, z):
        z = np.asarray(z)
        return (
            (z >= self.low - 1e-9).all(axis=-1)
            & (z <= self.high + 1e-9).all(axis=-1)
            & (np.linalg.norm(z, axis=-1) <= self.radius)
        )

    def jacobian(self, z, step=1e-4):
        eye = np.eye(self.rank) * step
        return ((self.prices(z + eye) - self.prices(z - eye)) / (2 * step)).T

    def project(self, target):
        z0 = self.coordinates(target)
        start = np.clip(z0, self.low + 1e-10, self.high - 1e-10)
        result = least_squares(
            lambda z: self.prices(z) - target,
            start,
            jac=self.jacobian,
            bounds=(self.low, self.high),
            max_nfev=100,
        )
        supported = bool(self.supported(result.x)) and not bool(
            np.any((result.x - self.low < 1e-5) | (self.high - result.x < 1e-5))
        )
        return {
            "unit": result.x,
            "point": self.prices(result.x),
            "distance": float(np.sqrt(np.mean(result.fun**2))),
            "supported": supported,
            "success": bool(result.success),
        }


def _fit(values, rank, ridge):
    mean = values.mean(axis=0)
    _, singular, vt = np.linalg.svd(values - mean, full_matrices=False)
    resolved = int((singular > max(singular[0] * 1e-8, 1e-10)).sum())
    rank = min(rank, resolved, values.shape[1] - 1)
    if rank < 2:
        raise ValueError("Market window has fewer than two resolved directions")
    basis, complement = vt[:rank], vt[rank:]
    scales = singular[:rank] / np.sqrt(len(values) - 1)
    z = (values - mean) @ basis.T / scales
    design = features(z)
    coefficients = np.zeros((design.shape[1], len(complement)))
    if ridge is not None:
        penalty = np.eye(design.shape[1]) * ridge * len(values)
        penalty[0, 0] = 0
        coefficients = np.linalg.solve(
            design.T @ design + penalty, design.T @ ((values - mean) @ complement.T)
        )
    return MarketSurface(
        mean,
        basis,
        complement,
        scales,
        coefficients,
        z.min(axis=0),
        z.max(axis=0),
        float(np.max(np.linalg.norm(z, axis=1))),
        {},
    )


def fit_market(history, dates=None, rank=5, validation=20):
    """Select curvature using a trailing chronological holdout, then refit on history.

    `history` must end strictly before the target. Values are in that target's USD
    metric. Validation is reconstruction via PCA coordinates, identical for all
    candidates; it does not use the current target or optimize a future manifold.
    """
    history = np.asarray(history, dtype=float)
    if history.ndim != 2 or not np.isfinite(history).all():
        raise ValueError("Finite history matrix required")
    if len(history) < max(2 * rank + 10, validation + 2 * rank + 5):
        raise ValueError("Insufficient trailing market history")
    train, test = history[:-validation], history[-validation:]
    candidates = []
    for ridge in (None, 0.001, 0.01, 0.1, 1.0):
        model = _fit(train, rank, ridge)
        residual = model.prices(model.coordinates(test)) - test
        candidates.append((float(np.sqrt(np.mean(residual**2))), ridge))
    # Require a 5% improvement over the plane to add curvature.
    best_error, best_ridge = min(candidates, key=lambda pair: pair[0])
    if best_error >= 0.95 * candidates[0][0]:
        best_error, best_ridge = candidates[0]
    model = _fit(history, rank, best_ridge)
    model.metadata = {
        "method": "pca_plane" if best_ridge is None else "quadratic_graph",
        "rank": model.rank,
        "ridge": best_ridge,
        "validation_days": validation,
        "validation_rmse_usd": best_error,
        "pca_validation_rmse_usd": candidates[0][0],
        "candidates": [{"ridge": r, "rmse_usd": e} for e, r in candidates],
        "history_days": len(history),
        "training_dates": list(dates or []),
    }
    return model


if __name__ == "__main__":
    raise SystemExit(
        "The neural training CLI was removed. Prepare with scripts/prepare_weekly.py; "
        "after approval launch scripts/run_weekly.py."
    )
