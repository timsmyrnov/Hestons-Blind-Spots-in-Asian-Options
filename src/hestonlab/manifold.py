"""Embedded price manifolds: a five-plane followed by an ambient diffeomorphism.

The guarantee concerns the learned map, not the exact Heston pricing map. See
docs/manifold.md for the proof, parameter boundaries and numerical limitations.
"""

import numpy as np
import torch
from torch import nn

DTYPE = torch.float64


class AffineCoupling(nn.Module):
    """Smooth triangular bijection; bounded log scales keep exp(s) positive."""

    def __init__(self, dimension, width, parity):
        super().__init__()
        self.register_buffer("mask", (torch.arange(dimension) % 2 == parity).to(DTYPE))
        self.conditioner = nn.Sequential(
            nn.Linear(dimension, width, dtype=DTYPE), nn.Tanh(),
            nn.Linear(width, width, dtype=DTYPE), nn.Tanh(),
            nn.Linear(width, 2 * dimension, dtype=DTYPE),
        )
        # Identity initially; initialization of the final affine map supplies the
        # linear price fit. Zero last layer still allows learning on the first step.
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)

    def coefficients(self, fixed):
        log_scale, shift = self.conditioner(fixed).chunk(2, dim=-1)
        return 1.5 * torch.tanh(log_scale) * (1 - self.mask), shift * (1 - self.mask)

    def forward(self, x):
        fixed = x * self.mask
        scale, shift = self.coefficients(fixed)
        return fixed + (1 - self.mask) * (x * torch.exp(scale) + shift)

    def inverse(self, y):
        fixed = y * self.mask
        scale, shift = self.coefficients(fixed)
        return fixed + (1 - self.mask) * ((y - shift) * torch.exp(-scale))


class ManifoldNetwork(nn.Module):
    """u -> A G((u - 1/2, 0)) + b, an embedding for N >= latent_dim (5 for Heston).

    A is a fixed nonsingular matrix; G is a composition of affine couplings.
    All trainable functions are smooth. There is no price clipping or projection
    that could destroy injectivity. The last N - latent_dim coordinates are zero.
    """

    architecture = "manifold"

    def __init__(self, n_outputs, width=64, depth=4, latent_dim=5):
        super().__init__()
        if latent_dim < 1:
            raise ValueError("Positive latent dimension required")
        if n_outputs < latent_dim:
            raise ValueError(f"A {latent_dim}-dimensional manifold requires at least "
                             f"{latent_dim} prices")
        if width < 1 or depth < 1:
            raise ValueError("Positive width and depth required")
        self.n_outputs, self.width, self.depth = n_outputs, width, depth
        self.latent_dim = latent_dim
        self.layers = nn.ModuleList(
            AffineCoupling(n_outputs, width, i % 2) for i in range(depth)
        )
        self.register_buffer("affine", torch.eye(n_outputs, dtype=DTYPE))
        self.register_buffer("offset", torch.zeros(n_outputs, dtype=DTYPE))

    def initialize_from_data(self, unit, prices):
        """TRAINING-only linear initialization, extended to a nonsingular N x N map.

        Singular values are floored only at initialization. This is not a claim
        that Heston has five identifiable directions or that training will remain
        well conditioned. Couplings may subsequently compress any direction.
        """
        d = self.latent_dim
        x, y = np.asarray(unit) - 0.5, np.asarray(prices)
        coefficient = np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]
        left, singular, right = np.linalg.svd(coefficient[:d].T, full_matrices=True)
        scale = np.ones(self.n_outputs)
        scale[:d] = np.maximum(singular, 1e-3)
        rotation = np.eye(self.n_outputs)
        rotation[:d, :d] = right
        affine = (left * scale) @ rotation
        with torch.no_grad():
            self.affine.copy_(torch.as_tensor(affine, dtype=DTYPE))
            self.offset.copy_(torch.as_tensor(coefficient[d], dtype=DTYPE))

    def ambient_forward(self, latent):
        for layer in self.layers:
            latent = layer(latent)
        return latent @ self.affine.T + self.offset

    def ambient_inverse(self, prices):
        latent = torch.linalg.solve(self.affine, (prices - self.offset).unsqueeze(-1)).squeeze(-1)
        for layer in reversed(self.layers):
            latent = layer.inverse(latent)
        return latent

    def forward(self, unit):
        zeros = unit.new_zeros((*unit.shape[:-1], self.n_outputs - self.latent_dim))
        return self.ambient_forward(torch.cat((unit - 0.5, zeros), dim=-1))


def geometry_diagnostics(model, units):
    """Numerical checks supplement the structural proof, never replace it."""
    if not isinstance(model.network, ManifoldNetwork):
        return {"guarantee": "none; unconstrained MLP"}
    network = model.network
    state = tuple(network.state_dict().values())
    if not all(torch.isfinite(v).all().item() for v in state):
        raise ValueError("Nonfinite manifold state")
    if not torch.linalg.svdvals(network.affine)[-1].item() > 0:
        raise ValueError("Singular ambient affine map")
    d = network.latent_dim
    unit = torch.as_tensor(units, dtype=DTYPE)
    if (unit.ndim != 2 or unit.shape[1] != d or len(unit) == 0
            or not torch.isfinite(unit).all()):
        raise ValueError(f"Finite (n, {d}) diagnostic points required")
    with torch.no_grad():
        latent = network.ambient_inverse(network(unit))
    expected = torch.cat((unit - 0.5, unit.new_zeros((len(unit), network.n_outputs - d))), -1)
    jac = torch.func.vmap(torch.func.jacrev(model.prices_from_unit))(unit).detach()
    if not torch.isfinite(jac).all() or not torch.isfinite(latent).all():
        raise ValueError("Nonfinite manifold geometry; inspect numerical conditioning")
    singular = torch.linalg.svdvals(jac)
    tolerance = max(network.n_outputs, d) * torch.finfo(DTYPE).eps * singular[:, :1]
    numerical_rank = (singular > tolerance).sum(-1)
    roundtrip = float((latent - expected).abs().max())
    return {
        "guarantee": f"smooth {d}D embedding on the open parameter box, for finite weights "
                     "and nonsingular affine map in exact arithmetic",
        "closed_domain": "embedded manifold with corners",
        "scope": "learned price map only; not a proof about exact Heston prices",
        "points": len(unit),
        "inverse_max_abs_error": roundtrip,
        "minimum_numerical_rank": int(numerical_rank.min()),
        "minimum_singular_value_usd_per_bbl": float(singular[:, -1].min()),
        "maximum_condition_number": float((singular[:, 0] / singular[:, -1]).max()),
        "numerically_resolved": bool((numerical_rank == d).all() and roundtrip < 1e-7),
    }
