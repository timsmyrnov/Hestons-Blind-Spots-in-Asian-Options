"""Tangent-plane split of market moves, on a known flat Heston manifold."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hestonlab import tangent_moves as tm  # noqa: E402
from hestonlab.manifold import ManifoldNetwork  # noqa: E402
from hestonlab.surrogate import OutputScaler, Surrogate  # noqa: E402

N = 8


def plane_heston():
    return Surrogate(ManifoldNetwork(N, width=8, depth=2),
                     OutputScaler(np.full(N, 20.0), np.ones(N)), [f"c{i}" for i in range(N)])


def test_moves_inside_the_tangent_plane_are_followed_and_others_are_not():
    heston = plane_heston()
    basis = tm.tangent_basis(heston, np.full(5, 0.5), np.ones(N))
    assert basis.shape == (N, 5)
    np.testing.assert_allclose(basis.T @ basis, np.eye(5), atol=1e-10)
    inside = basis @ np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    tangent, normal = tm.split_move(basis, inside)
    np.testing.assert_allclose(tangent, inside, atol=1e-10)
    np.testing.assert_allclose(normal, 0, atol=1e-10)
    off = np.linalg.svd(basis, full_matrices=True)[0][:, 5]  # perpendicular to the plane
    tangent, normal = tm.split_move(basis, 0.3 * off + inside)
    np.testing.assert_allclose(tangent, inside, atol=1e-10)
    np.testing.assert_allclose(normal, 0.3 * off, atol=1e-10)


def test_weights_change_the_plane_into_dollars():
    heston = plane_heston()
    weight = np.linspace(1, 3, N)
    basis = tm.tangent_basis(heston, np.full(5, 0.5), weight)
    jac = tm._jacobian(heston, np.full(5, 0.5)) * weight[:, None]
    tangent, normal = tm.split_move(basis, jac @ np.ones(5))
    np.testing.assert_allclose(normal, 0, atol=1e-9)


def test_best_fixed_subspace_and_chance_references():
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((400, 5)) @ rng.standard_normal((5, 21))
    assert tm.best_fixed_share(signal) == pytest.approx(1.0)
    noise = rng.standard_normal((4000, 21))
    assert tm.best_fixed_share(noise) == pytest.approx(5 / 21, abs=0.05)


def test_market_reference_plane_uses_only_moves_finished_by_day_t():
    rng = np.random.default_rng(1)
    moves = rng.standard_normal((300, 21))
    moves[200:] = 0  # anything the reference may not see
    basis = tm.trailing_basis(moves, t=200, horizon=1, window=120, rank=4)
    again = tm.trailing_basis(np.vstack([moves[:200], 99 * np.ones((100, 21))]), 200, 1, 120, 4)
    np.testing.assert_allclose(np.abs(basis), np.abs(again), atol=1e-12)  # future ignored
    assert basis.shape == (21, 4)
    assert tm.trailing_basis(moves, t=30, horizon=1, window=120, rank=4) is None  # too few
