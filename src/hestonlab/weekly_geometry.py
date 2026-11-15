"""Directly sampled parameter patches and independent numerical checks."""

import numpy as np
from .cpp_pricer import Estimate
from .direct_fit import Budget, decode, price_jacobian


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def independent_estimate(pricer, unit, config, seed, deadline, half_step=False):
    budget = Budget(deadline, config.validation_batches)
    estimates = []
    for batch in range(config.validation_batches):
        budget.consume()
        estimates.append(
            pricer.price(
                decode(unit),
                paths=config.validation_paths,
                seed=seed + batch,
                max_step=config.max_step_days / 365 / (2 if half_step else 1),
                covariance=True,
            )
        )
    n = len(estimates)
    covariance = sum(x.covariance for x in estimates) / n**2
    return Estimate(
        np.mean([x.prices for x in estimates], axis=0),
        np.sqrt(np.maximum(np.diag(covariance), 0)),
        covariance,
    )


def gap_statistics(target, estimate, scale):
    residual = (np.asarray(target) - estimate.prices) * scale
    gap = rms(residual)
    # Delta method uses cross-contract covariance, not independent-option errors.
    gradient = -residual * scale / (len(scale) * gap) if gap > 1e-12 else np.zeros(len(scale))
    se = float(np.sqrt(max(gradient @ estimate.covariance @ gradient, 0)))
    noise = rms(estimate.standard_errors * scale)
    return {
        "gap_usd": gap,
        "gap_delta_se_usd": se,
        "price_noise_rms_usd": noise,
        "noise_resolved": bool(gap > 2 * noise),
    }


def patches(pricer, unit, market, market_unit, scale, config, deadline, previous_axes=None):
    n = len(scale)
    metric = scale / np.sqrt(n)
    budget = Budget(deadline, config.surface_grid**2 + 60)
    jac = price_jacobian(
        pricer,
        unit,
        paths=config.surface_paths,
        seed=config.seed,
        max_step=config.max_step_days / 365,
        budget=budget,
    )
    free = (unit > 0.002) & (unit < 0.998)
    flags = []
    # A parameter at its bound is not a two-sided tangent direction.
    if free.sum() < 2:
        return {}, ["insufficient_free_heston_directions"]
    left, singular, right = np.linalg.svd(jac[:, free] * metric[:, None], full_matrices=False)
    if singular[1] <= max(singular[0] * 1e-5, 1e-8):
        return {}, ["unresolved_heston_slice_directions"]
    axes = left[:, :2]
    directions = np.zeros((5, 2))
    # Equalize first-order price displacement along the two displayed axes.
    # Cap both parameter moves by the weaker direction, avoiding a long thin sheet
    # or huge excursions along an almost-unidentifiable direction.
    directions[free] = right[:2].T * (singular[1] / singular[:2])
    if previous_axes is not None:
        a, _, b = np.linalg.svd(axes.T @ previous_axes)
        rotation = a @ b
        axes = axes @ rotation
        directions = directions @ rotation
    row_movement = np.sum(np.abs(directions), axis=1)
    active = row_movement > 1e-12
    room = np.minimum(unit, 1 - unit)
    radius = min(config.slice_radius, float(np.min(room[active] / row_movement[active])) * 0.95)
    axis = np.linspace(-radius, radius, config.surface_grid)
    mesh = np.array(np.meshgrid(axis, axis)).reshape(2, -1).T
    candidates = unit + mesh @ directions.T
    valid = ((candidates >= 0) & (candidates <= 1)).all(axis=1)
    prices = np.full((len(mesh), n), np.nan)
    for k in np.where(valid)[0]:
        budget.consume()
        prices[k] = pricer.price(
            decode(candidates[k]),
            paths=config.surface_paths,
            seed=config.seed,
            max_step=config.max_step_days / 365,
        ).prices
    # Choose market directions whose first-order price moves match Heston's slice.
    tangent = jac @ directions * metric[:, None]
    market_jac = market.jacobian(market_unit) / np.sqrt(n)
    market_directions = np.linalg.lstsq(market_jac, tangent, rcond=1e-5)[0]
    # Do not stretch a weak market direction arbitrarily far outside observed support.
    largest = np.max(np.linalg.norm(mesh @ market_directions.T, axis=1))
    if largest > 1:
        market_directions /= largest
        flags.append("market_patch_reduced_to_supported_scale")
    market_units = market_unit + mesh @ market_directions.T
    market_prices = market.prices(market_units)
    support = market.supported(market_units)
    market_prices[~support] = np.nan
    shape = (config.surface_grid, config.surface_grid, n)
    return {
        "heston_patch": prices.reshape(shape),
        "market_patch": market_prices.reshape(shape),
        "heston_patch_units": candidates,
        "market_patch_units": market_units,
        "heston_jacobian": jac,
        "axes": axes,
        "directions": directions,
        "singular_values": singular,
        "slice_parameter_radius": np.array(radius),
        "surface_valid_fraction": np.array(valid.mean()),
        "market_valid_fraction": np.array(support.mean()),
    }, flags


def geometry_checks(pricer, unit, geometry, scale, config, deadline):
    if not geometry:
        return {"status": "no_surface"}, ["geometry_unavailable"]
    budget = Budget(deadline, 30)
    jac2 = price_jacobian(
        pricer,
        unit,
        paths=config.surface_paths,
        seed=config.seed,
        step=0.002,
        max_step=config.max_step_days / 365,
        budget=budget,
    )
    a, b = geometry["heston_jacobian"] * scale[:, None], jac2 * scale[:, None]
    # Judge the two drawn directions; a weak unused fifth direction should not dominate.
    directions = geometry["directions"]
    relative = float(
        np.linalg.norm((a - b) @ directions) / max(np.linalg.norm(b @ directions), 1e-12)
    )
    flags = ["unstable_heston_derivatives"] if relative > config.derivative_tolerance else []
    grid = config.surface_grid
    candidates = geometry["heston_patch_units"].reshape(grid, grid, 5)
    prices = geometry["heston_patch"]
    errors = []
    for i, j in [(grid // 2 - 1, grid // 2 - 1), (0, 0), (grid - 2, grid - 2)]:
        # A point on the chosen triangle's edge: compare its direct price to linear interpolation.
        # The renderer triangulates each cell across (i,j+1)--(i+1,j).
        ends = [candidates[i, j + 1], candidates[i + 1, j]]
        if np.isfinite(prices[i, j + 1]).all() and np.isfinite(prices[i + 1, j]).all():
            budget.consume()
            p = pricer.price(
                decode(np.mean(ends, axis=0)),
                paths=config.surface_paths,
                seed=config.seed,
                max_step=config.max_step_days / 365,
            ).prices
            errors.append(rms((p - (prices[i, j + 1] + prices[i + 1, j]) / 2) * scale))
    error = max(errors, default=float("inf"))
    if error > config.absolute_tolerance_usd:
        flags.append("surface_mesh_too_coarse")
    return {
        "derivative_step_relative_change": relative,
        "surface_midpoint_max_error_usd": error,
    }, flags
