"""Fixed-footprint display charts built from saved prices; no Heston simulation.

Heston heights interpolate inside every saved patch. Market heights evaluate the
existing polynomial model on a continuous local section. Evaluation outside its
observed support is explicitly identified as model continuation, never hidden.
"""

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.spatial import ConvexHull
from .market_manifold import _fit


def common_reference(arrays):
    total = sum(a["axes"] @ a["axes"].T for a in arrays)
    _, vectors = np.linalg.eigh(total)
    reference = vectors[:, -2:][:, ::-1]
    u, _, vt = np.linalg.svd(reference.T @ arrays[0]["axes"])
    return reference @ u @ vt


def chart_frame(arrays, reference):
    n = len(arrays["scale"])
    gap = (arrays["market_point"] - arrays["heston_prices"] * arrays["scale"]) / np.sqrt(n)
    length = np.linalg.norm(gap)
    if length < 1e-12:
        # Coincident centers still need a finite display frame.
        candidate = np.eye(n)[:, np.argmin(np.sum(reference**2, axis=1))]
        up = candidate - reference @ (reference.T @ candidate)
        up /= np.linalg.norm(up)
    else:
        up = gap / length
    projected = reference - np.outer(up, up @ reference)
    u, singular, vt = np.linalg.svd(projected, full_matrices=False)
    if singular[-1] < 1e-7:
        raise ValueError("Reference price plane is degenerate against the gap direction")
    floor = u @ vt  # closest orthonormal frame; retains signs and order
    return np.column_stack((floor, up))


def heston_price_chart(arrays, frame):
    """Interpolate full RMS-scaled price changes, not only their visible height."""
    prices = arrays["heston_patch"]
    middle = prices.shape[0] // 2
    changes = (
        (prices - prices[middle, middle]) * arrays["scale"] / np.sqrt(len(arrays["scale"]))
    ).reshape(-1, len(arrays["scale"]))
    changes = changes[np.isfinite(changes).all(axis=1)]
    points = changes @ frame
    hull = ConvexHull(points[:, :2])
    normals, offsets = hull.equations[:, :2], hull.equations[:, 2]
    room = float(np.min(-offsets / np.sum(np.abs(normals), axis=1)))
    if room <= 0:
        raise ValueError("Saved Heston patch does not surround its central point")
    return CloughTocher2DInterpolator(points[:, :2], changes), room


def heston_chart(arrays, frame):
    prices, room = heston_price_chart(arrays, frame)
    return lambda xy: prices(xy) @ frame[:, 2], room


def local_comparisons(heston, market, heston_prices, market_prices):
    """Paired-point RMS differences, explicitly not nearest-manifold distances.

    Fixed mesh vertices keep each endpoint attached under temporal smoothing.
    Prices include all contracts in RMS coordinates, including hidden directions.
    """
    size = heston.shape[0]
    fractions = (
        (0.075, 0.075),
        (0.075, 0.925),
        (0.925, 0.075),
        (0.925, 0.925),
        (0.2, 0.2),
        (0.2, 0.5),
        (0.2, 0.8),
        (0.35, 0.35),
        (0.35, 0.65),
        (0.5, 0.2),
        (0.5, 0.8),
        (0.65, 0.35),
        (0.65, 0.65),
        (0.8, 0.2),
        (0.8, 0.5),
        (0.8, 0.8),
    )
    picks = sorted({(round(a * (size - 1)), round(b * (size - 1))) for a, b in fractions})
    return [
        {
            "start": market[i, j].copy(),
            "end": heston[i, j].copy(),
            "distance": float(np.linalg.norm(market_prices[i, j] - heston_prices[i, j])),
            "day": False,
            "visible": 1.0,
        }
        for i, j in picks
    ]


def market_jacobians(model, units):
    """Exact derivatives of the existing quadratic graph, (point, price, latent)."""
    base = (
        model.scales[:, None] * model.basis
        + model.coefficients[1 : 1 + model.rank] @ model.complement
    )
    jac = np.broadcast_to(base, (len(units), *base.shape)).copy()
    quadratic = model.coefficients[1 + model.rank :] @ model.complement
    k = 0
    for i in range(model.rank):
        for j in range(i, model.rank):
            jac[:, i, :] += units[:, j, None] * quadratic[k]
            jac[:, j, :] += units[:, i, None] * quadratic[k]
            k += 1
    return jac.transpose(0, 2, 1)


def market_chart(model, center, frame, targets, gap, tolerance=1e-7):
    """Follow a local model section outwards from the center to fixed floor points.

    Minimal latent changes define the section; continuation does not certify
    empirical support. Large excursions or failed solves reject the chart instead
    of making it disappear or inventing heights.
    """
    n = len(model.mean)
    origin = model.prices(center)
    units = np.broadcast_to(center, (len(targets), len(center))).copy()
    floor = frame[:, :2]
    for fraction in (0.25, 0.5, 0.75, 1.0):
        requested = targets * fraction
        for _ in range(35):
            values = (model.prices(units) - origin) / np.sqrt(n)
            residual = requested - values @ floor
            todo = np.linalg.norm(residual, axis=1) > tolerance
            if not todo.any():
                break
            jac = np.einsum("npr,pq->nqr", market_jacobians(model, units[todo]), floor) / np.sqrt(n)
            normal = jac @ jac.transpose(0, 2, 1)
            regularizer = np.maximum(np.trace(normal, axis1=1, axis2=2) * 1e-12, 1e-18)
            normal += regularizer[:, None, None] * np.eye(2)
            dual = np.linalg.solve(normal, residual[todo, ..., None])[..., 0]
            delta = np.einsum("nqr,nq->nr", jac, dual)
            length = np.linalg.norm(delta, axis=1, keepdims=True)
            delta *= np.minimum(1, 0.5 / np.maximum(length, 1e-12))
            units[todo] += delta
        else:
            raise ValueError("Market chart continuation did not converge")
        if np.max(np.linalg.norm(units - center, axis=1)) > max(4, 2 * model.radius):
            raise ValueError("Market chart requires excessive extrapolation")
    values = model.prices(units)
    miss = (values - origin) @ floor / np.sqrt(n) - targets
    if not np.isfinite(values).all() or np.max(np.linalg.norm(miss, axis=1)) > 2 * tolerance:
        raise ValueError("Market chart failed its fixed-floor constraint")
    heights = (values - origin) @ frame[:, 2] / np.sqrt(n) + gap
    return heights, model.supported(units), units


def fixed_snapshots(arrays, rows, panel, config, grid_size=41, window="fixed"):
    if window not in ("fixed", "local"):
        raise ValueError("Window must be fixed or local")
    reference = common_reference(arrays)
    records = []
    for data, row in zip(arrays, rows):
        frame = chart_frame(data, reference)
        interpolator, room = heston_price_chart(data, frame)
        index = row["panel_index"]
        history = panel["vectors"][index - config.history : index] * data["scale"]
        model = _fit(history, config.market_rank, row["market"]["ridge"])
        center = data["market_patch_units"][len(data["market_patch_units"]) // 2]
        # Rebuild only deterministic polynomial coefficients, not calibration.
        np.testing.assert_allclose(model.prices(center), data["market_point"], atol=1e-8, rtol=1e-9)
        stored = data["market_patch"].reshape(-1, len(model.mean))
        valid = np.isfinite(stored).all(axis=1)
        np.testing.assert_allclose(
            model.prices(data["market_patch_units"][valid]), stored[valid], atol=1e-8, rtol=1e-9
        )
        records.append((data, row, frame, interpolator, room, model, center))
    extent = 0.85 * min(r[4] for r in records)
    # One shared footprint for the entire movie. If a polynomial chart cannot span
    # it continuously, reduce the common footprint for ALL weeks, never one week.
    probe = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1])).reshape(2, -1).T
    if window == "fixed":
        for _ in range(12):
            try:
                for data, row, frame, interp, room, model, center in records:
                    market_chart(model, center, frame, probe * extent, row["gap_usd"])
                break
            except ValueError:
                extent *= 0.7
        else:
            raise ValueError("No common stable market footprint found")
    axis = np.linspace(-1, 1, grid_size)
    normalized_floor = np.array(np.meshgrid(axis, axis)).reshape(2, -1).T
    result, diagnostics = [], []
    for data, row, frame, interp, room, model, center in records:
        width = extent if window == "fixed" else 0.85 * room
        overlap_crop = False
        for _ in range(12):
            floor = normalized_floor * width
            try:
                m, support, units = market_chart(model, center, frame, floor, row["gap_usd"])
                h_changes = interp(floor) - interp([[0, 0]])
                if not np.isfinite(h_changes).all():
                    raise ValueError("Fixed Heston footprint would extrapolate saved prices")
                h = h_changes @ frame[:, 2]
                # A projection can overlap even when full price vectors differ.
                # Crop the local chart rather than distorting either surface.
                if window == "local" and np.min(m - h) < 0.02 * row["gap_usd"]:
                    overlap_crop = True
                    width *= 0.7
                    continue
                break
            except ValueError:
                if window == "fixed":
                    raise
                width *= 0.7
        else:
            raise ValueError(f"No continuous market section for {row['date']}")
        display_floor = normalized_floor if window == "local" else floor
        h_prices = h_changes + data["heston_prices"] * data["scale"] / np.sqrt(len(data["scale"]))
        m_prices = model.prices(units) / np.sqrt(len(data["scale"]))
        continuation = not support.all() or not row["market_projection_supported"]
        warnings = []
        if continuation:
            warnings.append("Market model continuation beyond observed support")
        if row["flags"]:
            warnings.append("Numerical checks flagged")
        shape = (grid_size, grid_size, 3)
        h_grid = np.column_stack((display_floor, h)).reshape(shape)
        m_grid = np.column_stack((display_floor, m)).reshape(shape)
        result.append(
            {
                "slice": 1,
                "date": row["date"],
                "heston": h_grid,
                "market": m_grid,
                "heston_alpha": np.ones((grid_size, grid_size)),
                "market_alpha": np.ones((grid_size, grid_size)),
                "paired_rms": np.linalg.norm(m_prices - h_prices, axis=1).reshape(
                    grid_size, grid_size
                ),
                "lines": [
                    {
                        "start": np.array([0, 0, row["gap_usd"]]),
                        "end": np.zeros(3),
                        "distance": row["gap_usd"],
                        "day": True,
                        "visible": 1.0,
                    }
                ]
                + local_comparisons(
                    h_grid,
                    m_grid,
                    h_prices.reshape(grid_size, grid_size, -1),
                    m_prices.reshape(grid_size, grid_size, -1),
                ),
                "warning": " · ".join(warnings),
            }
        )
        diagnostics.append(
            {
                "date": row["date"],
                "market_supported_fraction": float(support.mean()),
                "market_continuation_shown": continuation,
                "original_quality_flags": row["flags"],
                "market_latent_max_excursion": float(
                    np.max(np.linalg.norm(units - center, axis=1))
                ),
                "heston_saved_patch_half_width": room,
                "display_half_width_usd_per_bbl": width,
                "cropped_to_avoid_projected_overlap": overlap_crop,
                "market_model": "plane" if row["market"]["ridge"] is None else "quadratic",
            }
        )
    return result, {
        "method": "normalized_local_floor" if window == "local" else "fixed_physical_floor",
        "half_width_usd_per_bbl": extent if window == "fixed" else None,
        "horizontal_units": "section fraction" if window == "local" else "USD/bbl",
        "horizontal_dollar_distances_comparable_across_weeks": window == "fixed",
        "curvature_gain": 1,
        "local_lines": "RMS difference of paired full price vectors; not minimum distance",
        "local_lines_use_interpolated_mc_prices": True,
        "mesh_size": grid_size,
        "reference_axes": reference.tolist(),
        "heston_pricing_calls": 0,
        "heston_calibration_calls": 0,
        "heston_geometry": "C1 interpolation strictly inside cached price patches",
        "market_geometry": "same fitted polynomial; continuous fixed-floor section",
        "market_support_is_reported_not_hidden": True,
        "frame": "common reference axes projected perpendicular to each full-price gap",
        "weeks": diagnostics,
    }
