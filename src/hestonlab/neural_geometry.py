"""Exact evaluations of neural embeddings on local display sections; no price PCA."""

import numpy as np
import torch

from .neural_embedding import DTYPE
from .stable_geometry import chart_frame, local_comparisons


def section_basis(model, center, floor):
    free = (center > 0.002) & (center < 0.998)
    if free.sum() < 2:
        raise ValueError("Fewer than two interior latent directions for a two-sided slice")
    matrix = floor.T @ model.jacobian(center)[:, free] / np.sqrt(21)
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular[-1] < 1e-8 * singular[0]:
        raise ValueError("Neural slice floor is degenerate")
    basis = np.zeros((5, 2))
    basis[free] = np.linalg.pinv(matrix)
    reach = np.sum(np.abs(basis), axis=1)
    moving = free & (reach > 1e-15)
    room = np.min(np.minimum(center[moving], 1 - center[moving]) / reach[moving])
    return basis, room


def section(model, center, frame, basis, targets):
    """Solve two floor constraints inside a fixed 2D latent section of the 5D map."""
    root = model.prices(center)
    floor = frame[:, :2]
    coordinates = targets.copy()
    tensor_center = torch.as_tensor(center, dtype=DTYPE)
    tensor_basis = torch.as_tensor(basis, dtype=DTYPE)
    tensor_floor = torch.as_tensor(floor / np.sqrt(21), dtype=DTYPE)

    def project(q):
        return model.prices_from_unit(tensor_center + tensor_basis @ q) @ tensor_floor

    jacobian = torch.func.vmap(torch.func.jacrev(project))
    for _ in range(16):
        units = center + coordinates @ basis.T
        if (units < 0).any() or (units > 1).any():
            raise ValueError("Neural section would leave its latent parameter domain")
        prices = model.prices(units)
        residual = (prices - root) @ floor / np.sqrt(21) - targets
        todo = np.linalg.norm(residual, axis=1) > 1e-8
        if not todo.any():
            return prices, units
        indices = np.flatnonzero(todo)
        for chunk in np.array_split(indices, max(1, int(np.ceil(len(indices) / 256)))):
            if not len(chunk):
                continue
            j = jacobian(torch.as_tensor(coordinates[chunk], dtype=DTYPE)).detach().numpy()
            delta = np.linalg.solve(j, residual[chunk, ..., None])[..., 0]
            coordinates[chunk] -= delta
    raise ValueError("Neural chart failed its floor constraints")


def snapshot(heston, market, arrays, row, reference, size=31):
    frame = chart_frame(arrays, reference)
    hcenter, mcenter = arrays["unit"], arrays["market_unit"]
    hb, hr = section_basis(heston, hcenter, frame[:, :2])
    mb, mr = section_basis(market, mcenter, frame[:, :2])
    width = 0.65 * min(hr, mr)
    axis = np.linspace(-1, 1, size)
    xy = np.array(np.meshgrid(axis, axis)).reshape(2, -1).T
    gap = row["gap_usd"]
    hroot = heston.prices(hcenter)
    cropped = False
    for attempt in range(18):
        try:
            hp, hu = section(heston, hcenter, frame, hb, xy * width)
            mp, mu = section(market, mcenter, frame, mb, xy * width)
            h = (hp - hroot) @ frame[:, 2] / np.sqrt(21)
            m = (mp - hroot) @ frame[:, 2] / np.sqrt(21)
            if (m - h).min() < 0.02 * gap:
                cropped = True
                raise ValueError("Displayed projections overlap")
            break
        except (ValueError, np.linalg.LinAlgError):
            width *= 0.6
    else:
        raise ValueError(f"No valid common neural section for {row['date']}")
    hgrid = np.column_stack((xy, h)).reshape(size, size, 3)
    mgrid = np.column_stack((xy, m)).reshape(size, size, 3)
    lines = [
        dict(start=np.array([0.0, 0.0, gap]), end=np.zeros(3), distance=gap, day=True, visible=1.0)
    ]
    lines += local_comparisons(
        hgrid,
        mgrid,
        (hp / np.sqrt(21)).reshape(size, size, 21),
        (mp / np.sqrt(21)).reshape(size, size, 21),
    )
    result = dict(
        slice=1,
        date=row["date"],
        heston=hgrid,
        market=mgrid,
        heston_alpha=np.ones((size, size)),
        market_alpha=np.ones((size, size)),
        paired_rms=np.sqrt(np.mean((hp - mp) ** 2, axis=1)).reshape(size, size),
        lines=lines,
        warning="",
    )
    details = dict(
        date=row["date"],
        half_width_usd_per_bbl=float(width),
        attempts=attempt + 1,
        cropped_to_avoid_projected_overlap=cropped,
        minimum_projected_separation=float((m - h).min()),
        native_price_dimension=21,
        latent_dimensions=5,
        price_evaluation="neural embeddings evaluated directly; no price interpolation",
        heston_unit_min=float(hu.min()),
        heston_unit_max=float(hu.max()),
        market_unit_min=float(mu.min()),
        market_unit_max=float(mu.max()),
    )
    return result, details
