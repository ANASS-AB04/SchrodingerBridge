from collections.abc import Callable

import numpy as np
from scipy.stats import qmc

from phdtruel.feature_identification import DEFAULT_SEED
from phdtruel.fields.meshes import Mesh


def _bounds_with_margin(
    lower: np.ndarray,
    upper: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink axis-aligned bounds by a fractional margin."""
    span = upper - lower
    return lower + margin * span, upper - margin * span


def sample_boundary_points(mesh: Mesh, n: int) -> np.ndarray:
    """Sample *n* points uniformly along the ordered mesh boundary."""
    boundary = np.asarray(mesh.boundary_points)
    if boundary.shape[0] == 0:
        raise ValueError("Mesh has no boundary points to sample from.")
    if boundary.shape[0] <= n:
        return boundary.copy()
    indices = np.linspace(0, boundary.shape[0] - 1, n, dtype=int)
    return boundary[indices]


def sample_interior_points_lhs(
    mesh: Mesh,
    n: int,
    *,
    seed: int = DEFAULT_SEED,
    margin: float = 0.0,
) -> np.ndarray:
    """Sample *n* interior points via Latin Hypercube Sampling in mesh bounds."""
    bounds = mesh.get_bounds()
    lower = np.asarray(bounds.minima, dtype=float)
    upper = np.asarray(bounds.maxima, dtype=float)
    l_bounds, u_bounds = _bounds_with_margin(lower, upper, margin)

    sampler = qmc.LatinHypercube(d=lower.shape[0], seed=seed)
    unit_sample = sampler.random(n)
    return qmc.scale(unit_sample, l_bounds, u_bounds)


def sample_interior_points_lhs_masked(
    l_bounds: np.ndarray,
    u_bounds: np.ndarray,
    inside: Callable[[np.ndarray], np.ndarray],
    n: int,
    *,
    seed: int = DEFAULT_SEED,
    margin: float = 0.0,
    oversample_factor: float = 1.5,
    max_rounds: int = 100,
) -> np.ndarray:
    """Sample *n* interior points via LHS in a bbox, keeping masked points only."""
    lower = np.asarray(l_bounds, dtype=float)
    upper = np.asarray(u_bounds, dtype=float)
    l_bounds, u_bounds = _bounds_with_margin(lower, upper, margin)

    sampler = qmc.LatinHypercube(d=lower.shape[0], seed=seed)
    accepted: list[np.ndarray] = []

    for _round in range(max_rounds):
        if len(accepted) >= n:
            break
        remaining = n - len(accepted)
        batch_size = max(int(np.ceil(remaining * oversample_factor)), 1)
        unit_sample = sampler.random(batch_size)
        candidates = qmc.scale(unit_sample, l_bounds, u_bounds)
        mask = inside(candidates)
        if np.any(mask):
            accepted.extend(candidates[mask])
    else:
        raise RuntimeError(
            f"Could not sample {n} interior points after {max_rounds} LHS rounds."
        )

    return np.asarray(accepted[:n])
