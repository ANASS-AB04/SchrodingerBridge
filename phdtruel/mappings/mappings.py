import logging
from typing import Optional, Protocol, cast

import numpy as np
from scipy.interpolate import LinearNDInterpolator, griddata

logger = logging.getLogger(__name__)


class Mapping(Protocol):
    """A mapping is a function that takes points and return the mapped points.

    The scaling parameter s allows to interpolate between
    the identity mapping (s=0) and the full mapping (s=1).
    """

    def __call__(self, points: np.ndarray, s: Optional[float] = None) -> np.ndarray: ...


def identity_mapping(points: np.ndarray, s: Optional[float] = None) -> np.ndarray:
    return points


class DisplacementDefinedMapping:
    def __init__(
        self,
        support_points: np.ndarray,
        displacement: np.ndarray,
        name: str | None = None,
    ):
        if support_points.shape != displacement.shape:
            raise ValueError(
                "Support points and displacement must have the same shape."
            )

        self.name = "T" if name is None else name

        self._displacement_interpolator = LinearNDInterpolator(
            points=support_points,
            values=displacement,
            fill_value=0.0,
        )

    def __call__(self, points: np.ndarray, s: float = 1.0) -> np.ndarray:
        return points + s * self._displacement_interpolator(points)


class ComposedMapping:
    def __init__(self, *mappings: Mapping):
        self._mappings = mappings

    def __call__(self, points: np.ndarray, s: Optional[float] = None) -> np.ndarray:
        result = points
        for mapping in self._mappings:
            result = mapping(result, s)
        return result


def construct_inverse_mapping_on_mesh(
    mapping: Mapping,
    support_points: np.ndarray,
) -> Mapping:
    """Construct an approximate inverse mapping on a regular grid."""
    if support_points.ndim != 2:
        raise ValueError("Support points must be a 2D array of shape (N, D).")

    def inverse_mapping(points: np.ndarray, s: Optional[float] = None):
        s_value = 1.0 if s is None else s
        mapped_grid_points = mapping(support_points, s_value)

        # Use NaN as fill_value, then replace with original points for out-of-domain values
        result = griddata(
            points=mapped_grid_points,
            values=support_points,
            xi=points,
            method="linear",
            fill_value=np.nan,
        )

        # Replace NaN values (out-of-domain points) with the original points (no displacement)
        out_of_domain = np.isnan(result)
        result[out_of_domain] = points[out_of_domain]

        return result

    return cast(Mapping, inverse_mapping)


def check_inverse_mapping(points, T, T_inv):
    """Check that the inverse mapping is correct."""
    out = np.allclose(points, T_inv(T(points)))
    for s in np.linspace(0, 1, 10):
        out = out and np.allclose(points, T_inv(T(points, s), s))
    return out
