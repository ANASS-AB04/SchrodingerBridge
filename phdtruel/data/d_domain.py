"""Synthetic "D-shaped" (half-disk) domain and test fields.

This domain and its two-square translation test case were previously
duplicated verbatim across ``scripts/minimized_mappings/rbf_in_D_domain.py``
and ``scripts/minimized_mappings/rbf_free_cp_in_D_domain.py``.
"""

import logging
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from phdtruel.fields.field_of_interest import FieldOfInterest
from phdtruel.fields.meshes import PointCloud, RegularGrid
from phdtruel.mappings.cost_functional.types import MeshJaxed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DDomainGeometry:
    """Parameters defining a half-disk (D-shaped) domain."""

    x_left: float = 0.0
    cy: float = 0.0
    radius: float = 1.0


def _half_disk_mask(
    points: np.ndarray,
    geometry: DDomainGeometry,
) -> np.ndarray:
    """Return boolean mask for points inside the half-disk."""
    rel_x = points[:, 0] - geometry.x_left
    rel_y = points[:, 1] - geometry.cy
    return (points[:, 0] >= geometry.x_left) & (
        rel_x**2 + rel_y**2 <= geometry.radius**2 + 1e-12
    )


def _ordered_boundary_indices(
    points: np.ndarray,
    geometry: DDomainGeometry,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Build ordered boundary indices: left edge (bottom to top), then arc (top to bottom)."""
    tol_x = 0.51 * dx
    tol_arc = 0.51 * max(dx, dy)

    left_mask = np.abs(points[:, 0] - geometry.x_left) < tol_x
    left_idx = np.where(left_mask)[0]
    if left_idx.size == 0:
        raise ValueError("No left-edge points found on the D-domain mesh.")
    left_order = left_idx[np.argsort(points[left_idx, 1])]

    rel = points - np.array([geometry.x_left, geometry.cy])
    dist = np.linalg.norm(rel, axis=1)
    arc_mask = (np.abs(dist - geometry.radius) < tol_arc) & (
        points[:, 0] > geometry.x_left + tol_x
    )
    arc_idx = np.where(arc_mask)[0]
    if arc_idx.size == 0:
        raise ValueError("No arc points found on the D-domain mesh.")

    theta = np.arctan2(rel[arc_idx, 1], rel[arc_idx, 0])
    arc_order = arc_idx[np.argsort(-theta)]

    return np.concatenate([left_order, arc_order])


def create_d_shaped_domain(
    geometry: DDomainGeometry,
    n_mesh: int = 100,
) -> tuple[PointCloud, RegularGrid]:
    """Create a D-shaped PointCloud and its bounding-box RegularGrid.

    Args:
        geometry: Half-disk geometry (flat left, semicircular right).
        n_mesh: Number of points per axis on the background grid.

    Returns:
        Tuple of (d_mesh PointCloud, bounding_box RegularGrid).
    """
    x_axis = np.linspace(
        geometry.x_left,
        geometry.x_left + geometry.radius,
        n_mesh,
    )
    y_axis = np.linspace(
        geometry.cy - geometry.radius,
        geometry.cy + geometry.radius,
        n_mesh,
    )
    bbox_grid = RegularGrid([x_axis, y_axis])
    all_points = bbox_grid.points
    inside = _half_disk_mask(all_points, geometry)
    points = all_points[inside]

    if points.shape[0] == 0:
        raise ValueError("D-domain mask removed all points from the background grid.")

    dx = float(x_axis[1] - x_axis[0]) if len(x_axis) > 1 else geometry.radius
    dy = float(y_axis[1] - y_axis[0]) if len(y_axis) > 1 else geometry.radius
    boundary_indices = _ordered_boundary_indices(points, geometry, dx, dy)

    d_mesh = PointCloud(points, boundary_indices=boundary_indices)

    logger.info(
        "Created D-shaped domain: %d points (%d boundary), R=%.3f",
        points.shape[0],
        boundary_indices.shape[0],
        geometry.radius,
    )
    return d_mesh, bbox_grid


def square_values(
    points: np.ndarray,
    center: tuple[float, float],
    half_side: float,
    value: float,
) -> np.ndarray:
    """Axis-aligned square indicator on unstructured points."""
    cx, cy = center
    inside = (np.abs(points[:, 0] - cx) <= half_side) & (
        np.abs(points[:, 1] - cy) <= half_side
    )
    return np.where(inside, value, 0.0)


def _point_inside_half_disk(
    point: np.ndarray,
    geometry: DDomainGeometry,
) -> bool:
    rel_x = point[0] - geometry.x_left
    rel_y = point[1] - geometry.cy
    return bool(
        point[0] >= geometry.x_left
        and rel_x**2 + rel_y**2 <= geometry.radius**2 + 1e-12
    )


def _square_fits_in_half_disk(
    center: tuple[float, float],
    half_side: float,
    geometry: DDomainGeometry,
) -> bool:
    cx, cy = center
    corners = np.array(
        [
            [cx - half_side, cy - half_side],
            [cx + half_side, cy - half_side],
            [cx - half_side, cy + half_side],
            [cx + half_side, cy + half_side],
        ]
    )
    return all(_point_inside_half_disk(c, geometry) for c in corners)


def create_d_domain_fields(
    d_mesh: PointCloud,
    geometry: DDomainGeometry,
    *,
    square_value: float = 10.0,
    square_half_side: float | None = None,
    center_0: tuple[float, float] | None = None,
    center_1: tuple[float, float] | None = None,
) -> tuple[FieldOfInterest, FieldOfInterest, FieldOfInterest]:
    """Build u0, u1, u_ref with a square translating inside the D domain."""
    if square_half_side is None:
        square_half_side = 0.12 * geometry.radius

    if center_0 is None:
        center_0 = (geometry.x_left + 0.35 * geometry.radius, geometry.cy)
    if center_1 is None:
        center_1 = (geometry.x_left + 0.65 * geometry.radius, geometry.cy)

    center_ref = (
        0.5 * (center_0[0] + center_1[0]),
        0.5 * (center_0[1] + center_1[1]),
    )

    for name, center in (
        ("u0", center_0),
        ("u1", center_1),
        ("u_ref", center_ref),
    ):
        if not _square_fits_in_half_disk(center, square_half_side, geometry):
            raise ValueError(
                f"Square for {name} at {center} does not fit inside the D domain."
            )

    points = d_mesh.points
    values_0 = square_values(points, center_0, square_half_side, square_value)
    values_1 = square_values(points, center_1, square_half_side, square_value)
    values_ref = square_values(points, center_ref, square_half_side, square_value)

    field_0 = FieldOfInterest(
        parameter=None,
        mesh=d_mesh,
        values=values_0,
        name="$u_0$",
        fill_value=0.0,
        interpolated=False,
    )
    field_1 = FieldOfInterest(
        parameter=None,
        mesh=d_mesh,
        values=values_1,
        name="$u_1$",
        fill_value=0.0,
        interpolated=False,
    )
    field_ref = FieldOfInterest(
        parameter=None,
        mesh=d_mesh,
        values=values_ref,
        name="$u_{ref}$",
        fill_value=0.0,
        interpolated=False,
    )
    return field_0, field_1, field_ref


def setup_field_interpolation(
    *fields: FieldOfInterest,
) -> None:
    """Enable interpolation on fields (required for eval/CDI/GOT)."""
    for field in fields:
        field.setup_interpolation()


def pointcloud_to_jaxed(mesh: PointCloud) -> MeshJaxed:
    """Build MeshJaxed from a PointCloud (axes are unique coordinates)."""
    pts = mesh.points
    return MeshJaxed(
        points=jnp.array(pts),
        axes=(
            jnp.array(np.unique(pts[:, 0])),
            jnp.array(np.unique(pts[:, 1])),
        ),
    )
