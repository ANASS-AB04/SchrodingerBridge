import logging
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from math import prod
from typing import Iterable, Self, cast, override

import h5py
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DomainBounds:
    minima: list[float] | np.ndarray
    maxima: list[float] | np.ndarray

    @classmethod
    def by_axis(cls, *minmax: tuple[float, float]) -> Self:
        minima = [arg[0] for arg in minmax]
        maxima = [arg[1] for arg in minmax]
        return cls(minima, maxima)

    @classmethod
    def from_mesh(cls, grid_mesh: "RegularGrid") -> Self:
        return cls.by_axis(
            (grid_mesh.x.min(), grid_mesh.x.max()),
            (grid_mesh.y.min(), grid_mesh.y.max()),
        )

    def __post_init__(self):
        if len(self.minima) != len(self.maxima):
            raise ValueError("DomainBounds must have same length")

        self.minima = np.array(self.minima)
        self.maxima = np.array(self.maxima)

    def get_axis_tuples(self):
        return tuple((self.minima[i], self.maxima[i]) for i in range(len(self.minima)))

    def shrink(self, length: float) -> "DomainBounds":
        """Raise minima by length and reduce maxima by length."""
        minima = [v + length for v in self.minima]
        maxima = [v - length for v in self.maxima]
        return DomainBounds(minima, maxima)

    @property
    def lengths(self) -> np.ndarray:
        return np.asarray(self.maxima) - np.asarray(self.minima)

    def __repr__(self) -> str:
        return f"DomainBounds(axes={self.get_axis_tuples()})"

    def modify_axes(
        self, modifications: list[tuple[float | None, float | None]]
    ) -> "DomainBounds":
        """Return a new DomainBounds with specified axes modified.

        Each modification is a tuple of (new_min, new_max) for the corresponding axis.
        When new_min or new_max is None, the original value for that bound is retained.

        Returns:
            A new DomainBounds instance with the specified axes modified.

        Raises:
            ValueError: If any axis_index is out of range or if new_min >= new_max for any modification.
        """
        if len(modifications) != len(self.minima):
            raise ValueError(
                f"Number of modifications must match number of axes. "
                f"Expected {len(self.minima)}, got {len(modifications)}."
            )

        new_minima = []
        new_maxima = []

        for i, (new_min, new_max) in enumerate(modifications):
            original_min = self.minima[i]
            original_max = self.maxima[i]

            final_min = new_min if new_min is not None else original_min
            final_max = new_max if new_max is not None else original_max

            if final_min >= final_max:
                raise ValueError(
                    f"Invalid modification for axis {i}: new_min ({final_min}) "
                    f"must be less than new_max ({final_max})."
                )

            new_minima.append(final_min)
            new_maxima.append(final_max)

        return DomainBounds(new_minima, new_maxima)


class Mesh(ABC, metaclass=ABCMeta):
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @property
    @abstractmethod
    def points(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def boundary_points(self) -> np.ndarray:
        """Get the boundary points of the mesh.

        Returns:
            np.ndarray: Array of boundary points with shape (n_boundary_points, dimension).
        """
        pass

    @abstractmethod
    def truncate(self, bounds: DomainBounds) -> Self:
        pass

    @abstractmethod
    def save_in_hdf5(self, g: h5py.Group | h5py.File):
        pass

    def _indices_for_points_truncation(self, bounds: DomainBounds) -> np.ndarray:
        points = self.points

        if self.dimension != len(bounds.minima):
            raise ValueError(
                f"Incompatible mesh and bounds dimensions {self.dimension}!={len(bounds.minima)}"
            )

        to_keep = np.full(points.shape[0], True)
        for k in range(self.dimension):
            to_keep_col_k = (bounds.minima[k] <= points[:, k]) & (
                points[:, k] <= bounds.maxima[k]
            )
            to_keep = np.logical_and(to_keep, to_keep_col_k)
        to_keep = to_keep.ravel()
        return to_keep

    def truncate_points(self, bounds: DomainBounds) -> np.ndarray:
        to_keep = self._indices_for_points_truncation(bounds)
        return self.points[to_keep, :]

    def truncate_values(self, bounds: DomainBounds, values: np.ndarray) -> np.ndarray:
        if self.dimension != len(bounds.minima):
            raise ValueError(
                f"Incompatible mesh and bounds dimensions {self.dimension}!={len(bounds.minima)}"
            )
        to_keep = self._indices_for_points_truncation(bounds)
        return values[to_keep]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mesh):
            return False
        else:
            return (
                np.allclose(self.points, other.points)
                and self.dimension == other.dimension
            )

    def __hash__(self) -> int:
        return hash(tuple(self.points.flatten()))

    def __len__(self):
        return self.points.shape[0]

    def get_bounds(self) -> DomainBounds:
        minimas = np.min(self.points, axis=0)
        maximas = np.max(self.points, axis=0)
        return DomainBounds(minimas, maximas)


def extract_axes(points, shape):
    """
    Axes from an n-dimensional points array.

    Parameters:
    points -- A points array of shape (num_points, num_dims)
    shape -- The target multidimensional shape for reshaping

    Returns:
    A list containing axes values for each dimension.
    """
    num_dims = points.shape[1]  # Number of axes (dimensions in the data)
    axes = []

    for dim in range(num_dims):
        # Reshape the points corresponding to the dim-th coordinate
        reshaped_dim = points[:, dim].reshape(shape)

        # Extract the axis by slicing: keep full elements only along the current dimension (dim)
        slices = tuple(0 if d != dim else slice(None) for d in range(len(shape)))
        axes.append(reshaped_dim[slices])

    return axes


class RegularGrid(Mesh):
    def save_in_hdf5(self, g: h5py.Group | h5py.File):
        for i, ax in enumerate(self.axes):
            g.create_dataset(f"axis_{i}", data=ax)
        g.attrs["type"] = "RegularGrid"

    @classmethod
    def from_points(cls, points: np.ndarray, shape: tuple[int, ...]):
        return cls(extract_axes(points, shape))

    @classmethod
    def from_bounds(cls, bounds: DomainBounds, n_points: int) -> Self:
        axes = [
            np.linspace(_min, _max, n_points)
            for _min, _max in zip(bounds.minima, bounds.maxima)
        ]

        return cls(axes)

    def __init__(
        self,
        axes: Iterable[np.ndarray],
    ):
        self._axes = tuple(axes)
        self._set_axes_attrs()

    def _set_axes_attrs(self):
        if len(self._axes) == 1:
            self.x = self._axes[0]
        if len(self._axes) == 2:
            self.x = self._axes[0]
            self.y = self._axes[1]
        if len(self._axes) == 3:
            self.x = self._axes[0]
            self.y = self._axes[1]
            self.z = self._axes[2]

    def __repr__(self) -> str:
        return f"RegularGrid(axes=[{', '.join(f'array(shape={ax.shape})' for ax in self.axes)}, bounds={self.get_bounds()}])"

    @property
    @override
    def dimension(self):
        return len(self.axes)

    @property
    def shape(self):
        return tuple(len(ax) for ax in self.axes)

    @property
    def shape_order_F(self):
        shape = self.shape
        indexes = [1, 0, *range(2, len(shape))]
        shape_f = (shape[1], shape[0], *shape[2:])
        return shape_f, indexes

    @property
    @override
    def points(self) -> np.ndarray:
        xn = np.meshgrid(*self.axes, indexing="ij")
        return np.concatenate(
            [x.reshape(-1, 1) for x in xn],
            axis=1,
        )

    @property
    def axes(self) -> tuple[np.ndarray, ...]:
        return self._axes

    def __len__(self):
        return prod(self.shape)

    def truncate(self, bounds: DomainBounds) -> "RegularGrid":
        new_axes = [
            ax[(bounds.minima[k] <= ax) & (ax <= bounds.maxima[k])]
            for k, ax in enumerate(self.axes)
        ]
        return RegularGrid(new_axes)

    @property
    @override
    def boundary_points(self) -> np.ndarray:
        """Get ordered boundary points of the regular grid mesh.

        For 1D grids, returns the two extreme points.
        For 2D grids, returns ordered boundary points (traversing bottom → right → top → left).
        The boundary is NOT closed (first point is not repeated at the end).
        For 3D and higher dimensions, raises NotImplementedError.

        Returns:
            np.ndarray: Array of boundary points with shape (n_boundary_points, dimension).
                For 2D, the boundary forms an open path (not closed).

        Raises:
            NotImplementedError: If dimension > 2 (only 1D and 2D are supported).
        """
        return self.get_border_points()

    def get_border_points(self) -> np.ndarray:
        """
        Get ordered border points of the regular grid mesh.

        For 1D: Returns the two extreme points [min, max].
        For 2D: Returns ordered boundary points (NOT closed) by traversing:
            - Bottom edge (left to right, excluding last corner)
            - Right edge (bottom to top, excluding last corner)
            - Top edge (right to left, excluding last corner)
            - Left edge (top to bottom, excluding last corner)
        The boundary is NOT closed (first point is not repeated at the end).

        Returns:
            np.ndarray: Array of border points with shape (n_boundary_points, dimension).

        Raises:
            NotImplementedError: If dimension > 2 (visualization functions only support 2D).
        """
        if self.dimension == 0:
            return np.array([])

        elif self.dimension == 1:
            # Return the two extreme points
            return np.array([[self.axes[0][0]], [self.axes[0][-1]]])

        elif self.dimension == 2:
            # Construct ordered boundary for 2D grids
            x_axis = self.axes[0]
            y_axis = self.axes[1]

            # Handle edge case: single point grid
            if len(x_axis) == 1 and len(y_axis) == 1:
                point = np.array([[x_axis[0], y_axis[0]]])
                return point  # Return single point, not closed

            # Bottom edge: (x, y_min) from x_min to x_max, excluding last point
            bottom = np.column_stack([x_axis[:-1], np.full(len(x_axis) - 1, y_axis[0])])

            # Right edge: (x_max, y) from y_min to y_max, excluding last point
            right = np.column_stack([np.full(len(y_axis) - 1, x_axis[-1]), y_axis[:-1]])

            # Top edge: (x, y_max) from x_max to x_min, excluding last point
            top = np.column_stack(
                [x_axis[-1:0:-1], np.full(len(x_axis) - 1, y_axis[-1])]
            )

            # Left edge: (x_min, y) from y_max to y_min (excluding last point which is bottom-left corner)
            left = np.column_stack(
                [np.full(len(y_axis) - 1, x_axis[0]), y_axis[-1:0:-1]]
            )

            # Concatenate all edges
            boundary = np.vstack([bottom, right, top, left])

            return boundary

        # 3D and higher dimensions are not supported
        raise NotImplementedError(
            f"Ordered boundary points are only supported for 1D and 2D grids. "
            f"Got {self.dimension}D grid. Visualization functions only work with 2D fields."
        )


def split_regular_grid(
    grid: RegularGrid,
    subgrid_bounds: list[DomainBounds],
) -> list[RegularGrid]:
    """Split a RegularGrid into multiple subgrids based on provided bounds.

    It ensures that the subgrids together cover the entire original grid
    with only shared boundaries (no gaps or overlaps).
    Each subgrid is created from the DomainBounds, and the function extracts
    the closest corresponding axes segments to create new RegularGrid instances for each subgrid.
    """
    tolerance = 1e-10

    if grid.dimension != 2:
        raise NotImplementedError(
            f"split_regular_grid only supports 2D grids, got {grid.dimension}D"
        )

    if len(subgrid_bounds) == 0:
        raise ValueError("At least one subgrid bound must be provided")

    x_axis, y_axis = grid.axes
    subgrids: list[RegularGrid] = []

    for i, bounds in enumerate(subgrid_bounds):
        if len(bounds.minima) != 2:
            raise ValueError(
                f"Subgrid bounds at index {i} must be 2D, got {len(bounds.minima)}D"
            )

        x_selected = x_axis[
            (bounds.minima[0] - tolerance <= x_axis)
            & (x_axis <= bounds.maxima[0] + tolerance)
        ]
        y_selected = y_axis[
            (bounds.minima[1] - tolerance <= y_axis)
            & (y_axis <= bounds.maxima[1] + tolerance)
        ]

        if len(x_selected) == 0 or len(y_selected) == 0:
            raise ValueError(
                f"Subgrid bounds at index {i} select an empty axis: "
                f"x_points={len(x_selected)}, y_points={len(y_selected)}"
            )

        subgrids.append(RegularGrid([x_selected, y_selected]))

    decimals = int(-np.log10(tolerance)) if tolerance > 0 else 10
    parent_keys = {tuple(row) for row in np.round(grid.points, decimals=decimals)}

    point_to_owner_indices: dict[tuple[float, float], list[int]] = {}
    for grid_idx, subgrid in enumerate(subgrids):
        for row in np.round(subgrid.points, decimals=decimals):
            key = cast(tuple[float, float], tuple(row))
            point_to_owner_indices.setdefault(key, []).append(grid_idx)

    union_keys = set(point_to_owner_indices.keys())
    missing_points = parent_keys - union_keys
    extra_points = union_keys - parent_keys
    if missing_points or extra_points:
        raise ValueError(
            "Subgrid bounds must cover exactly the parent grid (no gaps/out-of-domain points). "
            f"missing_points={len(missing_points)}, extra_points={len(extra_points)}"
        )

    for point_key, owners in point_to_owner_indices.items():
        if len(owners) <= 1:
            continue

        point = np.asarray(point_key)
        for owner_idx in owners:
            owner_bounds = subgrids[owner_idx].get_bounds()
            on_owner_boundary = (
                np.isclose(point[0], owner_bounds.minima[0], atol=tolerance)
                or np.isclose(point[0], owner_bounds.maxima[0], atol=tolerance)
                or np.isclose(point[1], owner_bounds.minima[1], atol=tolerance)
                or np.isclose(point[1], owner_bounds.maxima[1], atol=tolerance)
            )
            if not on_owner_boundary:
                raise ValueError(
                    "Subgrid bounds overlap in interiors. Overlaps are only allowed on "
                    f"shared boundaries. Point {point.tolist()} is interior in subgrid "
                    f"{owner_idx}."
                )

    return subgrids


def merge_regular_grids(
    *grids: "RegularGrid",
    tolerance: float = 1e-10,
) -> tuple["PointCloud", list[tuple[tuple[int, ...], np.ndarray]]]:
    """Merge multiple RegularGrid meshes, separating exterior and interior boundaries.

    Concatenates points from all grids with deduplication. Identifies exterior boundaries
    (outer edges) and interior boundaries (shared edges between grids).

    Uses edge-counting: edges belonging to exactly one grid form the exterior boundary,
    edges shared by multiple grids form interior boundaries.

    Note:
        Currently only supports 2D grids. Domains with holes or disconnected components
        are not supported and will raise NotImplementedError.

    Args:
        grids: One or more RegularGrid instances. All must have the same dimension (2D).
        tolerance: Tolerance for point deduplication. Points within this distance
            are considered identical. Default is 1e-10.

    Returns:
        tuple containing:
            - pointcloud: PointCloud with deduplicated points and exterior boundary_indices
            - interior_boundaries: list of (mesh_indices, boundary_points) where
              mesh_indices is a tuple of grid indices that share this boundary

    Raises:
        ValueError: If no grids provided, grids have different dimensions, or
            all grids overlap completely (no exterior boundary).
        NotImplementedError: If dimension != 2, domain has holes, or domain is disconnected.
    """
    from collections import defaultdict

    # Validate inputs
    if len(grids) == 0:
        raise ValueError("At least one grid must be provided")

    dimension = grids[0].dimension
    for i, grid in enumerate(grids):
        if grid.dimension != dimension:
            raise ValueError(
                f"All grids must have same dimension. Grid 0 has {dimension}D, "
                f"grid {i} has {grid.dimension}D."
            )

    if dimension != 2:
        raise NotImplementedError(
            f"merge_regular_grids only supports 2D grids, got {dimension}D"
        )

    # Compute decimal places for rounding based on tolerance
    decimals = int(-np.log10(tolerance)) if tolerance > 0 else 10

    # Concatenate all points and deduplicate
    all_points = np.vstack([grid.points for grid in grids])
    # Round for robust deduplication
    rounded_points = np.round(all_points, decimals=decimals)
    unique_rounded, inverse_indices = np.unique(
        rounded_points, axis=0, return_inverse=True
    )
    unique_points = unique_rounded  # Use rounded points as canonical

    # Build a lookup from rounded point tuple to unique index
    def point_to_index(point: np.ndarray) -> int:
        """Map a point to its unique index via rounding."""
        rounded = tuple(np.round(point, decimals=decimals))
        # Find matching index in unique_points
        for idx, up in enumerate(unique_points):
            if tuple(up) == rounded:
                return idx
        raise ValueError(f"Point {point} not found in unique points")

    # Build edge ownership map: edge -> set of grid indices that own it
    # Edge is represented as (min_idx, max_idx) where idx1, idx2 are unique point indices
    edge_owners: dict[tuple[int, int], set[int]] = defaultdict(set)

    for grid_idx, grid in enumerate(grids):
        boundary = grid.get_border_points()  # ordered, NOT closed
        n_boundary = len(boundary)

        for i in range(n_boundary):
            p1 = boundary[i]
            p2 = boundary[(i + 1) % n_boundary]  # close the loop

            idx1 = point_to_index(p1)
            idx2 = point_to_index(p2)

            edge: tuple[int, int] = (min(idx1, idx2), max(idx1, idx2))
            edge_owners[edge].add(grid_idx)

    # Classify edges as exterior (1 owner) or interior (2+ owners)
    exterior_edges: list[tuple[int, int]] = []
    interior_edges_by_pair: dict[tuple[int, ...], list[tuple[int, int]]] = defaultdict(
        list
    )

    for edge, owners in edge_owners.items():
        if len(owners) == 1:
            exterior_edges.append(edge)
        else:
            # Shared by multiple grids -> interior boundary
            key = tuple(sorted(owners))
            interior_edges_by_pair[key].append(edge)

    # Check for complete overlap (no exterior boundary)
    if len(exterior_edges) == 0:
        raise ValueError(
            "All grids overlap completely - no exterior boundary exists. "
            "Ensure grids form a connected domain with an exterior boundary."
        )

    # Build adjacency graph from exterior edges
    adjacency: dict[int, list[int]] = defaultdict(list)
    for e0, e1 in exterior_edges:
        adjacency[e0].append(e1)
        adjacency[e1].append(e0)

    # Get all vertices in exterior boundary
    exterior_vertices = set()
    for e0, e1 in exterior_edges:
        exterior_vertices.add(e0)
        exterior_vertices.add(e1)

    # Walk the boundary to get ordered indices
    # Start from the first edge's first vertex
    start_vertex = exterior_edges[0][0]
    ordered_indices = [start_vertex]
    visited = {start_vertex}

    while True:
        current = ordered_indices[-1]
        neighbors = adjacency[current]
        next_vertex = None

        for n in neighbors:
            if n not in visited:
                next_vertex = n
                break

        if next_vertex is None:
            # Check if we can close the loop back to start
            if start_vertex in neighbors and len(ordered_indices) > 2:
                # Successfully closed the loop
                break
            else:
                # Cannot continue and cannot close - disconnected or stuck
                break

        ordered_indices.append(next_vertex)
        visited.add(next_vertex)

    # Check for disconnected domain
    if len(visited) != len(exterior_vertices):
        raise NotImplementedError(
            f"Disconnected domains are not supported. "
            f"Found {len(visited)} connected vertices but {len(exterior_vertices)} "
            f"total exterior vertices."
        )

    # Check for holes: if we visited all vertices but didn't use all edges
    # Each vertex should have exactly 2 edges in a simple closed boundary
    edges_used = len(
        ordered_indices
    )  # number of edges = number of vertices in closed loop
    if edges_used != len(exterior_edges):
        raise NotImplementedError(
            f"Domains with holes are not supported. "
            f"Expected {len(exterior_edges)} edges but boundary walk used {edges_used}."
        )

    # Build interior boundaries list
    interior_boundaries: list[tuple[tuple[int, ...], np.ndarray]] = []
    for mesh_indices, edges in interior_edges_by_pair.items():
        # Extract unique point indices from edges
        point_indices: set[int] = set()
        for e0, e1 in edges:
            point_indices.add(e0)
            point_indices.add(e1)

        # Get coordinates
        boundary_points = unique_points[list(point_indices)]
        interior_boundaries.append((mesh_indices, boundary_points))

    # Create PointCloud with exterior boundary
    boundary_indices = np.array(ordered_indices)
    pointcloud = PointCloud(
        unique_points,
        boundary_indices=boundary_indices,
        regular_grids_for_pcolormesh=list(grids),
    )

    return pointcloud, interior_boundaries


class PointCloud(Mesh):
    """A mesh defined by an unstructured collection of points.

    Args:
        points: Array of shape (n_points, dimension) containing the point coordinates.
        boundary_indices: Optional array of ordered indices defining the boundary points.
            If provided, the boundary_points property will return the points at these indices.
            If not provided, boundary_points will be computed using alpha shapes (2D only).
        sub
    """

    def __init__(
        self,
        points: np.ndarray,
        *,
        boundary_indices: np.ndarray | None = None,
        regular_grids_for_pcolormesh: list[RegularGrid] | None = None,
    ):
        self._points = points
        self._boundary_indices = boundary_indices
        self._regular_grids_for_pcolormesh = regular_grids_for_pcolormesh

    def __repr__(self) -> str:
        return f"PointCloud(points=array(shape={self._points.shape}), boundary_indices={self._boundary_indices})"

    @classmethod
    def from_boundary_points(
        cls,
        points: np.ndarray,
        boundary_points: np.ndarray,
        *,
        tolerance: float | None = None,
    ) -> Self:
        """Create a PointCloud with boundary indices computed from boundary points.

        Finds the closest mesh point for each provided boundary point and stores
        the corresponding indices.

        Args:
            points: Array of shape (n_points, dimension) containing all point coordinates.
            boundary_points: Array of shape (n_boundary, dimension) containing the
                boundary point coordinates. Must be a subset of (or very close to) points.
            tolerance: Maximum allowed distance between a boundary point and its closest
                mesh point. If None, defaults to 1e-6 times the mesh bounding box diagonal.
                Raises ValueError if any boundary point is farther than this tolerance.

        Returns:
            PointCloud with boundary_indices set to the indices of the closest mesh points.

        Raises:
            ValueError: If any boundary point is farther than tolerance from its closest
                mesh point in the mesh.
        """
        if points.shape[1] != boundary_points.shape[1]:
            raise ValueError(
                f"Points and boundary_points must have same dimension: "
                f"{points.shape[1]} != {boundary_points.shape[1]}"
            )

        # Compute default tolerance based on mesh bounding box diagonal
        if tolerance is None:
            bbox_min = np.min(points, axis=0)
            bbox_max = np.max(points, axis=0)
            diagonal = np.linalg.norm(bbox_max - bbox_min)
            tolerance = float(1e-6 * diagonal)
            if tolerance == 0:
                tolerance = 1e-10  # Fallback for single-point meshes

        # Find closest mesh point for each boundary point
        boundary_indices = []
        for i, bp in enumerate(boundary_points):
            distances = np.linalg.norm(points - bp, axis=1)
            closest_idx = np.argmin(distances)
            min_distance = distances[closest_idx]

            if min_distance > tolerance:
                raise ValueError(
                    f"Boundary point {i} at {bp} is too far from any mesh point. "
                    f"Closest mesh point is at distance {min_distance:.6e}, "
                    f"which exceeds tolerance {tolerance:.6e}."
                )

            boundary_indices.append(closest_idx)

        return cls(points, boundary_indices=np.array(boundary_indices))

    @classmethod
    def from_regular_grids(cls, *grids: RegularGrid, tolerance: float = 1e-10) -> Self:
        """Create a PointCloud from multiple RegularGrid meshes.

        Concatenates the points from all provided RegularGrid meshes into a single
        PointCloud with deduplication. Only exterior boundary points are kept;
        interior boundaries (shared edges between grids) are removed.

        Args:
            grids: One or more RegularGrid instances. All grids must have the same dimension.
            tolerance: Tolerance for point deduplication. Default is 1e-10.

        Returns:
            PointCloud containing all deduplicated points from the input grids with
            exterior boundary_indices set.

        Raises:
            ValueError: If no grids are provided or if grids have different dimensions.
            NotImplementedError: If domain has holes or is disconnected.
        """
        pointcloud, _ = merge_regular_grids(*grids, tolerance=tolerance)
        return pointcloud  # type: ignore[return-value]

    @property
    @override
    def dimension(self) -> int:
        return self._points.shape[1]

    @property
    @override
    def points(self) -> np.ndarray:
        return self._points

    @property
    def boundary_indices(self) -> np.ndarray | None:
        """Get the stored boundary indices, or None if not set."""
        return self._boundary_indices

    @property
    @override
    def boundary_points(self) -> np.ndarray:
        """Get the boundary points of the point cloud.

        If boundary_indices were provided at construction or via from_boundary_points,
        returns the points at those indices.

        Otherwise, computes the boundary using alpha shapes (2D only) and emits a warning.

        Returns:
            np.ndarray: Array of boundary points.

        Raises:
            ValueError: If dimension != 2 and no boundary_indices were provided.
        """
        if self._boundary_indices is not None:
            return self._points[self._boundary_indices]

        if self.dimension != 2:
            raise ValueError(
                f"Cannot compute boundary for {self.dimension}D PointCloud. "
                f"Provide boundary_indices at construction or use from_boundary_points()."
            )

        logger.warning(
            "Computing boundary using alpha shapes. For better performance and "
            "reproducibility, provide boundary_indices at construction."
        )
        return get_boundary_polygon(self._points)

    @override
    def truncate(self, bounds: DomainBounds) -> Self:
        """Truncate the point cloud to the given bounds.

        If boundary_indices are stored, they are remapped to the new index space.
        Boundary points that fall outside the bounds are removed from the indices.

        Args:
            bounds: The domain bounds to truncate to.

        Returns:
            A new PointCloud with only points within the bounds.
        """
        to_keep = self._indices_for_points_truncation(bounds)
        new_points = self._points[to_keep]

        # Remap boundary indices to new index space
        new_boundary_indices = None
        if self._boundary_indices is not None:
            # Create mapping from old indices to new indices
            old_to_new = np.full(len(self._points), -1, dtype=int)
            old_to_new[to_keep] = np.arange(np.sum(to_keep))

            # Find which boundary indices are still valid (point was kept)
            kept_boundary = [
                old_to_new[idx] for idx in self._boundary_indices if to_keep[idx]
            ]

            if kept_boundary:
                new_boundary_indices = np.array(kept_boundary)
            else:
                logger.warning(
                    "All boundary points were truncated. Boundary will be "
                    "recomputed via alpha shapes if accessed (2D only)."
                )

        return PointCloud(new_points, boundary_indices=new_boundary_indices)  # type: ignore[return-value]

    @override
    def save_in_hdf5(self, g: h5py.Group | h5py.File):
        g.create_dataset("points", data=self.points)
        g.attrs["type"] = "PointCloud"
        if self._boundary_indices is not None:
            g.create_dataset("boundary_indices", data=self._boundary_indices)

    @classmethod
    def from_hdf5(cls, g: h5py.Group | h5py.File) -> "PointCloud":
        points_dataset = cast(h5py.Dataset, g["points"])
        points = np.array(points_dataset[:])
        boundary_indices = None
        if "boundary_indices" in g:
            boundary_dataset = cast(h5py.Dataset, g["boundary_indices"])
            boundary_indices = np.array(boundary_dataset[:])
        return cls(points, boundary_indices=boundary_indices)


def get_boundary_polygon(points: np.ndarray, alpha: float | None = None) -> np.ndarray:
    """
    Compute the external boundary of a 2D point cloud using alpha shapes.

    Uses Delaunay triangulation with alpha shape filtering to find the concave
    boundary. Triangles with circumradius larger than 1/alpha are removed,
    then boundary edges (edges belonging to only one remaining triangle) are
    extracted and ordered to form a closed polygon.

    Args:
        points: Array of shape (n_points, 2) containing the 2D point coordinates.
        alpha: Alpha parameter controlling the level of detail. Larger values
            give more detailed (concave) boundaries. If None, automatically
            estimates alpha based on the average edge length in the triangulation.

    Returns:
        np.ndarray: Array of shape (n_boundary_points, 2) containing the ordered
            boundary points forming a closed polygon. The first point is repeated
            at the end to close the polygon.

    Raises:
        ValueError: If the points are not 2D.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"get_boundary_polygon only supports 2D points, got shape={points.shape}"
        )

    from scipy.spatial import Delaunay

    tri = Delaunay(points)

    # Compute circumradius for each triangle
    def circumradius(p1, p2, p3):
        """Compute circumradius of triangle with vertices p1, p2, p3."""
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)
        s = (a + b + c) / 2  # semi-perimeter
        area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
        if area < 1e-12:
            return np.inf
        return (a * b * c) / (4 * area)

    # Auto-estimate alpha if not provided
    if alpha is None:
        # Estimate based on average edge length
        edge_lengths = []
        for simplex in tri.simplices:
            for i in range(3):
                p1 = points[simplex[i]]
                p2 = points[simplex[(i + 1) % 3]]
                edge_lengths.append(np.linalg.norm(p2 - p1))
        avg_edge = np.mean(edge_lengths)
        # Alpha such that circumradius threshold is ~2x average edge length
        alpha = float(1.0 / (2.0 * avg_edge))

    # Filter triangles by circumradius (alpha shape)
    radius_threshold = 1.0 / alpha if alpha > 0 else np.inf
    valid_simplices = []
    for simplex in tri.simplices:
        p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        r = circumradius(p1, p2, p3)
        if r <= radius_threshold:
            valid_simplices.append(simplex)

    # Find boundary edges: edges that appear in only one valid triangle
    edge_count: dict[tuple[int, int], int] = {}
    for simplex in valid_simplices:
        for i in range(3):
            edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    # Boundary edges appear exactly once
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

    if not boundary_edges:
        return np.array([])

    # Order the boundary edges to form a polygon
    # Build adjacency list for boundary vertices
    adjacency: dict[int, list[int]] = {}
    for e0, e1 in boundary_edges:
        adjacency.setdefault(e0, []).append(e1)
        adjacency.setdefault(e1, []).append(e0)

    # Walk the boundary starting from the first edge
    ordered_indices = [boundary_edges[0][0]]
    visited = {ordered_indices[0]}

    while True:
        current = ordered_indices[-1]
        neighbors = adjacency[current]
        next_vertex = None
        for n in neighbors:
            if n not in visited:
                next_vertex = n
                break
        if next_vertex is None:
            break
        ordered_indices.append(next_vertex)
        visited.add(next_vertex)

    # Close the polygon by appending the first point
    ordered_indices.append(ordered_indices[0])

    return points[ordered_indices]


def load_mesh_from_hdf5(g: h5py.Group | h5py.File) -> Mesh:
    mesh_type = g.attrs["type"]
    if mesh_type == "RegularGrid":
        axes = [np.array(cast(h5py.Dataset, g[f"axis_{i}"])[:]) for i in range(len(g))]
        mesh = RegularGrid(axes)
    elif mesh_type == "PointCloud":
        mesh = PointCloud.from_hdf5(g)
    else:
        raise ValueError(f"Unknown mesh type {mesh_type}")

    return mesh
