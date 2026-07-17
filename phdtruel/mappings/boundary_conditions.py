"""Boundary condition utilities for piecewise FFD control point constraints.

This module provides functions to:
- Initialize control point masks with slip boundary conditions
- Identify internal vs external boundaries
- Relax constraints on shared boundaries
- Unify constraints at coincident control points
- Validate and visualize mask configurations

The mask arrays use NaN values to indicate free directions:
- mask[i, j, dim] = NaN: control point (i,j) is free in dimension dim
- mask[i, j, dim] = value: control point (i,j) has imposed displacement value in dim

Control point states emerge from the mask values:
- Fully fixed: both dimensions have non-NaN values (typically 0.0)
- Slip condition: one dimension NaN (free), other has value (fixed)
- Fully free: both dimensions are NaN
"""

import dataclasses
import logging
from typing import Any, Literal, Sequence

import numpy as np

from phdtruel.fields.meshes import DomainBounds, Mesh, RegularGrid

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclasses.dataclass(frozen=True)
class EdgeSegment:
    """Represents a segment of a region's boundary edge.

    Attributes:
        edge: Which edge this segment lies on ('left', 'right', 'top', 'bottom')
        range_start: Starting coordinate along the edge
        range_end: Ending coordinate along the edge
        is_partial: Whether segment spans less than the full edge
    """

    edge: Literal["left", "right", "top", "bottom"]
    range_start: float
    range_end: float
    is_partial: bool


# =============================================================================
# Tolerance Computation
# =============================================================================


def compute_spatial_tolerance(
    region_meshes: Sequence[Mesh],
    tolerance: float | None = None,
) -> float:
    """Compute single tolerance for all spatial comparisons.

    If tolerance is not provided, automatically computes it as 1% of the
    smallest region diagonal.

    Args:
        region_meshes: List of Mesh objects defining each region
        tolerance: Optional explicit tolerance value

    Returns:
        Tolerance value to use for spatial comparisons

    Example:
        >>> meshes = [mesh1, mesh2]
        >>> tol = compute_spatial_tolerance(meshes)
        >>> print(f"Using tolerance: {tol:.6f}")
    """
    if tolerance is not None:
        return tolerance

    all_bounds = [mesh.get_bounds() for mesh in region_meshes]
    diagonals = [np.linalg.norm(b.lengths) for b in all_bounds]
    computed_tolerance = float(0.01 * min(diagonals))

    logger.info(f"Computed spatial tolerance: {computed_tolerance:.6e}")
    return computed_tolerance


# =============================================================================
# Control Point Coordinate Utilities
# =============================================================================


def get_control_point_coordinates(
    bounds: DomainBounds,
    n_cp: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Get x and y coordinates of all control points in a region.

    Args:
        bounds: Domain bounds of the region
        n_cp: Number of control points (n_cp_x, n_cp_y)

    Returns:
        Tuple of (x_coords, y_coords) arrays of shape (n_cp_x,) and (n_cp_y,)

    Example:
        >>> bounds = region_mesh.get_bounds()
        >>> x_coords, y_coords = get_control_point_coordinates(bounds, (5, 5))
        >>> # Control point (i, j) is at position (x_coords[i], y_coords[j])
    """
    n_cp_x, n_cp_y = n_cp
    x_coords = np.linspace(bounds.minima[0], bounds.maxima[0], n_cp_x)
    y_coords = np.linspace(bounds.minima[1], bounds.maxima[1], n_cp_y)
    return x_coords, y_coords


# =============================================================================
# Edge Identification
# =============================================================================


def identify_shared_edge_segments(
    bounds: DomainBounds,
    sync_points: np.ndarray,
    tolerance: float,
) -> list[EdgeSegment]:
    """Identify which edge segments are shared with another region.

    Args:
        bounds: Domain bounds of the region
        sync_points: Array of shape (n, 2) containing shared boundary points
        tolerance: Distance tolerance for edge detection

    Returns:
        List of EdgeSegment objects describing shared edge segments

    Example:
        >>> bounds = region_mesh.get_bounds()
        >>> segments = identify_shared_edge_segments(bounds, sync_pts, 1e-6)
        >>> for seg in segments:
        ...     print(f"{seg.edge}: [{seg.range_start:.3f}, {seg.range_end:.3f}]")
    """
    if len(sync_points) == 0:
        return []

    xmin, xmax = bounds.minima[0], bounds.maxima[0]
    ymin, ymax = bounds.minima[1], bounds.maxima[1]

    edges = []
    n_points = len(sync_points)
    # Threshold: at least 50% of points must be on the edge
    threshold = max(1, n_points // 2)

    # Check each edge - count how many points lie on it
    left_mask = np.abs(sync_points[:, 0] - xmin) < tolerance
    right_mask = np.abs(sync_points[:, 0] - xmax) < tolerance
    bottom_mask = np.abs(sync_points[:, 1] - ymin) < tolerance
    top_mask = np.abs(sync_points[:, 1] - ymax) < tolerance

    left_count = np.sum(left_mask)
    right_count = np.sum(right_mask)
    bottom_count = np.sum(bottom_mask)
    top_count = np.sum(top_mask)

    # Create EdgeSegment for each detected edge
    if left_count >= threshold:
        y_values = sync_points[left_mask, 1]
        is_partial = (np.min(y_values) > ymin + tolerance) or (
            np.max(y_values) < ymax - tolerance
        )
        edges.append(
            EdgeSegment(
                edge="left",
                range_start=float(np.min(y_values)),
                range_end=float(np.max(y_values)),
                is_partial=is_partial,
            )
        )

    if right_count >= threshold:
        y_values = sync_points[right_mask, 1]
        is_partial = (np.min(y_values) > ymin + tolerance) or (
            np.max(y_values) < ymax - tolerance
        )
        edges.append(
            EdgeSegment(
                edge="right",
                range_start=float(np.min(y_values)),
                range_end=float(np.max(y_values)),
                is_partial=is_partial,
            )
        )

    if bottom_count >= threshold:
        x_values = sync_points[bottom_mask, 0]
        is_partial = (np.min(x_values) > xmin + tolerance) or (
            np.max(x_values) < xmax - tolerance
        )
        edges.append(
            EdgeSegment(
                edge="bottom",
                range_start=float(np.min(x_values)),
                range_end=float(np.max(x_values)),
                is_partial=is_partial,
            )
        )

    if top_count >= threshold:
        x_values = sync_points[top_mask, 0]
        is_partial = (np.min(x_values) > xmin + tolerance) or (
            np.max(x_values) < xmax - tolerance
        )
        edges.append(
            EdgeSegment(
                edge="top",
                range_start=float(np.min(x_values)),
                range_end=float(np.max(x_values)),
                is_partial=is_partial,
            )
        )

    # Log diagnostics
    if len(edges) == 0:
        logger.warning(
            f"Sync points do not align with any edge within tolerance {tolerance}. "
            f"Bounds: x=[{xmin:.6f}, {xmax:.6f}], y=[{ymin:.6f}, {ymax:.6f}]. "
            f"Edge counts: left={left_count}, right={right_count}, "
            f"bottom={bottom_count}, top={top_count}, threshold={threshold}"
        )
    elif len(edges) > 2:
        logger.warning(
            f"Sync points match {len(edges)} edges: {[e.edge for e in edges]}. "
            "This may indicate corner-only contact or numerical issues."
        )
    elif len(edges) == 2:
        edge_names = {e.edge for e in edges}
        opposite_pairs = [{"left", "right"}, {"bottom", "top"}]
        if edge_names in opposite_pairs:
            logger.warning(
                f"Sync points match opposite edges: {list(edge_names)}. "
                "This is unusual and may indicate an issue."
            )

    return edges


# =============================================================================
# Mask Initialization
# =============================================================================


def initialize_all_slip_masks(
    all_n_cp: list[tuple[int, int]],
) -> list[np.ndarray]:
    """Initialize masks with slip boundary conditions on all boundaries.

    Creates masks where:
    - Interior control points are fully free (NaN in both dimensions)
    - Edge control points have slip conditions (fixed normal, free tangent)
    - Corner control points are fully fixed (0.0 in both dimensions)

    Args:
        all_n_cp: List of (n_cp_x, n_cp_y) tuples for each region

    Returns:
        List of mask arrays with shape (n_cp_x, n_cp_y, 2)

    Example:
        >>> masks = initialize_all_slip_masks([(5, 5), (6, 4)])
        >>> # Region 0 has 5x5 CPs, Region 1 has 6x4 CPs
        >>> print(masks[0].shape)  # (5, 5, 2)
    """
    masks = []
    for n_cp_x, n_cp_y in all_n_cp:
        # Start with all CPs free
        mask = np.full((n_cp_x, n_cp_y, 2), np.nan)

        # Apply slip conditions on boundaries
        mask[[0, -1], :, 0] = 0.0  # Left/right edges: fix x, free y (slip)
        mask[:, [0, -1], 1] = 0.0  # Bottom/top edges: fix y, free x (slip)

        # Corners are fully fixed
        mask[[0, -1], [0, -1], :] = 0.0

        masks.append(mask)

    logger.info(f"Initialized {len(masks)} slip masks")
    return masks


# =============================================================================
# Relaxation Logic for Internal Boundaries
# =============================================================================


def relax_edge_segment_normal(
    mask: np.ndarray,
    segment: EdgeSegment,
    cp_coords: tuple[np.ndarray, np.ndarray],
    tolerance: float,
    exclude_hinges: bool = True,
) -> None:
    """Relax normal direction for CPs along specified edge segment.

    Modifies mask in-place to set normal direction to NaN (free) for control
    points lying on the edge segment within the specified range.

    Hinge points are endpoints of partial edge segments that connect to
    external boundaries. If exclude_hinges is True, these points remain
    fixed to maintain continuity.

    Args:
        mask: Mask array to modify, shape (n_cp_x, n_cp_y, 2)
        segment: EdgeSegment describing which edge and range to relax
        cp_coords: Tuple of (x_coords, y_coords) for control points
        tolerance: Spatial tolerance for comparisons
        exclude_hinges: If True, keep hinge points fixed

    Example:
        >>> segment = EdgeSegment(edge='left', range_start=0.2, range_end=0.8, is_partial=True)
        >>> relax_edge_segment_normal(mask, segment, cp_coords, 1e-6, exclude_hinges=True)
        >>> # CPs on left edge between y=0.2 and y=0.8 now have x-direction free
    """
    x_coords, y_coords = cp_coords
    bounds_xmin, bounds_xmax = x_coords[0], x_coords[-1]
    bounds_ymin, bounds_ymax = y_coords[0], y_coords[-1]

    # Determine the valid range for relaxation, accounting for hinges
    if segment.edge in ["left", "right"]:
        # Vertical edges: range is in y-direction
        valid_min = segment.range_start - tolerance
        valid_max = segment.range_end + tolerance

        # If segment doesn't start at domain boundary, exclude start point (hinge)
        if exclude_hinges and segment.range_start > bounds_ymin + tolerance:
            valid_min = segment.range_start + tolerance

        # If segment doesn't end at domain boundary, exclude end point (hinge)
        if exclude_hinges and segment.range_end < bounds_ymax - tolerance:
            valid_max = segment.range_end - tolerance

        # Find CPs within valid range
        mask_y = (y_coords >= valid_min) & (y_coords <= valid_max)
        n_relaxed = np.sum(mask_y)

        if segment.edge == "left":
            # Relax x-displacement (normal) for CPs on left edge
            mask[0, mask_y, 0] = np.nan
            logger.info(
                f"Relaxed left edge (x-disp) for {n_relaxed} CPs "
                f"(y-range: [{valid_min:.4f}, {valid_max:.4f}])"
            )
        else:  # right
            # Relax x-displacement (normal) for CPs on right edge
            mask[-1, mask_y, 0] = np.nan
            logger.info(
                f"Relaxed right edge (x-disp) for {n_relaxed} CPs "
                f"(y-range: [{valid_min:.4f}, {valid_max:.4f}])"
            )

    else:  # bottom or top
        # Horizontal edges: range is in x-direction
        valid_min = segment.range_start - tolerance
        valid_max = segment.range_end + tolerance

        # If segment doesn't start at domain boundary, exclude start point (hinge)
        if exclude_hinges and segment.range_start > bounds_xmin + tolerance:
            valid_min = segment.range_start + tolerance

        # If segment doesn't end at domain boundary, exclude end point (hinge)
        if exclude_hinges and segment.range_end < bounds_xmax - tolerance:
            valid_max = segment.range_end - tolerance

        # Find CPs within valid range
        mask_x = (x_coords >= valid_min) & (x_coords <= valid_max)
        n_relaxed = np.sum(mask_x)

        if segment.edge == "bottom":
            # Relax y-displacement (normal) for CPs on bottom edge
            mask[mask_x, 0, 1] = np.nan
            logger.info(
                f"Relaxed bottom edge (y-disp) for {n_relaxed} CPs "
                f"(x-range: [{valid_min:.4f}, {valid_max:.4f}])"
            )
        else:  # top
            # Relax y-displacement (normal) for CPs on top edge
            mask[mask_x, -1, 1] = np.nan
            logger.info(
                f"Relaxed top edge (y-disp) for {n_relaxed} CPs "
                f"(x-range: [{valid_min:.4f}, {valid_max:.4f}])"
            )


def relax_internal_boundary_masks(
    masks: list[np.ndarray],
    all_n_cp: list[tuple[int, int]],
    region_meshes: Sequence[Mesh],
    inter_region_sync: list[tuple[int, int, np.ndarray]],
    tolerance: float,
) -> None:
    """Relax normal constraints on shared (internal) boundaries.

    Modifies masks in-place to allow movement in the normal direction for
    control points on boundaries that are shared between regions.

    Args:
        masks: List of mask arrays to modify
        all_n_cp: List of (n_cp_x, n_cp_y) for each region
        region_meshes: List of Mesh objects for each region
        inter_region_sync: List of (region_i, region_j, sync_points) tuples
        tolerance: Spatial tolerance for comparisons

    Example:
        >>> relax_internal_boundary_masks(masks, all_n_cp, meshes, sync_data, 1e-6)
        >>> # Masks are modified in-place
    """
    for region_i, region_j, sync_points in inter_region_sync:
        if len(sync_points) == 0:
            logger.warning(
                f"Empty sync_points for regions {region_i} and {region_j}, skipping"
            )
            continue

        # Relax region i
        bounds_i = region_meshes[region_i].get_bounds()
        segments_i = identify_shared_edge_segments(bounds_i, sync_points, tolerance)
        cp_coords_i = get_control_point_coordinates(bounds_i, all_n_cp[region_i])

        for segment in segments_i:
            relax_edge_segment_normal(
                masks[region_i],
                segment,
                cp_coords_i,
                tolerance,
                exclude_hinges=True,
            )

        # Relax region j
        bounds_j = region_meshes[region_j].get_bounds()
        segments_j = identify_shared_edge_segments(bounds_j, sync_points, tolerance)
        cp_coords_j = get_control_point_coordinates(bounds_j, all_n_cp[region_j])

        for segment in segments_j:
            relax_edge_segment_normal(
                masks[region_j],
                segment,
                cp_coords_j,
                tolerance,
                exclude_hinges=True,
            )

    logger.info(
        f"Relaxed internal boundaries for {len(inter_region_sync)} region pairs"
    )


# =============================================================================
# Unification Logic for Coincident Control Points
# =============================================================================


def find_coincident_control_points(
    region_i: int,
    region_j: int,
    all_n_cp: list[tuple[int, int]],
    region_meshes: Sequence[Mesh],
    segments_i: list[EdgeSegment],
    segments_j: list[EdgeSegment],
    tolerance: float,
) -> list[tuple[tuple[int, int], tuple[int, int], np.ndarray]]:
    """Find pairs of coincident control points between two regions.

    Args:
        region_i: Index of first region
        region_j: Index of second region
        all_n_cp: List of (n_cp_x, n_cp_y) for each region
        region_meshes: List of Mesh objects for each region
        segments_i: List of EdgeSegments for region i
        segments_j: List of EdgeSegments for region j
        tolerance: Spatial tolerance for coincidence detection

    Returns:
        List of tuples (idx_i, idx_j, position) where:
        - idx_i: (ix, iy) indices in region i
        - idx_j: (jx, jy) indices in region j
        - position: (x, y) spatial coordinates

    Example:
        >>> coincident = find_coincident_control_points(0, 1, all_n_cp, meshes, segs_i, segs_j, 1e-6)
        >>> for (i_idx, j_idx, pos) in coincident:
        ...     print(f"CP at {pos} is shared between regions")
    """
    bounds_i = region_meshes[region_i].get_bounds()
    bounds_j = region_meshes[region_j].get_bounds()

    # Get all CPs on the relevant edges for each region
    cps_i = _get_edge_control_points(region_i, bounds_i, all_n_cp[region_i], segments_i)
    cps_j = _get_edge_control_points(region_j, bounds_j, all_n_cp[region_j], segments_j)

    # Find coincident pairs
    coincident = []
    for idx_i, pos_i in cps_i:
        for idx_j, pos_j in cps_j:
            distance = np.linalg.norm(pos_i - pos_j)
            if distance < tolerance:
                coincident.append((idx_i, idx_j, pos_i))

    logger.debug(
        f"Found {len(coincident)} coincident CPs between regions {region_i} and {region_j}"
    )
    return coincident


def _get_edge_control_points(
    region_idx: int,
    bounds: DomainBounds,
    n_cp: tuple[int, int],
    segments: list[EdgeSegment],
) -> list[tuple[tuple[int, int], np.ndarray]]:
    """Get control points lying on specified edge segments.

    Helper function for find_coincident_control_points.

    Args:
        region_idx: Region index (for logging)
        bounds: Domain bounds of the region
        n_cp: Number of control points (n_cp_x, n_cp_y)
        segments: List of EdgeSegments to extract CPs from

    Returns:
        List of ((ix, iy), position) tuples
    """
    x_coords, y_coords = get_control_point_coordinates(bounds, n_cp)
    n_cp_x, n_cp_y = n_cp

    cps = []
    for segment in segments:
        if segment.edge == "left":
            idx_x = 0
            for iy in range(n_cp_y):
                cps.append(((idx_x, iy), np.array([x_coords[idx_x], y_coords[iy]])))
        elif segment.edge == "right":
            idx_x = n_cp_x - 1
            for iy in range(n_cp_y):
                cps.append(((idx_x, iy), np.array([x_coords[idx_x], y_coords[iy]])))
        elif segment.edge == "bottom":
            idx_y = 0
            for ix in range(n_cp_x):
                cps.append(((ix, idx_y), np.array([x_coords[ix], y_coords[idx_y]])))
        elif segment.edge == "top":
            idx_y = n_cp_y - 1
            for ix in range(n_cp_x):
                cps.append(((ix, idx_y), np.array([x_coords[ix], y_coords[idx_y]])))

    return cps


def propagate_constraint_between_masks(
    mask_a: np.ndarray,
    idx_a: tuple[int, int],
    mask_b: np.ndarray,
    idx_b: tuple[int, int],
) -> None:
    """Propagate fixed constraints from one mask entry to another.

    If one mask has a fixed constraint (non-NaN) in a dimension and the other
    doesn't, propagate the constraint to ensure consistency.

    Modifies masks in-place.

    Args:
        mask_a: First mask array
        idx_a: (ix, iy) index in first mask
        mask_b: Second mask array
        idx_b: (jx, jy) index in second mask

    Example:
        >>> # If mask_a[2,3,0] = 0.0 (fixed) and mask_b[1,1,0] = NaN (free)
        >>> propagate_constraint_between_masks(mask_a, (2,3), mask_b, (1,1))
        >>> # Now mask_b[1,1,0] = 0.0 (fixed)
    """
    for dim in [0, 1]:
        val_a = mask_a[idx_a][dim]
        val_b = mask_b[idx_b][dim]

        is_fixed_a = not np.isnan(val_a)
        is_fixed_b = not np.isnan(val_b)

        if is_fixed_a and not is_fixed_b:
            # Propagate A's constraint to B
            mask_b[idx_b][dim] = val_a
            logger.debug(
                f"Propagated constraint val={val_a} from A{idx_a} to B{idx_b} at dim {dim}"
            )
        elif is_fixed_b and not is_fixed_a:
            # Propagate B's constraint to A
            mask_a[idx_a][dim] = val_b
            logger.debug(
                f"Propagated constraint val={val_b} from B{idx_b} to A{idx_a} at dim {dim}"
            )


def unify_coincident_control_point_masks(
    masks: list[np.ndarray],
    all_n_cp: list[tuple[int, int]],
    region_meshes: Sequence[Mesh],
    inter_region_sync: list[tuple[int, int, np.ndarray]],
    tolerance: float,
) -> None:
    """Ensure consistent constraints for coincident control points.

    Modifies masks in-place to propagate fixed constraints between control
    points at the same spatial location in different regions.

    Args:
        masks: List of mask arrays to modify
        all_n_cp: List of (n_cp_x, n_cp_y) for each region
        region_meshes: List of Mesh objects for each region
        inter_region_sync: List of (region_i, region_j, sync_points) tuples
        tolerance: Spatial tolerance for coincidence detection

    Example:
        >>> unify_coincident_control_point_masks(masks, all_n_cp, meshes, sync_data, 1e-6)
        >>> # Masks are modified in-place
    """
    for region_i, region_j, sync_points in inter_region_sync:
        if len(sync_points) == 0:
            continue

        # Identify edges involved in sync
        bounds_i = region_meshes[region_i].get_bounds()
        bounds_j = region_meshes[region_j].get_bounds()
        segments_i = identify_shared_edge_segments(bounds_i, sync_points, tolerance)
        segments_j = identify_shared_edge_segments(bounds_j, sync_points, tolerance)

        # Find coincident CPs
        coincident = find_coincident_control_points(
            region_i,
            region_j,
            all_n_cp,
            region_meshes,
            segments_i,
            segments_j,
            tolerance,
        )

        # Unify their masks
        for idx_i, idx_j, _ in coincident:
            propagate_constraint_between_masks(
                masks[region_i], idx_i, masks[region_j], idx_j
            )

    logger.info(f"Unified constraints for {len(inter_region_sync)} region pairs")


# =============================================================================
# High-Level Public API
# =============================================================================


def create_slip_on_external_boundaries_masks(
    all_n_cp: list[tuple[int, int]],
    region_meshes: Sequence[RegularGrid],
    inter_region_sync: list[tuple[int, int, np.ndarray]] | None,
    tolerance: float | None = None,
) -> list[np.ndarray]:
    """Create masks for slip conditions on external boundaries only.

    Internal boundaries between regions are relaxed to allow inter-region
    continuity through the optimization process.

    This is the main public API function for creating boundary condition masks
    in the "slip_on_external_boundaries" mode.

    Args:
        all_n_cp: List of (n_cp_x, n_cp_y) tuples for each region
        region_meshes: List of Mesh objects defining each region
        inter_region_sync: List of (region_i, region_j, sync_points) tuples
            defining shared boundaries, or None if no shared boundaries
        tolerance: Spatial tolerance for comparisons, or None to auto-compute

    Returns:
        List of mask arrays with shape (n_cp_x, n_cp_y, 2), one per region

    Raises:
        ValueError: If mask validation fails

    Example:
        >>> masks = create_slip_on_external_boundaries_masks(
        ...     all_n_cp=[(5, 5), (6, 4)],
        ...     region_meshes=[mesh1, mesh2],
        ...     inter_region_sync=[(0, 1, shared_points)],
        ...     tolerance=1e-6,
        ... )
        >>> # masks[0] has slip BCs on external boundaries, relaxed on shared boundary
    """
    logger.info("Creating slip_on_external_boundaries masks")

    # Compute tolerance
    tol = compute_spatial_tolerance(region_meshes, tolerance)

    # Step 1: Initialize with slip on all boundaries
    masks = initialize_all_slip_masks(all_n_cp)

    # Step 2: Relax constraints on internal (shared) boundaries
    if inter_region_sync is not None and len(inter_region_sync) > 0:
        relax_internal_boundary_masks(
            masks, all_n_cp, region_meshes, inter_region_sync, tol
        )

        # Step 3: Unify constraints for coincident CPs
        unify_coincident_control_point_masks(
            masks, all_n_cp, region_meshes, inter_region_sync, tol
        )
    else:
        logger.info("No inter-region sync specified, all boundaries are external")

    # Step 4: Validate the result
    validation = validate_mask_configuration(
        masks, all_n_cp, region_meshes, inter_region_sync, tol
    )

    if not validation["is_valid"]:
        error_msg = "Invalid mask configuration:\n" + "\n".join(validation["issues"])
        logger.error(error_msg)
        raise ValueError(error_msg)

    if validation["warnings"]:
        for warning in validation["warnings"]:
            logger.warning(warning)

    logger.info("Successfully created slip_on_external_boundaries masks")
    return masks


# =============================================================================
# Validation and Debugging Utilities
# =============================================================================


def count_control_point_states(mask: np.ndarray) -> dict[str, int]:
    """Count control points by state.

    Args:
        mask: Mask array of shape (n_cp_x, n_cp_y, 2)

    Returns:
        Dictionary with counts:
        - 'fully_fixed': Both dimensions have non-NaN values
        - 'slip_x': x-direction fixed, y-direction free
        - 'slip_y': y-direction fixed, x-direction free
        - 'fully_free': Both dimensions are NaN

    Example:
        >>> counts = count_control_point_states(mask)
        >>> print(f"Free CPs: {counts['fully_free']}")
    """
    n_cp_x, n_cp_y, _ = mask.shape

    fully_fixed = 0
    slip_x = 0
    slip_y = 0
    fully_free = 0

    for i in range(n_cp_x):
        for j in range(n_cp_y):
            x_free = np.isnan(mask[i, j, 0])
            y_free = np.isnan(mask[i, j, 1])

            if not x_free and not y_free:
                fully_fixed += 1
            elif not x_free and y_free:
                slip_x += 1
            elif x_free and not y_free:
                slip_y += 1
            else:
                fully_free += 1

    return {
        "fully_fixed": fully_fixed,
        "slip_x": slip_x,
        "slip_y": slip_y,
        "fully_free": fully_free,
    }


def validate_mask_configuration(
    masks: list[np.ndarray],
    all_n_cp: list[tuple[int, int]],
    region_meshes: Sequence[Mesh],
    inter_region_sync: list[tuple[int, int, np.ndarray]] | None,
    tolerance: float,
) -> dict[str, Any]:
    """Validate mask configuration and return diagnostic information.

    Args:
        masks: List of mask arrays to validate
        all_n_cp: List of (n_cp_x, n_cp_y) for each region
        region_meshes: List of Mesh objects for each region
        inter_region_sync: List of (region_i, region_j, sync_points) tuples
        tolerance: Spatial tolerance used for mask creation

    Returns:
        Dictionary with keys:
        - 'is_valid': bool indicating if configuration is valid
        - 'issues': list of error messages (empty if valid)
        - 'warnings': list of warning messages
        - 'statistics': dict with CP state counts per region

    Example:
        >>> validation = validate_mask_configuration(masks, all_n_cp, meshes, sync, 1e-6)
        >>> if not validation['is_valid']:
        ...     print("Errors:", validation['issues'])
    """
    issues = []
    warnings = []
    statistics = {}

    # Check number of masks matches number of regions
    if len(masks) != len(all_n_cp):
        issues.append(
            f"Number of masks ({len(masks)}) doesn't match number of regions ({len(all_n_cp)})"
        )
        return {
            "is_valid": False,
            "issues": issues,
            "warnings": warnings,
            "statistics": statistics,
        }

    # Validate each mask
    for idx, (mask, (n_cp_x, n_cp_y)) in enumerate(zip(masks, all_n_cp)):
        # Check shape
        if mask.shape != (n_cp_x, n_cp_y, 2):
            issues.append(
                f"Region {idx}: Expected shape ({n_cp_x}, {n_cp_y}, 2), got {mask.shape}"
            )
            continue

        # Count states
        counts = count_control_point_states(mask)
        statistics[f"region_{idx}"] = counts

        # Check for corner points being fixed
        for i, j in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
            if np.isnan(mask[i, j, 0]) or np.isnan(mask[i, j, 1]):
                warnings.append(
                    f"Region {idx}: Corner CP ({i}, {j}) is not fully fixed"
                )

        # Warn if no CPs are free (might be over-constrained)
        if (
            counts["fully_free"] == 0
            and counts["slip_x"] == 0
            and counts["slip_y"] == 0
        ):
            warnings.append(
                f"Region {idx}: All CPs are fully fixed (possible over-constraint)"
            )

    # Check coincident CPs have consistent constraints
    if inter_region_sync is not None:
        for region_i, region_j, sync_points in inter_region_sync:
            if len(sync_points) == 0:
                continue

            bounds_i = region_meshes[region_i].get_bounds()
            bounds_j = region_meshes[region_j].get_bounds()
            segments_i = identify_shared_edge_segments(bounds_i, sync_points, tolerance)
            segments_j = identify_shared_edge_segments(bounds_j, sync_points, tolerance)

            coincident = find_coincident_control_points(
                region_i,
                region_j,
                all_n_cp,
                region_meshes,
                segments_i,
                segments_j,
                tolerance,
            )

            for idx_i, idx_j, pos in coincident:
                mask_i_val = masks[region_i][idx_i]
                mask_j_val = masks[region_j][idx_j]

                for dim in [0, 1]:
                    is_fixed_i = not np.isnan(mask_i_val[dim])
                    is_fixed_j = not np.isnan(mask_j_val[dim])

                    if is_fixed_i != is_fixed_j:
                        issues.append(
                            f"Inconsistent constraint at coincident CP {pos}: "
                            f"region {region_i}{idx_i} dim {dim} fixed={is_fixed_i}, "
                            f"region {region_j}{idx_j} dim {dim} fixed={is_fixed_j}"
                        )

    is_valid = len(issues) == 0

    return {
        "is_valid": is_valid,
        "issues": issues,
        "warnings": warnings,
        "statistics": statistics,
    }


def visualize_mask_configuration(
    masks: list[np.ndarray],
    region_meshes: Sequence[Mesh],
    all_n_cp: list[tuple[int, int]],
    output_path: str | None = None,
) -> None:
    """Create visualization of control point states across all regions.

    Creates a matplotlib figure showing:
    - Control point locations colored by state (fixed/slip/free)
    - Region boundaries
    - Legend explaining colors

    Args:
        masks: List of mask arrays to visualize
        region_meshes: List of Mesh objects for each region
        all_n_cp: List of (n_cp_x, n_cp_y) for each region
        output_path: Optional path to save figure, or None to display

    Example:
        >>> visualize_mask_configuration(masks, meshes, all_n_cp, "bc_config.png")
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        logger.error("matplotlib not available for visualization")
        return

    fig, ax = plt.subplots(figsize=(12, 10))

    # Define colors for each state
    colors = {
        "fully_fixed": "red",
        "slip_x": "orange",
        "slip_y": "yellow",
        "fully_free": "green",
    }

    for region_idx, (mask, mesh, (n_cp_x, n_cp_y)) in enumerate(
        zip(masks, region_meshes, all_n_cp)
    ):
        bounds = mesh.get_bounds()
        x_coords, y_coords = get_control_point_coordinates(bounds, (n_cp_x, n_cp_y))

        # Draw region boundary
        width = bounds.maxima[0] - bounds.minima[0]
        height = bounds.maxima[1] - bounds.minima[1]
        rect = Rectangle(
            (bounds.minima[0], bounds.minima[1]),
            width,
            height,
            fill=False,
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(rect)

        # Plot control points colored by state
        for i in range(n_cp_x):
            for j in range(n_cp_y):
                x, y = x_coords[i], y_coords[j]
                x_free = np.isnan(mask[i, j, 0])
                y_free = np.isnan(mask[i, j, 1])

                if not x_free and not y_free:
                    state = "fully_fixed"
                elif not x_free and y_free:
                    state = "slip_x"
                elif x_free and not y_free:
                    state = "slip_y"
                else:
                    state = "fully_free"

                ax.plot(x, y, "o", color=colors[state], markersize=8)

        # Add region label
        center_x = (bounds.minima[0] + bounds.maxima[0]) / 2
        center_y = (bounds.minima[1] + bounds.maxima[1]) / 2
        ax.text(
            center_x,
            center_y,
            f"Region {region_idx}",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Create legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[state],
            markersize=10,
            label=state.replace("_", " ").title(),
        )
        for state in ["fully_fixed", "slip_x", "slip_y", "fully_free"]
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Control Point Boundary Conditions")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


# =============================================================================
# Region Adjacency Detection
# =============================================================================


def find_adjacent_regions(
    region_meshes: list[RegularGrid],
    tolerance: float | None = None,
    n_sample_points_per_edge: int = 50,
) -> list[tuple[int, int, np.ndarray]]:
    """Find adjacent FFD regions and their shared boundary points.

    This function detects pairs of regions that share a boundary edge and
    returns the points along that shared boundary. These points can be used
    to enforce continuity constraints between adjacent FFD mappings.

    Args:
        region_meshes: List of RegularGrid objects defining each FFD region.
        tolerance: Distance tolerance for considering points as shared.
            If None, automatically computed as 1% of the smallest region diagonal.
        n_sample_points_per_edge: Number of points to sample along each edge
            of the region bounding boxes.

    Returns:
        List of tuples (region_i, region_j, shared_points) where:
            - region_i, region_j: Indices of adjacent regions (i < j)
            - shared_points: Array of shape (n_points, 2) containing points
              that lie on the shared boundary between the two regions.

    Example:
        >>> region_meshes = [mesh_top, mesh_bottom]
        >>> adjacencies = find_adjacent_regions(region_meshes, tolerance=0.01)
        >>> for i, j, pts in adjacencies:
        ...     print(f"Regions {i} and {j} share {len(pts)} boundary points")
    """
    n_regions = len(region_meshes)
    if n_regions < 2:
        return []

    # Get bounds for each region
    all_bounds = [mesh.get_bounds() for mesh in region_meshes]

    # Auto-compute tolerance if not provided
    if tolerance is None:
        diagonals = [np.linalg.norm(b.lengths) for b in all_bounds]
        tolerance = float(0.01 * min(diagonals))

    adjacencies: list[tuple[int, int, np.ndarray]] = []

    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            bounds_i = all_bounds[i]
            bounds_j = all_bounds[j]

            # Sample points along edges of region i
            edges_i = _sample_boundary_edges(bounds_i, n_sample_points_per_edge)
            # Sample points along edges of region j
            edges_j = _sample_boundary_edges(bounds_j, n_sample_points_per_edge)

            # Find points from region i that are close to region j boundary
            shared_points = _find_shared_boundary_points(edges_i, edges_j, tolerance)

            if len(shared_points) > 0:
                adjacencies.append((i, j, shared_points))

    return adjacencies


def _sample_boundary_edges(bounds: DomainBounds, n_points: int) -> np.ndarray:
    """Sample points along all 4 edges of a 2D bounding box.

    Args:
        bounds: Domain bounds defining the box.
        n_points: Number of points to sample per edge.

    Returns:
        Array of shape (4 * n_points, 2) containing sampled boundary points.
    """
    xmin, ymin = bounds.minima
    xmax, ymax = bounds.maxima

    # Bottom edge (y = ymin)
    bottom = np.column_stack(
        [np.linspace(xmin, xmax, n_points), np.full(n_points, ymin)]
    )
    # Top edge (y = ymax)
    top = np.column_stack([np.linspace(xmin, xmax, n_points), np.full(n_points, ymax)])
    # Left edge (x = xmin)
    left = np.column_stack([np.full(n_points, xmin), np.linspace(ymin, ymax, n_points)])
    # Right edge (x = xmax)
    right = np.column_stack(
        [np.full(n_points, xmax), np.linspace(ymin, ymax, n_points)]
    )

    return np.vstack([bottom, top, left, right])


def _find_shared_boundary_points(
    edges_i: np.ndarray,
    edges_j: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Find points that are shared between two sets of boundary edges.

    Uses a simple distance-based approach. Points from edges_i that are
    within tolerance distance of any point in edges_j are considered shared.

    Args:
        edges_i: Points along boundary of region i, shape (n_i, 2).
        edges_j: Points along boundary of region j, shape (n_j, 2).
        tolerance: Maximum distance to consider points as shared.

    Returns:
        Array of shared points, shape (n_shared, 2).
    """
    try:
        from scipy.spatial import cKDTree

        tree_j = cKDTree(edges_j)
        distances, _ = tree_j.query(edges_i, k=1)
        mask = distances <= tolerance
        return edges_i[mask]
    except ImportError:
        # Fallback without scipy - slower but works
        shared = []
        for pt_i in edges_i:
            dists = np.linalg.norm(edges_j - pt_i, axis=1)
            if np.min(dists) <= tolerance:
                shared.append(pt_i)
        return np.array(shared) if shared else np.empty((0, 2))
