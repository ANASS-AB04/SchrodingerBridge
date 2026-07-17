from collections.abc import Mapping, Sequence
from typing import Literal, overload

import numpy as np

from phdtruel import visualisations
from phdtruel.fields.field_of_interest import FieldOfInterest


def _as_2d_points(points: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    points_array = np.asarray(points)
    if points_array.ndim == 2 and points_array.shape[1] == 2:
        return points_array
    if points_array.ndim == 3 and points_array.shape[-1] == 2:
        return points_array.reshape(-1, 2)
    raise ValueError("Control-point arrays must have shape (N, 2) or (A, B, 2).")


def _normalise_regions(
    points: np.ndarray | Sequence[np.ndarray | Sequence[Sequence[float]]],
) -> list[np.ndarray]:
    if isinstance(points, np.ndarray):
        return [_as_2d_points(points)]
    return [_as_2d_points(region_points) for region_points in points]


def _normalise_displacements(
    displacements: np.ndarray | Sequence[np.ndarray] | None,
    *,
    n_regions: int,
) -> list[np.ndarray | None]:
    if displacements is None:
        return [None] * n_regions

    if isinstance(displacements, np.ndarray):
        normalised = [_as_2d_points(displacements)]
    else:
        normalised = [_as_2d_points(region_disp) for region_disp in displacements]

    if len(normalised) != n_regions:
        raise ValueError(
            f"Displacements contain {len(normalised)} regions but points contain "
            f"{n_regions} regions."
        )
    return normalised


def _normalise_sync_points(
    sync_points: np.ndarray
    | Sequence[np.ndarray]
    | Sequence[tuple[int, int, np.ndarray]],
) -> list[tuple[str, np.ndarray]]:
    if isinstance(sync_points, np.ndarray):
        return [("sync", _as_2d_points(sync_points))]

    normalised: list[tuple[str, np.ndarray]] = []
    for item in sync_points:
        if isinstance(item, tuple) and len(item) == 3:
            region_i, region_j, points = item
            normalised.append((f"sync {region_i}-{region_j}", _as_2d_points(points)))
        else:
            normalised.append(("sync", _as_2d_points(item)))
    return normalised


def _selector_to_points(selector, reference_points: np.ndarray) -> np.ndarray:
    selector_array = np.asarray(selector)
    if selector_array.ndim == 1 and selector_array.dtype == bool:
        if selector_array.shape[0] != reference_points.shape[0]:
            raise ValueError("Boolean selector length must match number of points.")
        return reference_points[selector_array]
    if selector_array.ndim == 1 and np.issubdtype(selector_array.dtype, np.integer):
        return reference_points[selector_array]
    return _as_2d_points(selector_array)


def classify_region_constraint_points(
    control_points: np.ndarray | Sequence[Sequence[float]],
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify one region's control points by constraint status.

    Constraint semantics follow the boundary-condition mask used by optimization:
    - free: both displacement components are NaN
    - fixed: both displacement components are finite
    - partial: mixed NaN/finite components
    """
    cp_array = _as_2d_points(control_points)
    mask_flat = np.asarray(mask).reshape(-1, 2)
    if mask_flat.shape[0] != cp_array.shape[0]:
        raise ValueError(
            "Mask and control points are inconsistent: "
            f"mask has {mask_flat.shape[0]} points but control points have "
            f"{cp_array.shape[0]}."
        )

    fixed_selector = np.all(~np.isnan(mask_flat), axis=1)
    free_selector = np.all(np.isnan(mask_flat), axis=1)
    partial_selector = ~(fixed_selector | free_selector)

    return (
        cp_array[free_selector],
        cp_array[partial_selector],
        cp_array[fixed_selector],
    )


@overload
def classify_constraint_status(
    control_points: np.ndarray | Sequence[np.ndarray],
    masks: np.ndarray | Sequence[np.ndarray | object],
    *,
    stack: Literal[True] = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


@overload
def classify_constraint_status(
    control_points: np.ndarray | Sequence[np.ndarray],
    masks: np.ndarray | Sequence[np.ndarray | object],
    *,
    stack: Literal[False],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]: ...


def classify_constraint_status(
    control_points: np.ndarray | Sequence[np.ndarray],
    masks: np.ndarray | Sequence[np.ndarray | object],
    *,
    stack: bool = True,
) -> tuple[
    np.ndarray | list[np.ndarray],
    np.ndarray | list[np.ndarray],
    np.ndarray | list[np.ndarray],
]:
    """Classify free/partial/fixed control points from BC masks.

    Args:
        control_points: Control points as one region array or a region list.
        masks: BC masks matching ``control_points`` region layout.
        stack: If True, returns 3 stacked arrays (free, partial, fixed).
            If False, returns 3 per-region lists.
    """
    region_points = _normalise_regions(control_points)
    if isinstance(masks, np.ndarray):
        region_masks = [masks]
    else:
        region_masks = [np.asarray(mask) for mask in masks]

    if len(region_points) != len(region_masks):
        raise ValueError(
            "Number of region masks and control-point regions must match: "
            f"{len(region_masks)} != {len(region_points)}."
        )

    free_per_region: list[np.ndarray] = []
    partial_per_region: list[np.ndarray] = []
    fixed_per_region: list[np.ndarray] = []

    for cp_array, mask in zip(region_points, region_masks):
        free, partial, fixed = classify_region_constraint_points(cp_array, mask)
        free_per_region.append(free)
        partial_per_region.append(partial)
        fixed_per_region.append(fixed)

    if not stack:
        return free_per_region, partial_per_region, fixed_per_region

    def _stack_or_empty(groups: list[np.ndarray]) -> np.ndarray:
        if any(group.size > 0 for group in groups):
            return np.vstack(groups)
        return np.empty((0, 2))

    return (
        _stack_or_empty(free_per_region),
        _stack_or_empty(partial_per_region),
        _stack_or_empty(fixed_per_region),
    )


def overlay_control_points(
    fig,
    ax,
    control_points: np.ndarray | Sequence[np.ndarray],
    displacements: np.ndarray | Sequence[np.ndarray] | None = None,
    *,
    region_colors: Sequence[str] = ("red", "blue", "green", "orange"),
    region_markers: Sequence[str] = ("o", "s", "^", "D"),
    show_displacements: bool = True,
) -> tuple:
    region_points = _normalise_regions(control_points)
    region_displacements = _normalise_displacements(
        displacements,
        n_regions=len(region_points),
    )

    for region_index, (points, region_displacement) in enumerate(
        zip(region_points, region_displacements)
    ):
        color = region_colors[region_index % len(region_colors)]
        marker = region_markers[region_index % len(region_markers)]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=45,
            c=color,
            marker=marker,
            edgecolors="black",
            linewidths=0.8,
            alpha=0.85,
            zorder=10,
            label=f"CP region {region_index}",
        )

        if show_displacements and region_displacement is not None:
            displaced_points = points + region_displacement
            ax.scatter(
                displaced_points[:, 0],
                displaced_points[:, 1],
                s=38,
                c=color,
                marker=marker,
                edgecolors="black",
                linewidths=0.6,
                alpha=0.4,
                zorder=9,
            )
            ax.quiver(
                points[:, 0],
                points[:, 1],
                region_displacement[:, 0],
                region_displacement[:, 1],
                angles="xy",
                scale_units="xy",
                scale=1,
                color=color,
                width=0.003,
                alpha=0.8,
                zorder=8,
            )

    return fig, ax


def overlay_sync_points(
    fig,
    ax,
    sync_points: np.ndarray
    | Sequence[np.ndarray]
    | Sequence[tuple[int, int, np.ndarray]],
    *,
    marker: str = "x",
    color: str = "yellow",
) -> tuple:
    for label, points in _normalise_sync_points(sync_points):
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=35,
            c=color,
            marker=marker,
            edgecolors="black",
            linewidths=1.0,
            zorder=11,
            label=label,
        )
    return fig, ax


def overlay_constraint_status(
    fig,
    ax,
    control_points: np.ndarray | Sequence[np.ndarray],
    *,
    free: np.ndarray | Sequence[int] | Sequence[Sequence[float]] | None = None,
    partial: np.ndarray | Sequence[int] | Sequence[Sequence[float]] | None = None,
    fixed: np.ndarray | Sequence[int] | Sequence[Sequence[float]] | None = None,
) -> tuple:
    all_points = np.vstack(_normalise_regions(control_points))
    categories = {
        "free": (free, dict(color="tab:green", marker="o", alpha=0.8)),
        "partial": (partial, dict(color="tab:orange", marker="^", alpha=0.9)),
        "fixed": (fixed, dict(color="tab:red", marker="s", alpha=0.9)),
    }

    for name, (selector, style) in categories.items():
        if selector is None:
            continue
        selected_points = _selector_to_points(selector, all_points)
        ax.scatter(
            selected_points[:, 0],
            selected_points[:, 1],
            s=55,
            edgecolors="black",
            linewidths=0.8,
            zorder=12,
            label=f"{name} constraints",
            **style,
        )

    return fig, ax


def plot_mapping_with_cp_overlay(
    field: FieldOfInterest,
    control_points: np.ndarray | Sequence[np.ndarray],
    *,
    displacements: np.ndarray | Sequence[np.ndarray] | None = None,
    sync_points: np.ndarray
    | Sequence[np.ndarray]
    | Sequence[tuple[int, int, np.ndarray]]
    | None = None,
    constraint_status: Mapping[
        str, np.ndarray | Sequence[int] | Sequence[Sequence[float]]
    ]
    | None = None,
    fig=None,
    ax=None,
) -> tuple:
    if fig is None or ax is None:
        fig, ax = visualisations.subplots(1, 1)

    visualisations.pcolormesh(fig, ax, field)
    overlay_control_points(
        fig,
        ax,
        control_points=control_points,
        displacements=displacements,
    )

    if sync_points is not None:
        overlay_sync_points(fig, ax, sync_points)

    if constraint_status is not None:
        overlay_constraint_status(
            fig,
            ax,
            control_points=control_points,
            free=constraint_status.get("free"),
            partial=constraint_status.get("partial"),
            fixed=constraint_status.get("fixed"),
        )

    return fig, ax
