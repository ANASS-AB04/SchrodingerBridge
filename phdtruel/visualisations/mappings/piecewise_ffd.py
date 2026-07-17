import logging

import numpy as np
from matplotlib.colors import CenteredNorm, Normalize
from matplotlib.patches import Rectangle

from phdtruel import visualisations
from phdtruel.fields.field_of_interest import FieldOfInterest
from phdtruel.fields.meshes import RegularGrid
from phdtruel.mappings.functional_solver import SolveResult
from phdtruel.visualisations.fields import get_norm_for_fields
from phdtruel.visualisations.mappings.mapping_primitives import (
    plot_deformed_checkerboard,
    plot_piecewise_jacobian_pair,
)
from phdtruel.visualisations.minimisation.traces import plot_solver_trace_panels

logger = logging.getLogger(__name__)


def get_region_control_points(
    region_meshes: list[RegularGrid],
    all_n_cp: list[tuple[int, int]],
) -> list[np.ndarray]:
    """Compute control points for each region mesh.

    Args:
        region_meshes: Piecewise region meshes.
        all_n_cp: Number of control points ``(n_cp_x, n_cp_y)`` per region.

    Returns:
        A list of arrays with shape ``(n_points, 2)``.
    """
    from phdtruel.mappings.ffd import get_control_points

    control_points: list[np.ndarray] = []
    for region_mesh, (n_cp_x, n_cp_y) in zip(region_meshes, all_n_cp):
        bounds = region_mesh.get_bounds()
        cp = np.asarray(
            get_control_points(
                box_origin=np.array(bounds.minima),
                box_length=np.array(bounds.lengths),
                n_control_points=np.array([n_cp_x, n_cp_y]),
            )
        ).reshape(-1, 2)
        control_points.append(cp)
    return control_points


def draw_region_on_axis(
    ax,
    mesh: RegularGrid,
    region_index: int,
    color: str | None = None,
    fill: bool = False,
    alpha: float = 0.3,
    label: str | None = None,
):
    """Draw a region rectangle on a given axis.

    Args:
        ax: Matplotlib axis to draw on.
        mesh: RegularGrid defining the region.
        region_index: Index of the region (used for default coloring).
        color: Color of the rectangle edge (default: cycles through colors).
        fill: Whether to fill the rectangle.
        alpha: Alpha value for the fill.
        label: Label for the region (default: "Region {region_index}").
    """
    default_colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]
    if color is None:
        color = default_colors[region_index % len(default_colors)]
    if label is None:
        label = f"Region {region_index}"

    bounds = mesh.get_bounds()
    width = bounds.maxima[0] - bounds.minima[0]
    height = bounds.maxima[1] - bounds.minima[1]
    rect = Rectangle(
        (bounds.minima[0], bounds.minima[1]),
        width,
        height,
        fill=fill,
        facecolor=color if fill else None,
        alpha=alpha if fill else 1.0,
        edgecolor=color,
        linewidth=2,
        linestyle="--" if not fill else "-",
        label=label,
    )
    ax.add_patch(rect)


def draw_control_points_on_axes(
    axes: list,
    region_meshes: list[RegularGrid],
    all_n_cp: list[tuple[int, int]],
):
    """Draw control points for each region on the provided axes.

    Args:
        axes: List of matplotlib axes to draw control points on.
        region_meshes: List of region meshes.
        all_n_cp: List of (n_cp_x, n_cp_y) tuples for each region.
    """
    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]
    markers = ["o", "s", "^", "v", "D", "p", "*", "h", "X", "P"]

    region_control_points = get_region_control_points(region_meshes, all_n_cp)

    for i, ctrl_pts_flat in enumerate(region_control_points):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # Plot on all provided axes
        for ax in axes:
            ax.scatter(
                ctrl_pts_flat[:, 0],
                ctrl_pts_flat[:, 1],
                s=30,
                c=color,
                marker=marker,
                edgecolors="black",
                linewidths=0.5,
                alpha=0.8,
                zorder=10,
            )


def plot_fields_and_regions(
    fields: list[FieldOfInterest],
    region_meshes: list[RegularGrid],
    all_n_cp: list[tuple[int, int]] | None = None,
    save_path: str | None = None,
    axsize: tuple[float, float] | None = None,
    norm: Normalize | CenteredNorm | None = None,
    cmap: str | None = None,
):
    """Plot the fields and region boundaries with optional control points.

    Args:
        fields: List of fields to plot.
        region_meshes: List of region meshes.
        all_n_cp: Optional list of (n_cp_x, n_cp_y) per region to draw control points.
        save_path: Optional path to save the figure.
        axsize: Optional per-axis (width, height) for the figure layout.
        norm: Shared color normalization; defaults to min–max over all ``fields``.
        cmap: Optional colormap passed to ``pcolormesh``.
    """
    if norm is None:
        norm = get_norm_for_fields(fields)

    n_fields = len(fields)
    fig, axes = visualisations.subplots(1, n_fields, axsize=axsize)

    # Ensure axes is always a list
    if n_fields == 1:
        axes = [axes]

    # Plot each field with region boundaries
    for field_idx, field in enumerate(fields):
        pcolormesh_kwargs: dict[str, object] = {"norm": norm}
        if cmap is not None:
            pcolormesh_kwargs["cmap"] = cmap
        visualisations.pcolormesh(fig, axes[field_idx], field, **pcolormesh_kwargs)

        # Draw region boundaries
        for i, mesh in enumerate(region_meshes):
            draw_region_on_axis(axes[field_idx], mesh, i, fill=False)
        axes[field_idx].legend()

    # Draw control points if provided
    if all_n_cp is not None:
        draw_control_points_on_axes(axes, region_meshes, all_n_cp)

    fig.tight_layout()
    visualisations.savefig(fig, save_path or "piecewise_ffd_setup.png")
    logger.info("Saved piecewise FFD setup plot")


def plot_deformed_checkerboards(
    mesh: RegularGrid,
    inverse_mapping_t,
    inverse_mapping_w,
    *,
    figsize: tuple[float, float] = (15, 6),
    save_name: str = "piecewise_ffd_checkerboards.png",
) -> None:
    """Plot checkerboard patterns deformed by piecewise FFD mappings W and T."""
    if not isinstance(mesh, RegularGrid):
        raise ValueError("Mesh must be a RegularGrid for checkerboard generation.")

    fig, axes = visualisations.subplots(1, 2, figsize=figsize)
    plot_deformed_checkerboard(
        mesh, inverse_mapping_t, fig=fig, ax=axes[0], title=r"$\mathrm{Damier}(W)$"
    )
    plot_deformed_checkerboard(
        mesh, inverse_mapping_w, fig=fig, ax=axes[1], title=r"$\mathrm{Damier}(T)$"
    )
    fig.tight_layout()
    visualisations.savefig(fig, save_name)


def plot_jacobian_determinants(
    mesh: RegularGrid,
    displacements_w: tuple,
    displacements_t: tuple,
    all_ffd_params: tuple,
    *,
    figsize: tuple[float, float] = (15, 6),
    title_t: str = "Jacobian of T",
    title_w: str = "Jacobian of W",
    save_name: str = "piecewise_ffd_jacobians.png",
) -> None:
    """Plot Jacobian determinants for piecewise FFD mappings W and T."""
    plot_piecewise_jacobian_pair(
        mesh,
        displacements_t,
        displacements_w,
        all_ffd_params,
        figsize=figsize,
        title_t=title_t,
        title_w=title_w,
        save_path=save_name,
    )


def plot_cost_evolution(
    result: SolveResult,
    *,
    figsize: tuple[float, float] | None = None,
    save_name: str = "piecewise_ffd_cost_evolution.png",
) -> None:
    """Plot the piecewise FFD solver cost evolution with all components.

    Panel 1: total cost evolution (log scale). Panel 2: individual cost
    components over iterations.
    """
    fig, _ = plot_solver_trace_panels(traces=result.history)
    if figsize is not None:
        fig.set_size_inches(*figsize)
    fig.tight_layout()
    visualisations.savefig(fig, save_name)
    logger.info("Saved cost evolution plot with all components")
