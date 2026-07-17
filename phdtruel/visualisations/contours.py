import logging
from typing import Any, Literal, cast

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from phdtruel.fields import FieldOfInterest
from phdtruel.fields.meshes import DomainBounds, PointCloud, RegularGrid

logger = logging.getLogger(__name__)


def create_norm(
    values: np.ndarray | FieldOfInterest,
    norm: (
        mcolors.Normalize
        | Literal["centered_mean", "centered_median", "centered_zero", "minmax"]
        | None
    ) = None,
    norm_halfrange: float | None = None,
) -> mcolors.Normalize:
    """
    Create a matplotlib color normalization object from preset strings or pass through existing Normalize objects.

    Args:
        values: Array of values to normalize. Used to compute statistics for centering and scaling.
        norm: Normalization strategy. Can be:
            - "centered_zero": Center colormap at zero
            - "centered_mean": Center colormap at mean value
            - "centered_median": Center colormap at median value
            - "minmax": Linear scaling from min to max
            - matplotlib.colors.Normalize object (returned as-is)
            - None: Auto-selects "minmax" if all values have same sign, else "centered_zero"
        norm_halfrange: Half-range for centered normalizations (distance from center to edge).
            If None, uses max absolute deviation from center.

    Returns:
        matplotlib.colors.Normalize object ready for use with colormaps.

    Raises:
        ValueError: If norm is an unsupported string preset.
    """
    if isinstance(values, FieldOfInterest):
        values = values.values

    if norm is None:
        # Auto-select based on data sign
        if values.min() * values.max() > 0:  # Same sign
            norm = "minmax"
        else:
            norm = "centered_zero"

    if isinstance(norm, str):
        if norm == "centered_mean":
            return mcolors.CenteredNorm(
                vcenter=float(np.mean(values)), halfrange=norm_halfrange
            )
        elif norm == "centered_median":
            return mcolors.CenteredNorm(
                vcenter=float(np.median(values)), halfrange=norm_halfrange
            )
        elif norm == "centered_zero":
            return mcolors.CenteredNorm(vcenter=0.0, halfrange=norm_halfrange)
        elif norm == "minmax":
            return mcolors.Normalize(float(values.min()), float(values.max()))
        else:
            raise ValueError(f"Unsupported preset for norm: {norm}")
    else:
        # Already a Normalize object, return as-is
        return norm


def bounds_from_fields(field):
    l1 = field.mean() - field.min()
    l2 = field.max() - field.mean()

    if l1 > l2:
        vmin = field.mean() - l1
        vmax = field.mean() + l1
    else:
        vmin = field.mean() - l2
        vmax = field.mean() + l2
    return vmin, vmax


def contourf(
    fig: Figure,
    ax: Axes,
    field: FieldOfInterest,
    *,
    cbar_label: str = "",
    norm: (
        mcolors.Normalize
        | Literal["centered_mean", "centered_median", "centered_zero", "minmax"]
        | None
    ) = None,
    norm_halfrange: float | None = None,
    cmap=None,
    scalar_mappable=None,
    return_colorbar: bool = False,
    return_cax: bool = False,
    truncation: DomainBounds | None = None,
    plot_boundary: bool = False,
    boundary_kwargs: dict | None = None,
    **kwargs,
):
    """Create filled contour plot with optional boundary overlay.

    Args:
        fig: Matplotlib figure object.
        ax: Axes to plot on.
        field: Field of interest containing values and mesh data.
        cbar_label: Label for the colorbar.
        norm: Color normalization strategy (see create_norm for options).
        norm_halfrange: Half-range for centered normalizations.
        cmap: Colormap name or object.
        scalar_mappable: Pre-configured ScalarMappable.
        return_colorbar: If True, include colorbar in return tuple.
        return_cax: If True, include colorbar axes in return tuple.
        truncation: Optional domain bounds for field cropping before plotting.
        plot_boundary: If True, plot the mesh boundary as a line overlay.
        boundary_kwargs: Style options for boundary plot (e.g., {'color': 'k', 'linewidth': 2}).
        **kwargs: Additional arguments passed to tricontourf/pcolormesh.

    Returns:
        Contour object, or tuple of (Contour, [Colorbar], [ColorbarAxes]) if return flags are set.
    """
    if truncation is not None:
        field = field.truncate(truncation)

    ax.set_title(field.name)

    values = field.values
    mesh = field.mesh
    is_point_cloud = isinstance(mesh, PointCloud)

    if isinstance(mesh, RegularGrid):
        x = mesh.x
        y = mesh.y
    elif is_point_cloud:
        if mesh.dimension != 2:
            raise ValueError(
                f"PointCloud must be 2D for contourf, got dimension={mesh.dimension}"
            )
        x = mesh.points[:, 0]
        y = mesh.points[:, 1]
    else:
        raise NotImplementedError("Unsupported mesh type")

    if cmap is None:
        # cmap = "RdBu_r"
        if norm is None:
            norm = "centered_zero"

    norm_obj = create_norm(values, norm, norm_halfrange)

    if scalar_mappable is None:
        scalar_mappable = plt.cm.ScalarMappable(norm=norm_obj, cmap=cmap)

    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    try:
        if is_point_cloud:
            if getattr(mesh, "_regular_grids_for_pcolormesh", None) is not None:
                c = None
                for rg in mesh._regular_grids_for_pcolormesh:
                    logger.info("plot grid portion as contourf ")
                    values_rg = field(rg.points)
                    values_rg = values_rg.reshape((rg.shape[1], rg.shape[0]), order="F")
                    c = ax.contourf(
                        rg.x,
                        rg.y,
                        values_rg,
                        norm=scalar_mappable.norm,
                        cmap=scalar_mappable.cmap,
                        **kwargs,
                    )
            else:
                c = ax.tricontourf(
                    x,
                    y,
                    values.ravel(),
                    norm=scalar_mappable.norm,
                    cmap=scalar_mappable.cmap,
                    **kwargs,
                )
        else:
            values = values.reshape((mesh.shape[1], mesh.shape[0]), order="F")
            c = ax.contourf(
                x,
                y,
                values,
                norm=scalar_mappable.norm,
                cmap=scalar_mappable.cmap,
                **kwargs,
            )
    except ValueError as e:
        raise e

    # Plot boundary if requested
    if plot_boundary:
        default_boundary_kwargs = {"color": "k", "linewidth": 1.5, "alpha": 0.8}
        if boundary_kwargs is not None:
            default_boundary_kwargs.update(boundary_kwargs)
        try:
            boundary = mesh.boundary_points
            ax.plot(
                boundary[:, 0],
                boundary[:, 1],
                **cast(dict[str, Any], default_boundary_kwargs),
            )
        except NotImplementedError:
            pass  # Silently skip if boundary not supported (e.g., 3D+ grids)

    ax.grid(True)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(
        scalar_mappable,
        cax=cax,
        label=cbar_label,
        extend="both",
    )

    to_return: list[Any] = [c]
    if return_colorbar:
        to_return.append(cb)
    if return_cax:
        to_return.append(cax)

    if len(to_return) == 1:
        return to_return[0]
    else:
        return tuple(to_return)


def pcolormesh(
    fig: Figure,
    ax: Axes,
    field: FieldOfInterest,
    *,
    cbar_label: str = "",
    cbar_label_range: bool = False,
    cbar_label_range_fmt: str | None = None,
    title: str | None = None,
    norm: (
        mcolors.Normalize
        | Literal["centered_mean", "centered_median", "centered_zero", "minmax"]
        | None
    ) = None,
    norm_halfrange: float | None = None,
    cmap=None,
    scalar_mappable=None,
    return_colorbar: bool = False,
    return_cax: bool = False,
    truncation: DomainBounds | None = None,
    plot_boundary: bool = False,
    boundary_kwargs: dict | None = None,
    show_mesh: bool = False,
    **kwargs,
):
    """
    Create a pseudocolor plot with automatic normalization and colorbar.

    Args:
        fig: Matplotlib figure object.
        ax: Axes to plot on.
        field: Field of interest containing values and mesh data.
        cbar_label: Label for the colorbar.
        cbar_label_range: If True, annotate colorbar with displayed vmin/vmax (top/bottom).
        cbar_label_range_fmt: Python format spec (e.g. ".2e") for the range labels. If None, defaults to ".2e".
        title: Plot title. If None, uses field.name.
        norm: Color normalization strategy. Can be:
            - "centered_zero": Center colormap at zero (auto-default for mixed-sign data)
            - "centered_mean": Center colormap at mean value
            - "centered_median": Center colormap at median value
            - "minmax": Linear scaling from min to max (auto-default for same-sign data)
            - matplotlib.colors.Normalize object for custom normalization
            - None: Auto-selects based on data sign
        norm_halfrange: Half-range for centered normalizations (distance from center to edge).
        cmap: Colormap name or object. If None, auto-selected based on norm.
        scalar_mappable: Pre-configured ScalarMappable. If provided, overrides norm and cmap.
        return_colorbar: If True, include colorbar in return tuple.
        return_cax: If True, include colorbar axes in return tuple.
        truncation: Optional domain bounds for field cropping before plotting.
        plot_boundary: If True, plot the mesh boundary as a line overlay.
        boundary_kwargs: Style options for boundary plot (e.g., {'color': 'r', 'linewidth': 2}).
        show_mesh: If True, draw a thin wireframe on top of cells by setting
            pcolormesh/tripcolor edge styling. Explicit kwargs always take precedence.
        **kwargs: Additional arguments passed to ax.pcolormesh().

    Returns:
        QuadMesh object, or tuple of (QuadMesh, [Colorbar], [ColorbarAxes]) if return flags are set.

    Raises:
        NotImplementedError: If mesh type is not RegularGrid.
        ValueError: If invalid norm preset string is provided.
    """
    if truncation is not None:
        field = field.truncate(truncation)

    ax.set_title(title if title is not None else field.name)

    raw_values = field.values
    data_min = float(np.min(raw_values))
    data_max = float(np.max(raw_values))

    values = raw_values
    mesh = field.mesh
    mesh_is_point_cloud = isinstance(mesh, PointCloud)

    if isinstance(mesh, RegularGrid):
        x = mesh.x
        y = mesh.y
    elif mesh_is_point_cloud:
        if mesh.dimension != 2:
            raise ValueError(
                f"PointCloud must be 2D for pcolormesh, got dimension={mesh.dimension}"
            )
        x = mesh.points[:, 0]
        y = mesh.points[:, 1]
    else:
        raise NotImplementedError("Unsupported mesh type")

    norm_obj = create_norm(values, norm, norm_halfrange)

    if scalar_mappable is None:
        scalar_mappable = plt.cm.ScalarMappable(norm=norm_obj, cmap=cmap)

    # Ensure ScalarMappable has finite vmin/vmax for range labeling and colorbar.
    # (Some norms leave vmin/vmax unset until autoscale is called.)
    scalar_mappable.set_array(values)
    if (
        getattr(scalar_mappable.norm, "vmin", None) is None
        or getattr(scalar_mappable.norm, "vmax", None) is None
    ):
        scalar_mappable.autoscale()

    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.grid(True)

    plot_kwargs = dict(kwargs)
    if show_mesh:
        # ax.grid(False)
        if mesh_is_point_cloud:
            plot_kwargs.setdefault("edgecolors", (0.0, 0.0, 0.0, 0.35))
            plot_kwargs.setdefault("linewidth", 0.2)
        else:
            plot_kwargs.setdefault("edgecolors", (0.0, 0.0, 0.0, 0.35))
            plot_kwargs.setdefault("linewidth", 0.2)
            plot_kwargs.setdefault("shading", "nearest")
            plot_kwargs.setdefault("snap", False)

    try:
        if mesh_is_point_cloud:
            if mesh._regular_grids_for_pcolormesh is not None:
                for rg in mesh._regular_grids_for_pcolormesh:
                    logger.info("plot grid portion as pcolormesh ")

                    values_rg = field(rg.points)
                    values_rg = values_rg.reshape((rg.shape[1], rg.shape[0]), order="F")
                    c = ax.pcolormesh(
                        rg.x,
                        rg.y,
                        values_rg,
                        norm=scalar_mappable.norm,
                        cmap=scalar_mappable.cmap,
                        **plot_kwargs,
                    )
            else:
                logger.info("plot as point cloud ")
                # Default shading to 'flat' for PointCloud
                if "shading" not in plot_kwargs:
                    plot_kwargs["shading"] = "flat"
                c = ax.tripcolor(
                    x,
                    y,
                    values,
                    norm=scalar_mappable.norm,
                    cmap=scalar_mappable.cmap,
                    **plot_kwargs,
                )
        else:
            values = values.reshape((mesh.shape[1], mesh.shape[0]), order="F")
            c = ax.pcolormesh(
                x,
                y,
                values,
                norm=scalar_mappable.norm,
                cmap=scalar_mappable.cmap,
                **plot_kwargs,
            )
    except ValueError as e:
        raise e

    # Plot boundary if requested
    if plot_boundary:
        default_boundary_kwargs = {"color": "k", "linewidth": 1.5, "alpha": 0.8}
        if boundary_kwargs is not None:
            default_boundary_kwargs.update(boundary_kwargs)
        try:
            boundary = mesh.boundary_points
            # Close the boundary for plotting by appending the first point
            if len(boundary) > 0:
                boundary_closed = np.vstack([boundary, boundary[0:1]])
                ax.plot(
                    boundary_closed[:, 0],
                    boundary_closed[:, 1],
                    **cast(dict[str, Any], default_boundary_kwargs),
                    label="Boundary",
                )
        except NotImplementedError:
            pass  # Silently skip if boundary not supported (e.g., 3D+ grids)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(
        scalar_mappable,
        cax=cax,
        label=cbar_label,
        extend="both",
    )

    if cbar_label_range:
        fmt = cbar_label_range_fmt or ".2e"

        vmin_val = scalar_mappable.norm.vmin
        vmax_val = scalar_mappable.norm.vmax
        vmin = float(vmin_val) if vmin_val is not None else 0.0
        vmax = float(vmax_val) if vmax_val is not None else 1.0

        low_clipped = data_min < vmin
        high_clipped = data_max > vmax

        bottom_label = format(vmin, fmt) + (" (clipped)" if low_clipped else "")
        top_label = format(vmax, fmt) + (" (clipped)" if high_clipped else "")

        # Place the labels just to the right of the colorbar.
        x_text = 1.00
        cb.ax.text(
            x_text,
            -0.08,
            bottom_label,
            transform=cb.ax.transAxes,
            ha="left",
            va="bottom",
            fontsize="small",
            clip_on=False,
        )
        cb.ax.text(
            x_text,
            1.08,
            top_label,
            transform=cb.ax.transAxes,
            ha="left",
            va="top",
            fontsize="small",
            clip_on=False,
        )

    to_return: list[Any] = [c]
    if return_colorbar:
        to_return.append(cb)
    if return_cax:
        to_return.append(cax)

    if len(to_return) == 1:
        return to_return[0]
    else:
        return tuple(to_return)


def contour(
    fig: Figure,
    ax: Axes,
    field: FieldOfInterest,
    truncation: DomainBounds | None = None,
    plot_boundary: bool = False,
    boundary_kwargs: dict | None = None,
    **kwargs,
):
    """Create a contour line plot with optional boundary overlay.

    Args:
        fig: Matplotlib figure object.
        ax: Axes to plot on.
        field: Field of interest containing values and mesh data.
        truncation: Optional domain bounds for field cropping before plotting.
        plot_boundary: If True, plot the mesh boundary as a line overlay.
        boundary_kwargs: Style options for boundary plot (e.g., {'color': 'k', 'linewidth': 2}).
        **kwargs: Additional arguments passed to tricontour/contour.

    Returns:
        Contour object.
    """
    if truncation is not None:
        field = field.truncate(truncation)

    values = field.values
    mesh = field.mesh
    is_point_cloud = isinstance(mesh, PointCloud)

    if isinstance(mesh, RegularGrid):
        x = mesh.x
        y = mesh.y
    elif is_point_cloud:
        if mesh.dimension != 2:
            raise ValueError(
                f"PointCloud must be 2D for contour, got dimension={mesh.dimension}"
            )
        x = mesh.points[:, 0]
        y = mesh.points[:, 1]
    else:
        raise NotImplementedError("Unsupported mesh type")

    if "vmin" not in kwargs and "vmax" not in kwargs:
        vmin, vmax = bounds_from_fields(field=values)
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax

    if "colors" in kwargs and "cmap" not in kwargs:
        kwargs["cmap"] = None

    if "cmap" not in kwargs:
        kwargs["cmap"] = "RdBu_r"

    if "levels" not in kwargs:
        kwargs["levels"] = 7

    if "levels" in kwargs and isinstance(kwargs["levels"], int):
        kwargs["levels"] = np.linspace(values.min(), values.max(), kwargs["levels"])

    kwargs["levels"] = np.sort(kwargs["levels"])  # Contour levels must be increasing

    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    # Plot boundary if requested
    if plot_boundary:
        default_boundary_kwargs = {"color": "k", "linewidth": 1.5, "alpha": 0.8}
        if boundary_kwargs is not None:
            default_boundary_kwargs.update(boundary_kwargs)
        try:
            boundary = mesh.boundary_points
            # Close the boundary for plotting by appending the first point
            if len(boundary) > 0:
                boundary_closed = np.vstack([boundary, boundary[0:1]])
                ax.plot(
                    boundary_closed[:, 0],
                    boundary_closed[:, 1],
                    **cast(dict[str, Any], default_boundary_kwargs),
                )
        except NotImplementedError:
            pass  # Silently skip if boundary not supported (e.g., 3D+ grids)

    ax.grid(True)

    if is_point_cloud:
        if getattr(mesh, "_regular_grids_for_pcolormesh", None) is not None:
            c = None
            for rg in mesh._regular_grids_for_pcolormesh:
                logger.info("plot grid portion as contour ")
                values_rg = field(rg.points)
                values_rg = values_rg.reshape((rg.shape[1], rg.shape[0]), order="F")
                c = ax.contour(
                    rg.x,
                    rg.y,
                    values_rg,
                    **kwargs,
                )
            return c
        return ax.tricontour(x, y, values.ravel(), **kwargs)
    else:
        values = values.reshape((mesh.shape[1], mesh.shape[0]), order="F")
        return ax.contour(x, y, values, **kwargs)
