import logging
import time
from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.colors import CenteredNorm, Normalize
from matplotlib.patches import Rectangle
from typing_extensions import deprecated

from phdtruel import printable_path, visualisations
from phdtruel.fields import DynamicOfInterest, FieldOfInterest
from phdtruel.fields.meshes import DomainBounds
from phdtruel.fields.parameters import ParameterSet

logger = logging.getLogger(__name__)


def overlay_domain_bounds_rectangle(
    ax: plt.Axes,
    bounds: DomainBounds,
    *,
    edgecolor: str = "k",
    linewidth: float = 1.5,
    linestyle: str = "-",
) -> Rectangle:
    """Draw a rectangle outlining ``bounds`` on a field plot axis."""
    xmin = float(bounds.minima[0])
    ymin = float(bounds.minima[1])
    width = float(bounds.maxima[0] - bounds.minima[0])
    height = float(bounds.maxima[1] - bounds.minima[1])
    rect = Rectangle(
        (xmin, ymin),
        width,
        height,
        fill=False,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=10,
    )
    ax.add_patch(rect)
    return rect


def plot_naca_coordinates(ax, naca_coordinates, color="k", linewidth=0.5, **kwargs):
    ax.plot(
        naca_coordinates[0, :, 0],
        naca_coordinates[0, :, 1],
        color=color,
        linewidth=linewidth,
        **kwargs,
    )
    return ax


def centered_norm(
    field: FieldOfInterest,
    on: Literal["mean", "median", "zero"] = "mean",
    truncation: DomainBounds | None = None,
):
    if truncation is not None:
        field = field.truncate(truncation)

    if on == "mean":
        vcenter = np.mean(field.values)
    elif on == "median":
        vcenter = np.median(field.values)
    elif on == "zero":
        vcenter = 0
    else:
        raise ValueError(f"Unknown on: {on}")
    norm = matplotlib.colors.CenteredNorm(vcenter=vcenter)
    return norm


def get_norm_for_fields(
    fields: list[FieldOfInterest],
    norm_type: Literal[
        "minmax", "centered_mean", "centered_median", "centered_zero"
    ] = "minmax",
    truncation: DomainBounds | None = None,
) -> Normalize | CenteredNorm:
    """Create a shared normalization for multiple fields.

    Args:
        fields: List of FieldOfInterest objects to normalize together
        norm_type: Type of normalization:
            - "minmax": Linear scaling from global min to max
            - "centered_mean": Center colormap at mean value
            - "centered_median": Center colormap at median value
            - "centered_zero": Center colormap at zero
        truncation: Optional DomainBounds to truncate all fields before computing norm

    Returns:
        A matplotlib Normalize or CenteredNorm object that can be passed to pcolormesh
    """
    if not fields:
        raise ValueError("At least one field is required")

    # Collect all values from all fields
    all_values = []
    for field in fields:
        if truncation is not None:
            field = field.truncate(truncation)
        all_values.append(field.values)

    all_values = np.concatenate(all_values)

    if norm_type == "minmax":
        vmin = np.min(all_values)
        vmax = np.max(all_values)
        return Normalize(vmin=vmin, vmax=vmax)
    elif norm_type == "centered_mean":
        vcenter = np.mean(all_values)
        return CenteredNorm(vcenter=vcenter)
    elif norm_type == "centered_median":
        vcenter = np.median(all_values)
        return CenteredNorm(vcenter=vcenter)
    elif norm_type == "centered_zero":
        return CenteredNorm(vcenter=0.0)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


@deprecated("Not used in a while? Possible duplicates")
def plot_fields(
    fig: plt.Figure,
    axes: list[plt.Axes],
    fields: dict[ParameterSet, FieldOfInterest],
    norm: Normalize
    | Literal[
        "centered_mean", "centered_median", "centered_zero", "minmax"
    ] = "centered_mean",
    **kwargs,
) -> tuple[plt.Figure, list[plt.Axes]]:
    if len(axes) < len(fields):
        raise ValueError("Number of fields does not match with number of axes")

    for k, (mu, sol) in enumerate(fields.items()):
        ax = axes[k]
        # ax.plot(
        #     mean_dynamics[mu].naca_coordinates[0, :, 0],
        #     mean_dynamics[mu].naca_coordinates[0, :, 1],
        #     "k",
        #     linewidth=0.5,
        # )
        ax.set_title(f"Mean field for {mu}")

        visualisations.contourf(fig, ax, sol, norm=norm, **kwargs)
        ax.grid()
    fig.tight_layout()
    return fig, axes


def animate_dynamics(
    *dynamics: DynamicOfInterest,
    sup_title: str = "",
    filepath: Path | str | None = None,
    dpi: int = 200,
    levels: int = 50,
    fps: int = 10,
    use_pcolormesh: bool = True,
    original_bounds: DomainBounds | None = None,
):
    """Create an animation of dynamic fields over time.

    Args:
        *dynamics: One or more DynamicOfInterest objects to animate
        sup_title: Title for the animation
        filepath: Path to save the animation, if None uses default location
        dpi: DPI for the saved animation
        levels: Number of contour levels
        fps: Frames per second for the animation
        use_pcolormesh: Whether to use pcolormesh (faster) instead of contourf (prettier)
        original_bounds: If set, draw a rectangle marking the original domain on each panel
    """
    start_time = time.time()
    fig, axes = visualisations.subplots(1, len(dynamics))
    if len(dynamics) == 1:
        axes = [axes]

    reshaped_fields = []
    for dynamic in dynamics:
        dynamic_fields = []
        for field in dynamic:
            reshaped_value = field.values.reshape(
                field.mesh.shape[1], field.mesh.shape[0], order="F"
            )
            dynamic_fields.append((field, reshaped_value))
        reshaped_fields.append(dynamic_fields)

    all_values = []
    for dynamic in dynamics:
        for field in dynamic:
            all_values.append(field.values)
    all_values = np.concatenate(all_values)
    vmin, vmax = np.quantile(all_values, 0.05), np.quantile(all_values, 0.95)
    norm = Normalize(vmin, vmax)

    plots = []
    colorbars = []

    for i, dynamic_fields in enumerate(reshaped_fields):
        field, reshaped_value = dynamic_fields[0]
        dynamic_name = (
            dynamics[i].name
            if hasattr(dynamics[i], "name") and dynamics[i].name
            else f"Dynamic {i + 1}"
        )

        ax = axes[i]
        ax.set_aspect("equal")
        ax.set(xlabel="", ylabel="")
        ax.set_title(f"{dynamic_name}: {field.parameter}")

        if use_pcolormesh:
            p = ax.pcolormesh(
                field.mesh.x, field.mesh.y, reshaped_value, norm=norm, shading="auto"
            )
            cbar = fig.colorbar(p, ax=ax)
            colorbars.append(cbar)
        else:
            p = visualisations.contourf(fig, ax, field, norm=norm, levels=levels)
            colorbars.append(None)

        plots.append(p)
        if original_bounds is not None:
            overlay_domain_bounds_rectangle(ax, original_bounds)
    fig.tight_layout()

    def update(frame):
        # Get current timestep
        timestep = (
            dynamics[0].timesteps[frame]
            if hasattr(dynamics[0], "timesteps") and frame < len(dynamics[0].timesteps)
            else frame
        )

        # Update suptitle with timestep information
        current_title = (
            f"{sup_title} - Time: {timestep}" if sup_title else f"Time: {timestep}"
        )
        fig.suptitle(current_title)

        for i, dynamic_fields in enumerate(reshaped_fields):
            field, reshaped_value = dynamic_fields[frame]
            ax = axes[i]
            dynamic_name = (
                dynamics[i].name
                if hasattr(dynamics[i], "name") and dynamics[i].name
                else f"Dynamic {i + 1}"
            )

            if use_pcolormesh:
                plots[i].set_array(reshaped_value.ravel())
                ax.set_title(f"{dynamic_name}: {field.parameter}")
            else:
                ax.clear()
                ax.set_aspect("equal")
                ax.set(xlabel="", ylabel="")
                ax.set_title(f"{dynamic_name}: {field.parameter}")

                p = ax.contourf(
                    field.mesh.x, field.mesh.y, reshaped_value, norm=norm, levels=levels
                )
                plots[i] = p
                if original_bounds is not None:
                    overlay_domain_bounds_rectangle(ax, original_bounds)

        return plots

    ani = FuncAnimation(fig, update, frames=len(dynamics[0]))

    if filepath is not None:
        if isinstance(filepath, str):
            filepath = visualisations.get_plot_subfolder() / filepath
        ani.save(filepath, writer=FFMpegWriter(fps=fps), dpi=dpi)
        logger.info(f"Animation saved to {printable_path(filepath)}")
    else:
        folder = visualisations.get_plot_subfolder()
        save_path = folder / "nascar_video.mp4"
        ani.save(
            save_path,
            writer=FFMpegWriter(fps=fps),
            dpi=dpi,
        )
        logger.info(f"Animation saved to {printable_path(save_path)}")

    end_time = time.time()
    execution_time = end_time - start_time
    if execution_time > 10:
        logger.info(f"Animation creation took {execution_time:.2f} seconds")

    return None


def plot_f0ref1(fig, axes, fields: list[FieldOfInterest]) -> CenteredNorm:
    f0, f_ref, f1 = fields
    ref_norm = CenteredNorm(
        np.mean(f_ref.values), f_ref.values.max() - np.mean(f_ref.values)
    )
    for ax, field, name in zip(axes, [f0, f_ref, f1], ["$u_0$", "$u_{ref}$", "$u_1$"]):
        visualisations.pcolormesh(fig, ax, field, norm=ref_norm)
        ax.set_title(name)
    return ref_norm
