"""Animations for optimized two-field mappings and CDI along s."""

import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotloom import Loom
from matplotlib.colors import Normalize

from phdtruel import printable_path, visualisations
from phdtruel.fields.field_of_interest import FieldOfInterest
from phdtruel.interpolations.cdi import convex_displacement_interpolation
from phdtruel.mappings.mappings import Mapping

logger = logging.getLogger(__name__)

DEFAULT_N_FRAMES = 41
DEFAULT_FPS = 10
DEFAULT_DPI = 150
DEFAULT_FIGSIZE = (8.0, 6.5)


def _field_color_norm(
    field_0: FieldOfInterest,
    field_1: FieldOfInterest,
) -> Normalize:
    """Shared color scale for mapping/CDI animations."""
    values = np.concatenate([field_0.values.ravel(), field_1.values.ravel()])
    vmin, vmax = np.quantile(values, [0.05, 0.95])
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def _zero_to_max_field_norm(
    field_0: FieldOfInterest,
    field_1: FieldOfInterest,
) -> Normalize:
    """Color scale from 0 to the maximum value in u₀ and u₁."""
    values = np.concatenate([field_0.values.ravel(), field_1.values.ravel()])
    vmax = float(np.max(values))
    if np.isclose(vmax, 0.0):
        vmax = 1.0
    return Normalize(vmin=0.0, vmax=vmax)


def _save_sweep_animation(
    output_path: Path,
    *,
    desc: str,
    draw_frame: Callable[[plt.Figure, plt.Axes, float], None],
    n_frames: int = DEFAULT_N_FRAMES,
    fps: int = DEFAULT_FPS,
    dpi: int = DEFAULT_DPI,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    s_values = np.linspace(0.0, 1.0, n_frames)

    with Loom(output_path, fps=fps, overwrite=True) as loom:
        for frame_idx, s in enumerate(s_values):
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            try:
                draw_frame(fig, ax, float(s))
                fig.tight_layout()
                loom.save_frame(fig, frame_idx)
            finally:
                plt.close(fig)

    logger.info("Saved %s animation to %s", desc, printable_path(output_path))


def save_linear_interpolation_animation(
    field_0: FieldOfInterest,
    field_1: FieldOfInterest,
    *,
    output_dir: Path | None = None,
    output_name: str = "linear_interpolation.mp4",
    n_frames: int = DEFAULT_N_FRAMES,
    fps: int = DEFAULT_FPS,
    dpi: int = DEFAULT_DPI,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    cmap: str | None = None,
    norm: Normalize | None = None,
) -> Path:
    """Save an MP4 of the linear blend (1 − s) u₀ + s u₁ for s in [0, 1]."""
    if output_dir is None:
        output_dir = visualisations.get_plot_subfolder()

    if norm is None:
        norm = _zero_to_max_field_norm(field_0, field_1)

    def draw_linear(fig: plt.Figure, ax: plt.Axes, s: float) -> None:
        values = np.ravel((1.0 - s) * field_0.values + s * field_1.values)
        field = FieldOfInterest(
            parameter=None,
            mesh=field_0.mesh,
            values=values,
            name=rf"Linear interpolation ($s={s:.2f}$)",
        )
        pcolormesh_kwargs: dict[str, object] = {"norm": norm, "title": field.name}
        if cmap is not None:
            pcolormesh_kwargs["cmap"] = cmap
        visualisations.pcolormesh(fig, ax, field, **pcolormesh_kwargs)

    output_path = output_dir / output_name
    _save_sweep_animation(
        output_path,
        desc="linear interpolation",
        draw_frame=draw_linear,
        n_frames=n_frames,
        fps=fps,
        dpi=dpi,
        figsize=figsize,
    )
    return output_path


def save_optimized_mapping_animations(
    field_0: FieldOfInterest,
    field_1: FieldOfInterest,
    inverse_mapping_t: Mapping,
    inverse_mapping_w: Mapping,
    *,
    output_dir: Path | None = None,
    n_frames: int = DEFAULT_N_FRAMES,
    fps: int = DEFAULT_FPS,
    dpi: int = DEFAULT_DPI,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    cmap: str | None = None,
    norm: Normalize | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Save four MP4 animations sweeping the interpolation parameter s in [0, 1].

    Animations:
    - ``u0_mapped_to_u1.mp4``: u₀ composed with T⁻¹ at scale s (identity → aligned to u₁).
    - ``u1_mapped_to_u0.mp4``: u₁ composed with W⁻¹ at scale s (identity → aligned to u₀).
    - ``cdi_optimized.mp4``: convex displacement interpolation with the optimized inverses.
    - ``linear_interpolation.mp4``: linear blend (1 − s) u₀ + s u₁ on the mesh.

    Args:
        field_0: Source field u₀.
        field_1: Target field u₁.
        inverse_mapping_t: Inverse of mapping T (u₀ → u₁).
        inverse_mapping_w: Inverse of mapping W (u₁ → u₀).
        output_dir: Folder for MP4 files; defaults to the active plot subfolder.
        n_frames: Number of frames from s=0 to s=1.
        fps: Frames per second for the output video.
        dpi: Resolution passed to Loom when saving frames.
        figsize: Matplotlib figure size per frame.

    Returns:
        Paths to the four saved animation files.
    """
    if output_dir is None:
        output_dir = visualisations.get_plot_subfolder()

    points = field_0.mesh.points
    if norm is None:
        norm = _field_color_norm(field_0, field_1)

    def _pcolormesh_kwargs(**extra: object) -> dict[str, object]:
        kwargs: dict[str, object] = {"norm": norm, **extra}
        if cmap is not None:
            kwargs["cmap"] = cmap
        return kwargs

    def draw_u0(fig: plt.Figure, ax: plt.Axes, s: float) -> None:
        mapped = field_0.eval_mapped(
            inverse_mapping_t,
            points,
            with_s=s,
            as_foi=True,
            with_name=rf"$u_0 \circ T^{{-1}}(s={s:.2f})$",
        )
        visualisations.pcolormesh(
            fig, ax, mapped, **_pcolormesh_kwargs(title=mapped.name)
        )

    def draw_u1(fig: plt.Figure, ax: plt.Axes, s: float) -> None:
        mapped = field_1.eval_mapped(
            inverse_mapping_w,
            points,
            with_s=s,
            as_foi=True,
            with_name=rf"$u_1 \circ W^{{-1}}(s={s:.2f})$",
        )
        visualisations.pcolormesh(
            fig, ax, mapped, **_pcolormesh_kwargs(title=mapped.name)
        )

    def draw_cdi(fig: plt.Figure, ax: plt.Axes, s: float) -> None:
        field = convex_displacement_interpolation(
            points,
            field_0,
            field_1,
            inverse_mapping_w,
            inverse_mapping_t,
            s,
            as_foi=True,
        )
        field.name = rf"CDI ($s={s:.2f}$)"
        visualisations.pcolormesh(
            fig, ax, field, **_pcolormesh_kwargs(title=field.name)
        )

    paths = (
        output_dir / "u0_mapped_to_u1.mp4",
        output_dir / "u1_mapped_to_u0.mp4",
        output_dir / "cdi_optimized.mp4",
    )
    descriptions = (
        "u₀ mapped toward u₁",
        "u₁ mapped toward u₀",
        "optimized CDI",
    )
    draw_fns = (draw_u0, draw_u1, draw_cdi)

    for path, desc, draw_fn in zip(paths, descriptions, draw_fns, strict=True):
        _save_sweep_animation(
            path,
            desc=desc,
            draw_frame=draw_fn,
            n_frames=n_frames,
            fps=fps,
            dpi=dpi,
            figsize=figsize,
        )

    linear_path = save_linear_interpolation_animation(
        field_0,
        field_1,
        output_dir=output_dir,
        n_frames=n_frames,
        fps=fps,
        dpi=dpi,
        figsize=figsize,
        cmap=cmap,
        norm=norm,
    )

    return (*paths, linear_path)
