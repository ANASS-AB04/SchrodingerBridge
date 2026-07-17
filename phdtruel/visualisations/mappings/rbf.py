"""Plotting helpers shared by the RBF mapping benchmark scripts.

These were previously duplicated near-verbatim across
``scripts/minimized_mappings/rbf_in_square.py``,
``rbf_free_cp_in_square.py``, ``rbf_in_D_domain.py`` and
``rbf_free_cp_in_D_domain.py``.
"""

import logging

import jax.numpy as jnp
import numpy as np

from phdtruel import visualisations
from phdtruel.fields.field_of_interest import FieldOfInterest
from phdtruel.fields.meshes import RegularGrid
from phdtruel.mappings.cost_functional.rbf.cost_fn import _jacobian_det_2d
from phdtruel.mappings.functional_solver import SolveResult
from phdtruel.visualisations.mappings.cdi_animations import (
    _field_color_norm,
    _save_sweep_animation,
    _zero_to_max_field_norm,
)
from phdtruel.visualisations.mappings.mapping_primitives import (
    get_jacobian_cmap_and_norm,
    plot_deformed_checkerboard,
)
from phdtruel.visualisations.minimisation.traces import plot_solver_trace_panels

logger = logging.getLogger(__name__)


def plot_deformed_checkerboards(
    mesh: RegularGrid,
    inverse_mapping_t,
    inverse_mapping_w,
    *,
    save_name: str = "rbf_checkerboards.png",
) -> None:
    """Plot checkerboard patterns deformed by RBF mappings W and T."""
    if not isinstance(mesh, RegularGrid):
        raise ValueError("Mesh must be a RegularGrid for checkerboard generation.")

    fig, axes = visualisations.subplots(1, 2, figsize=(15, 6))
    plot_deformed_checkerboard(
        mesh, inverse_mapping_t, fig=fig, ax=axes[0], title=r"$\mathrm{Damier}(W)$"
    )
    plot_deformed_checkerboard(
        mesh, inverse_mapping_w, fig=fig, ax=axes[1], title=r"$\mathrm{Damier}(T)$"
    )
    fig.tight_layout()
    visualisations.savefig(fig, save_name)


def plot_jacobian_determinants(
    mesh,
    displacements_w_arr: np.ndarray,
    displacements_t_arr: np.ndarray,
    rbf_kwargs: dict,
    *,
    centers_arr: np.ndarray | None = None,
    centers_w_arr: np.ndarray | None = None,
    centers_t_arr: np.ndarray | None = None,
    include_range_in_title: bool = True,
    save_name: str = "rbf_jacobians.png",
) -> None:
    """Plot Jacobian determinants for RBF mappings W and T.

    Works for both fixed-center RBF mappings (pass `centers_arr`, shared by
    both W and T) and free-control-point RBF mappings (pass `centers_w_arr`
    and `centers_t_arr` separately).
    """
    if centers_arr is not None:
        centers_w_arr = centers_arr
        centers_t_arr = centers_arr
    if centers_w_arr is None or centers_t_arr is None:
        raise ValueError(
            "Provide either `centers_arr` (fixed centers) or both "
            "`centers_w_arr` and `centers_t_arr` (free centers)."
        )

    points_jax = jnp.array(mesh.points)
    jacobian_w_values = np.asarray(
        _jacobian_det_2d(
            points_jax,
            jnp.array(centers_w_arr),
            jnp.array(displacements_w_arr),
            rbf_kwargs,
        )
    )
    jacobian_t_values = np.asarray(
        _jacobian_det_2d(
            points_jax,
            jnp.array(centers_t_arr),
            jnp.array(displacements_t_arr),
            rbf_kwargs,
        )
    )

    jacobian_w = FieldOfInterest(None, mesh, jacobian_w_values, interpolated=False)
    jacobian_t = FieldOfInterest(None, mesh, jacobian_t_values, interpolated=False)

    fig, axes = visualisations.subplots(1, 2, figsize=(15, 6))

    jac_norm_t, cmap_t = get_jacobian_cmap_and_norm(jacobian_t)
    visualisations.pcolormesh(fig, axes[0], jacobian_t, cmap=cmap_t, norm=jac_norm_t)
    title_t = "Jacobian of T"
    if include_range_in_title:
        title_t += f": [{jacobian_t.values.min():.2e}, {jacobian_t.values.max():.2e}]"
    axes[0].set_title(title_t)

    jac_norm_w, cmap_w = get_jacobian_cmap_and_norm(jacobian_w)
    visualisations.pcolormesh(fig, axes[1], jacobian_w, cmap=cmap_w, norm=jac_norm_w)
    title_w = "Jacobian of W"
    if include_range_in_title:
        title_w += f": [{jacobian_w.values.min():.2e}, {jacobian_w.values.max():.2e}]"
    axes[1].set_title(title_w)

    fig.tight_layout()
    visualisations.savefig(fig, save_name)


def plot_results(
    field_0: FieldOfInterest,
    field_1: FieldOfInterest,
    inverse_mapping_w,
    inverse_mapping_t,
    *,
    save_name: str = "rbf_aligned_fields.png",
) -> None:
    """Plot optimization results: original fields, aligned fields, differences."""
    fig, axes = visualisations.subplots(2, 3, figsize=(15, 10))
    mesh = field_0.mesh
    points = mesh.points

    field_0.name = "$u_0$ (original)"
    visualisations.pcolormesh(fig, axes[0, 0], field_0)

    field_1.name = "$u_1$ (original)"
    visualisations.pcolormesh(fig, axes[0, 1], field_1)

    diff_original = FieldOfInterest(
        parameter=None,
        mesh=mesh,
        values=np.abs(field_0.values - field_1.values),
        name="$|u_0 - u_1|$ (L2={:.4f})".format(
            np.linalg.norm(np.abs(field_0.values - field_1.values))
        ),
    )
    visualisations.pcolormesh(fig, axes[0, 2], diff_original)

    aligned_0 = field_0.eval(inverse_mapping_t(points), as_foi=True)
    aligned_0.name = "$u_0 \\circ W$"
    visualisations.pcolormesh(fig, axes[1, 0], aligned_0)

    aligned_1 = field_1.eval(inverse_mapping_w(points), as_foi=True)
    aligned_1.name = "$u_1 \\circ T$"
    visualisations.pcolormesh(fig, axes[1, 1], aligned_1)

    l2_norm = np.linalg.norm(np.abs(aligned_0.values - field_1.values))
    diff_aligned_0 = FieldOfInterest(
        parameter=None,
        mesh=mesh,
        values=np.abs(aligned_0.values - field_1.values),
        name="$|u_0 \\circ W - u_1|$ (L2={:.4f})".format(l2_norm),
    )
    visualisations.pcolormesh(fig, axes[1, 2], diff_aligned_0)

    fig.tight_layout()
    visualisations.savefig(fig, save_name)
    logger.info("Saved aligned fields plot")


def plot_cost_evolution(
    result: SolveResult,
    *,
    save_name: str = "rbf_cost_evolution.png",
) -> None:
    """Plot the RBF solver cost evolution with all components."""
    fig, _ = plot_solver_trace_panels(traces=result.history)
    fig.tight_layout()
    visualisations.savefig(fig, save_name)


def clone_for_plot(field: FieldOfInterest) -> FieldOfInterest:
    """Return a non-interpolated copy with 1-D values, for PointCloud plotting."""
    return FieldOfInterest(
        field.parameter,
        field.mesh,
        np.asarray(field.values).ravel(),
        name=field.name,
        fill_value=field._fill_value,
        interpolated=False,
    )


def save_pointcloud_mapping_animations(
    field_0: FieldOfInterest,
    field_1: FieldOfInterest,
    inverse_mapping_t,
    inverse_mapping_w,
) -> None:
    """Save mapping/CDI animations; PointCloud fields need 1-D values for tripcolor."""
    from phdtruel.interpolations.cdi import convex_displacement_interpolation

    output_dir = visualisations.get_plot_subfolder()
    points = field_0.mesh.points
    norm = _field_color_norm(field_0, field_1)

    def draw_u0(fig, ax, s: float) -> None:
        mapped = field_0.eval_mapped(
            inverse_mapping_t,
            points,
            with_s=s,
            as_foi=True,
            with_name=rf"$u_0 \circ T^{{-1}}(s={s:.2f})$",
        )
        visualisations.pcolormesh(
            fig, ax, clone_for_plot(mapped), norm=norm, title=mapped.name
        )

    def draw_u1(fig, ax, s: float) -> None:
        mapped = field_1.eval_mapped(
            inverse_mapping_w,
            points,
            with_s=s,
            as_foi=True,
            with_name=rf"$u_1 \circ W^{{-1}}(s={s:.2f})$",
        )
        visualisations.pcolormesh(
            fig, ax, clone_for_plot(mapped), norm=norm, title=mapped.name
        )

    def draw_cdi(fig, ax, s: float) -> None:
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
            fig, ax, clone_for_plot(field), norm=norm, title=field.name
        )

    linear_norm = _zero_to_max_field_norm(field_0, field_1)

    def draw_linear(fig, ax, s: float) -> None:
        values = np.ravel((1.0 - s) * field_0.values + s * field_1.values)
        field = FieldOfInterest(
            parameter=None,
            mesh=field_0.mesh,
            values=values,
            name=rf"Linear interpolation ($s={s:.2f}$)",
            interpolated=False,
        )
        visualisations.pcolormesh(fig, ax, field, norm=linear_norm, title=field.name)

    for path, desc, draw_fn in (
        (output_dir / "u0_mapped_to_u1.mp4", "u₀ mapped toward u₁", draw_u0),
        (output_dir / "u1_mapped_to_u0.mp4", "u₁ mapped toward u₀", draw_u1),
        (output_dir / "cdi_optimized.mp4", "optimized CDI", draw_cdi),
    ):
        _save_sweep_animation(path, desc=desc, draw_frame=draw_fn)

    _save_sweep_animation(
        output_dir / "linear_interpolation.mp4",
        desc="linear interpolation",
        draw_frame=draw_linear,
    )
