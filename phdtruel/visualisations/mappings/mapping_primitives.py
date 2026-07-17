from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

from phdtruel import visualisations
from phdtruel.fields.field_of_interest import FieldOfInterest
from phdtruel.fields.meshes import RegularGrid
from phdtruel.mappings.ffd import piecewise_jacobian_det_2d

JacobianSignCase = Literal["positive-only", "negative-only", "mixed-sign"]


def _build_checkerboard_field(
    mesh: RegularGrid,
    checkerboard_cell_size: int,
) -> FieldOfInterest:
    if checkerboard_cell_size <= 0:
        raise ValueError("checkerboard_cell_size must be a strictly positive integer.")

    indices = np.indices(mesh.shape)
    checkerboard_values = (
        (indices[0] // checkerboard_cell_size) + (indices[1] // checkerboard_cell_size)
    ) % 2
    return FieldOfInterest(
        parameter=None,
        mesh=mesh,
        values=checkerboard_values.astype(float),
    )


def plot_deformed_checkerboard(
    mesh: RegularGrid,
    inverse_mapping: Callable[[np.ndarray], np.ndarray],
    *,
    fig=None,
    ax=None,
    checkerboard_cell_size: int | None = None,
    cmap: str = "gray",
    title: str | None = None,
) -> tuple:
    if checkerboard_cell_size is None:
        checkerboard_cell_size = max(20, max(mesh.shape) // 10)

    checkerboard = _build_checkerboard_field(mesh, checkerboard_cell_size)
    deformed_checkerboard = checkerboard.eval(inverse_mapping(mesh.points), as_foi=True)

    if fig is None or ax is None:
        fig, ax = visualisations.subplots(1, 1)

    visualisations.pcolormesh(fig, ax, deformed_checkerboard, cmap=cmap)
    if title is not None:
        ax.set_title(title)

    return fig, ax, deformed_checkerboard


def infer_jacobian_sign_case(jacobian_field: FieldOfInterest) -> JacobianSignCase:
    jacobian_values = np.asarray(jacobian_field.values)
    if np.all(jacobian_values > 0.0):
        return "positive-only"
    if np.all(jacobian_values < 0.0):
        return "negative-only"
    return "mixed-sign"


def get_jacobian_cmap_and_norm(
    jacobian_field: FieldOfInterest,
) -> tuple[Normalize | TwoSlopeNorm, str]:
    sign_case = infer_jacobian_sign_case(jacobian_field)
    if sign_case == "positive-only":
        return Normalize(), "Reds"
    if sign_case == "negative-only":
        return Normalize(), "Blues_r"
    return TwoSlopeNorm(0.0), "RdBu_r"


def validate_jacobian_style_case(
    jacobian_field: FieldOfInterest,
    expected_case: JacobianSignCase,
) -> JacobianSignCase:
    inferred_case = infer_jacobian_sign_case(jacobian_field)
    if inferred_case != expected_case:
        raise ValueError(
            f"Unexpected Jacobian case: inferred '{inferred_case}', expected "
            f"'{expected_case}'."
        )
    return inferred_case


def plot_piecewise_jacobian_pair(
    mesh: RegularGrid,
    displacements_t: tuple,
    displacements_w: tuple,
    all_ffd_params: tuple,
    *,
    figsize: tuple[float, float] = (15, 6),
    title_t: str = "Jacobian of T",
    title_w: str = "Jacobian of W",
    save_path: str | None = None,
) -> tuple:
    """Plot side-by-side Jacobian determinant fields for piecewise T and W mappings."""
    points = jnp.array(mesh.points)

    jacobian_t_values = np.asarray(
        piecewise_jacobian_det_2d(points, displacements_t, all_ffd_params)
    )
    jacobian_w_values = np.asarray(
        piecewise_jacobian_det_2d(points, displacements_w, all_ffd_params)
    )

    jacobian_t = FieldOfInterest(None, mesh, jacobian_t_values)
    jacobian_w = FieldOfInterest(None, mesh, jacobian_w_values)

    fig, axes = visualisations.subplots(1, 2, figsize=figsize)

    jac_norm_t, cmap_t = get_jacobian_cmap_and_norm(jacobian_t)
    visualisations.pcolormesh(fig, axes[0], jacobian_t, cmap=cmap_t, norm=jac_norm_t)
    axes[0].set_title(
        f"{title_t}: [{jacobian_t.values.min():.2e}, {jacobian_t.values.max():.2e}]"
    )

    jac_norm_w, cmap_w = get_jacobian_cmap_and_norm(jacobian_w)
    visualisations.pcolormesh(fig, axes[1], jacobian_w, cmap=cmap_w, norm=jac_norm_w)
    axes[1].set_title(
        f"{title_w}: [{jacobian_w.values.min():.2e}, {jacobian_w.values.max():.2e}]"
    )

    fig.tight_layout()
    if save_path is not None:
        visualisations.savefig(fig, save_path)

    return fig, axes, jacobian_t, jacobian_w
