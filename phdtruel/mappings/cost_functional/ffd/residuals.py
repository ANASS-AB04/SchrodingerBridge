"""Per-term residual vectors for piecewise FFD (Gauss-Newton / natural gradient).

Each returned vector ``r_k`` satisfies ``||r_k||^2 = cost_k`` for the
corresponding squared term in :mod:`cost_fn`.  The W2 (Gaussian OT) term is
omitted because it is not a plain sum-of-squares residual.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Float

from phdtruel.mappings.cost_functional.ffd.cost_fn import _interpolate_field_piecewise
from phdtruel.mappings.cost_functional.ffd.types import (
    FFDStaticConfig,
    FFDTracedParams,
)
from phdtruel.mappings.cost_functional.utils import (
    BarrierFnName,
    get_barrier_fn,
    impose_control_displacements,
)
from phdtruel.mappings.ffd import (
    FFDParameters,
    ffd_mapping_2d,
    jacobian_matrix_2d,
    piecewise_ffd_mapping_2d,
    piecewise_jacobian_det_2d,
)


def _alignment_residuals(
    u0_aligned: Float[jax.Array, " n"],
    u1_aligned: Float[jax.Array, " n"],
    u0_values: Float[jax.Array, " n"],
    u1_values: Float[jax.Array, " n"],
    alpha_A_algn: Float[jax.Array, ""],
) -> dict[str, Float[jax.Array, " n"]]:
    norm_u0 = jnp.linalg.norm(u0_values.ravel())
    norm_u1 = jnp.linalg.norm(u1_values.ravel())
    normalization = norm_u0**2 + norm_u1**2
    normalization = jnp.maximum(normalization, jnp.finfo(u0_values.dtype).eps)
    scale = jnp.sqrt(alpha_A_algn / normalization)
    return {
        "alignment_0": scale * (u0_aligned.ravel() - u1_values.ravel()),
        "alignment_1": scale * (u1_aligned.ravel() - u0_values.ravel()),
    }


def _jacobian_barrier_residuals(
    displacements_w: tuple[Float[jax.Array, "n_cp_x n_cp_y 2"], ...],
    displacements_t: tuple[Float[jax.Array, "n_cp_x n_cp_y 2"], ...],
    mesh_points: Float[jax.Array, "n_points 2"],
    all_ffd_parameters: tuple[FFDParameters, ...],
    alpha_C_jac: Float[jax.Array, ""],
    barrier_epsilon: Float[jax.Array, ""],
    barrier_fn_name: BarrierFnName,
    n_mesh: int,
) -> dict[str, Float[jax.Array, " n_points"]]:
    def barrier_disabled() -> dict[str, Float[jax.Array, " n_points"]]:
        z = jnp.zeros((n_mesh,), dtype=mesh_points.dtype)
        return {"jacobian_barrier_w": z, "jacobian_barrier_t": z}

    def barrier_enabled() -> dict[str, Float[jax.Array, " n_points"]]:
        barrier_fn = get_barrier_fn(barrier_fn_name)
        scale = jnp.sqrt(alpha_C_jac / n_mesh)
        jacobian_w = jnp.asarray(
            piecewise_jacobian_det_2d(mesh_points, displacements_w, all_ffd_parameters)
        )
        jacobian_t = jnp.asarray(
            piecewise_jacobian_det_2d(mesh_points, displacements_t, all_ffd_parameters)
        )
        return {
            "jacobian_barrier_w": scale * barrier_fn(jacobian_w, barrier_epsilon),
            "jacobian_barrier_t": scale * barrier_fn(jacobian_t, barrier_epsilon),
        }

    return jax.lax.cond(alpha_C_jac == 0.0, barrier_disabled, barrier_enabled)


def _mapping_sync_residuals(
    mapping_w: Callable,
    mapping_t: Callable,
    mesh_points: Float[jax.Array, "n_points 2"],
    alpha_B_bij: Float[jax.Array, ""],
    n_mesh: int,
) -> dict[str, Float[jax.Array, " n_sync"]]:
    mapped_w_t = mapping_w(mapping_t(mesh_points))
    mapped_t_w = mapping_t(mapping_w(mesh_points))
    scale = jnp.sqrt(alpha_B_bij / n_mesh)
    return {
        "sync_w_t": scale * (mapped_w_t - mesh_points).ravel(),
        "sync_t_w": scale * (mapped_t_w - mesh_points).ravel(),
    }


def _control_reg_residuals(
    displacements_w: tuple[Float[jax.Array, "n_cp_x n_cp_y 2"], ...],
    displacements_t: tuple[Float[jax.Array, "n_cp_x n_cp_y 2"], ...],
    alpha_D_cp_norm: Float[jax.Array, ""],
) -> dict[str, Float[jax.Array, " n_cp"]]:
    total_n_cp = sum(
        disp_w.shape[0] * disp_w.shape[1]
        for disp_w, _ in zip(displacements_w, displacements_t)
    )
    scale = jnp.sqrt(alpha_D_cp_norm / total_n_cp)
    out: dict[str, Float[jax.Array, " n_cp"]] = {}
    for idx, (disp_w, disp_t) in enumerate(zip(displacements_w, displacements_t)):
        out[f"control_reg_w_{idx}"] = scale * disp_w.ravel()
        out[f"control_reg_t_{idx}"] = scale * disp_t.ravel()
    return out


def _inter_region_sync_residuals(
    displacements_w: tuple[Float[jax.Array, "n_cp_x n_cp_y 2"], ...],
    displacements_t: tuple[Float[jax.Array, "n_cp_x n_cp_y 2"], ...],
    all_ffd_parameters: tuple[FFDParameters, ...],
    inter_region_sync_indices: tuple[tuple[int, int], ...],
    inter_region_sync_points: tuple[Float[jax.Array, "n_sync 2"], ...],
    alpha_H_inter_sync: Float[jax.Array, ""],
) -> dict[str, Float[jax.Array, " n_sync_pts"]]:
    total_n_pts = sum(
        sync_points.shape[0]
        for (_, _), sync_points in zip(
            inter_region_sync_indices, inter_region_sync_points
        )
    )
    if total_n_pts == 0:
        return {}

    scale = jnp.sqrt(alpha_H_inter_sync / total_n_pts)
    active = jnp.where(alpha_H_inter_sync == 0.0, 0.0, 1.0)
    out: dict[str, Float[jax.Array, " n_sync_pts"]] = {}
    for pair_idx, ((region_i, region_j), sync_points) in enumerate(
        zip(inter_region_sync_indices, inter_region_sync_points)
    ):
        n_pts = sync_points.shape[0]
        if n_pts == 0:
            continue
        params_i = all_ffd_parameters[region_i]
        params_j = all_ffd_parameters[region_j]
        mapped_w_i = ffd_mapping_2d(sync_points, displacements_w[region_i], params_i)
        mapped_t_i = ffd_mapping_2d(sync_points, displacements_t[region_i], params_i)
        mapped_w_j = ffd_mapping_2d(sync_points, displacements_w[region_j], params_j)
        mapped_t_j = ffd_mapping_2d(sync_points, displacements_t[region_j], params_j)
        out[f"inter_sync_w_{pair_idx}"] = (
            active * scale * (mapped_w_i - mapped_w_j).ravel()
        )
        out[f"inter_sync_t_{pair_idx}"] = (
            active * scale * (mapped_t_i - mapped_t_j).ravel()
        )
    return out


def _inter_region_jac_sync_residuals(
    displacements_w: tuple[Float[jax.Array, "n_cp_x n_cp_y 2"], ...],
    displacements_t: tuple[Float[jax.Array, "n_cp_x n_cp_y 2"], ...],
    all_ffd_parameters: tuple[FFDParameters, ...],
    inter_region_sync_indices: tuple[tuple[int, int], ...],
    inter_region_sync_points: tuple[Float[jax.Array, "n_sync 2"], ...],
    alpha_I_inter_jac_sync: Float[jax.Array, ""],
) -> dict[str, Float[jax.Array, " n_jac"]]:
    total_n_pts = sum(
        sync_points.shape[0]
        for (_, _), sync_points in zip(
            inter_region_sync_indices, inter_region_sync_points
        )
    )
    if total_n_pts == 0:
        return {}

    scale = jnp.sqrt(alpha_I_inter_jac_sync / total_n_pts)
    active = jnp.where(alpha_I_inter_jac_sync == 0.0, 0.0, 1.0)
    out: dict[str, Float[jax.Array, " n_jac"]] = {}
    for pair_idx, ((region_i, region_j), sync_points) in enumerate(
        zip(inter_region_sync_indices, inter_region_sync_points)
    ):
        n_pts = sync_points.shape[0]
        if n_pts == 0:
            continue
        params_i = all_ffd_parameters[region_i]
        params_j = all_ffd_parameters[region_j]
        jac_w_i = jacobian_matrix_2d(sync_points, displacements_w[region_i], params_i)
        jac_t_i = jacobian_matrix_2d(sync_points, displacements_t[region_i], params_i)
        jac_w_j = jacobian_matrix_2d(sync_points, displacements_w[region_j], params_j)
        jac_t_j = jacobian_matrix_2d(sync_points, displacements_t[region_j], params_j)
        out[f"inter_jac_sync_w_{pair_idx}"] = (
            active * scale * (jac_w_i - jac_w_j).ravel()
        )
        out[f"inter_jac_sync_t_{pair_idx}"] = (
            active * scale * (jac_t_i - jac_t_j).ravel()
        )
    return out


def _point_registration_residuals(
    mapping_w: Callable,
    mapping_t: Callable,
    point_registration_points_0: Float[jax.Array, "n_registration 2"],
    point_registration_points_1: Float[jax.Array, "n_registration 2"],
    alpha_J_point_registration: Float[jax.Array, ""],
) -> dict[str, Float[jax.Array, " n_registration_pts"]]:
    """Return residual vectors for the bidirectional point registration term."""
    n_pts = point_registration_points_0.shape[0]
    if n_pts == 0:
        return {}

    scale = jnp.sqrt(alpha_J_point_registration / n_pts)
    active = jnp.where(alpha_J_point_registration == 0.0, 0.0, 1.0)
    mapped_w = mapping_w(point_registration_points_0)
    mapped_t = mapping_t(point_registration_points_1)
    return {
        "point_reg_w": active
        * scale
        * (mapped_w - point_registration_points_1).ravel(),
        "point_reg_t": active
        * scale
        * (mapped_t - point_registration_points_0).ravel(),
    }


def piecewise_ffd_residual_function(
    traced: FFDTracedParams,
    config: FFDStaticConfig,
) -> dict[str, Float[jax.Array, " n"]]:
    """Return unsquared residual vectors for each FFD cost term.

    W2 (``alpha_G_got_w2``) is excluded: it is not a sum-of-squares residual.

    Args:
        traced: Traced FFD parameters.
        config: Static FFD configuration.

    Returns:
        Dictionary mapping term names to 1-D residual vectors.
    """
    alpha_A_algn = jax.lax.stop_gradient(traced.alpha_A_algn)
    alpha_B_bij = jax.lax.stop_gradient(traced.alpha_B_bij)
    alpha_C_jac = jax.lax.stop_gradient(traced.alpha_C_jac)
    alpha_D_cp_norm = jax.lax.stop_gradient(traced.alpha_D_cp_norm)
    alpha_H_inter_sync = jax.lax.stop_gradient(traced.alpha_H_inter_sync)
    alpha_I_inter_jac_sync = jax.lax.stop_gradient(traced.alpha_I_inter_jac_sync)
    alpha_J_point_registration = jax.lax.stop_gradient(
        traced.alpha_J_point_registration
    )
    barrier_epsilon = jax.lax.stop_gradient(traced.barrier_config.barrier_epsilon)
    barrier_fn_name = traced.barrier_config.barrier_fn_name

    displacements_w = tuple(
        impose_control_displacements(disp, mask)
        for disp, mask in zip(
            traced.displacements_w, config.all_imposed_displacements_masks
        )
    )
    displacements_t = tuple(
        impose_control_displacements(disp, mask)
        for disp, mask in zip(
            traced.displacements_t, config.all_imposed_displacements_masks
        )
    )

    mapping_w = partial(
        piecewise_ffd_mapping_2d,
        all_control_points_displacement=displacements_w,
        all_params=config.all_ffd_parameters,
    )
    mapping_t = partial(
        piecewise_ffd_mapping_2d,
        all_control_points_displacement=displacements_t,
        all_params=config.all_ffd_parameters,
    )

    mesh_points = config.mesh.points
    n_mesh = mesh_points.shape[0]

    u0_comp_w = _interpolate_field_piecewise(
        jnp.asarray(mapping_w(mesh_points)),
        config.region_meshes,
        config.u0_per_region,
    )
    u1_comp_t = _interpolate_field_piecewise(
        jnp.asarray(mapping_t(mesh_points)),
        config.region_meshes,
        config.u1_per_region,
    )

    residuals: dict[str, Float[jax.Array, " n"]] = {}
    residuals.update(
        _alignment_residuals(
            u0_comp_w, u1_comp_t, config.u0_values, config.u1_values, alpha_A_algn
        )
    )
    residuals.update(
        _jacobian_barrier_residuals(
            displacements_w,
            displacements_t,
            mesh_points,
            config.all_ffd_parameters,
            alpha_C_jac,
            barrier_epsilon,
            barrier_fn_name,
            n_mesh,
        )
    )
    residuals.update(
        _mapping_sync_residuals(mapping_w, mapping_t, mesh_points, alpha_B_bij, n_mesh)
    )
    residuals.update(
        _control_reg_residuals(displacements_w, displacements_t, alpha_D_cp_norm)
    )
    if config.enable_inter_sync:
        residuals.update(
            _inter_region_sync_residuals(
                displacements_w,
                displacements_t,
                config.all_ffd_parameters,
                config.inter_region_sync_indices,
                config.inter_region_sync_points,
                alpha_H_inter_sync,
            )
        )
        residuals.update(
            _inter_region_jac_sync_residuals(
                displacements_w,
                displacements_t,
                config.all_ffd_parameters,
                config.inter_region_sync_indices,
                config.inter_region_sync_points,
                alpha_I_inter_jac_sync,
            )
        )

    if config.enable_point_registration:
        residuals.update(
            _point_registration_residuals(
                mapping_w,
                mapping_t,
                config.point_registration_points_0,
                config.point_registration_points_1,
                alpha_J_point_registration,
            )
        )

    return residuals
