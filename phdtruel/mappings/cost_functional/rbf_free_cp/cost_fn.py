"""RBF free control-point cost function with optimizable interior centers."""

from functools import partial
import logging
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from phdtruel.mappings.cost_functional.rbf.cost_fn import (
    _compute_alignment,
    _compute_control_reg,
    _compute_imposed_disp_cost,
    _compute_mapping_sync,
    _compute_point_registration,
    _compute_w2_cost,
    _interpolate_field,
    _jacobian_det_2d,
    _rbf_mapping,
    _sanitize_cost,
    _squared_l2,
    _warn_if_non_finite,
)
from phdtruel.mappings.cost_functional.rbf.types import RBFStaticConfig
from phdtruel.mappings.cost_functional.rbf_free_cp.types import (
    RBFFreeStaticConfig,
    RBFFreeTracedParams,
)
from phdtruel.mappings.cost_functional.utils import BarrierFnName, get_barrier_fn

logger = logging.getLogger(__name__)


def _build_mapping_rbf_kwargs(config: RBFFreeStaticConfig) -> dict[str, Any]:
    """Build kwargs for mapping RBFInterpolator from free-CP static config."""
    kwargs: dict[str, Any] = {
        "kernel": config.rbf_kernel,
        "neighbors": config.rbf_neighbors,
        "smoothing": config.rbf_smoothing,
    }
    if config.rbf_epsilon is not None:
        kwargs["epsilon"] = config.rbf_epsilon
    if config.rbf_degree is not None:
        kwargs["degree"] = config.rbf_degree
    return kwargs


def _interpolate_field_free(
    points: Float[Array, "n_points 2"],
    field_values: Float[Array, " n"],
    config: RBFFreeStaticConfig,
) -> Float[Array, " n_points"]:
    """Interpolate field values using the free-CP static config."""
    rbf_config = RBFStaticConfig(
        mesh=config.mesh,
        u0_values=config.u0_values,
        u1_values=config.u1_values,
        centers=config.boundary_centers,
        imposed_disp_points=config.imposed_disp_points,
        imposed_disp_targets=config.imposed_disp_targets,
        point_registration_points_0=config.point_registration_points_0,
        point_registration_points_1=config.point_registration_points_1,
        rbf_kernel=config.rbf_kernel,
        rbf_neighbors=config.rbf_neighbors,
        rbf_epsilon=config.rbf_epsilon,
        rbf_smoothing=config.rbf_smoothing,
        rbf_degree=config.rbf_degree,
        field_interp_kind=config.field_interp_kind,
        field_linear_frozen_points=config.field_linear_frozen_points,
        field_rbf_neighbors=config.field_rbf_neighbors,
        field_rbf_kernel=config.field_rbf_kernel,
        alpha_A_algn=config.alpha_A_algn,
        alpha_B_bij=config.alpha_B_bij,
        alpha_C_jac=config.alpha_C_jac,
        alpha_D_cp_norm=config.alpha_D_cp_norm,
        alpha_E_imposed_disp=config.alpha_E_imposed_disp,
        alpha_G_got_w2=config.alpha_G_got_w2,
        alpha_J_point_registration=config.alpha_J_point_registration,
        enable_w2_cost=config.enable_w2_cost,
        enable_point_registration=config.enable_point_registration,
    )
    return _interpolate_field(points, field_values, rbf_config)


def _build_full_centers(
    boundary_centers: Float[Array, "n_b 2"],
    interior_centers: Float[Array, "n_i 2"],
) -> Float[Array, "n_c 2"]:
    """Concatenate pinned boundary centers with optimizable interior centers."""
    return jnp.concatenate([boundary_centers, interior_centers], axis=0)


def _compute_jacobian_barrier(
    displacements_w: Float[Array, "n_c 2"],
    displacements_t: Float[Array, "n_c 2"],
    mesh_points: Float[Array, "n_points 2"],
    centers_w: Float[Array, "n_c 2"],
    centers_t: Float[Array, "n_c 2"],
    rbf_kwargs: dict[str, Any],
    alpha_C_jac: Float[Array, ""],
    barrier_epsilon: Float[Array, ""],
    barrier_fn_name: BarrierFnName,
    n_mesh: int,
) -> Float[Array, ""]:
    """Jacobian barrier with separate center sets for W and T mappings."""

    def barrier_disabled():
        return jnp.array(0.0)

    def barrier_enabled():
        barrier_fn = get_barrier_fn(barrier_fn_name)

        jacobian_w = _jacobian_det_2d(
            mesh_points, centers_w, displacements_w, rbf_kwargs
        )
        jacobian_t = _jacobian_det_2d(
            mesh_points, centers_t, displacements_t, rbf_kwargs
        )

        barrier_cost = (
            alpha_C_jac
            * (
                _squared_l2(barrier_fn(jacobian_w, barrier_epsilon))
                + _squared_l2(barrier_fn(jacobian_t, barrier_epsilon))
            )
            / n_mesh
        )
        return barrier_cost

    return jax.lax.cond(alpha_C_jac == 0.0, barrier_disabled, barrier_enabled)


def _compute_jacobian_violations(
    displacements_w: Float[Array, "n_c 2"],
    displacements_t: Float[Array, "n_c 2"],
    mesh_points: Float[Array, "n_points 2"],
    centers_w: Float[Array, "n_c 2"],
    centers_t: Float[Array, "n_c 2"],
    rbf_kwargs: dict[str, Any],
    alpha_C_jac: Float[Array, ""],
) -> Float[Array, ""]:
    """Count non-positive Jacobian determinants for separate W/T center sets."""

    def barrier_disabled():
        return jnp.array(-1)

    def barrier_enabled():
        jacobian_w = _jacobian_det_2d(
            mesh_points, centers_w, displacements_w, rbf_kwargs
        )
        jacobian_t = _jacobian_det_2d(
            mesh_points, centers_t, displacements_t, rbf_kwargs
        )

        n_violations_w = jnp.sum(jacobian_w <= 0.0)
        n_violations_t = jnp.sum(jacobian_t <= 0.0)
        return n_violations_w + n_violations_t

    return jax.lax.cond(alpha_C_jac == 0.0, barrier_disabled, barrier_enabled)


def _compute_interior_pos_reg(
    interior_centers_w: Float[Array, "n_i 2"],
    interior_centers_t: Float[Array, "n_i 2"],
    initial_interior_centers_w: Float[Array, "n_i 2"],
    initial_interior_centers_t: Float[Array, "n_i 2"],
    alpha_F_cp_pos_reg: Float[Array, ""],
    n_interior: int,
) -> Float[Array, ""]:
    """Penalize deviation of interior centers from their initial positions."""

    def reg_disabled():
        return jnp.array(0.0)

    def reg_enabled():
        diff_w = _squared_l2(interior_centers_w - initial_interior_centers_w)
        diff_t = _squared_l2(interior_centers_t - initial_interior_centers_t)
        return alpha_F_cp_pos_reg * (diff_w + diff_t) / n_interior

    return jax.lax.cond(alpha_F_cp_pos_reg == 0.0, reg_disabled, reg_enabled)


def rbf_free_cp_cost_function(
    traced: RBFFreeTracedParams,
    config: RBFFreeStaticConfig,
) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
    """Pure functional RBF free-CP cost function.

    Interior control-point positions are optimized separately for W and T
    mappings.  Boundary center positions remain pinned in the static config.

    Args:
        traced: Traced parameters (interior centers, displacements, alphas).
        config: Static configuration (meshes, fields, RBF params, BC points).

    Returns:
        Tuple of (total cost, auxiliary dict with cost components).
    """
    alpha_A_algn = jax.lax.stop_gradient(traced.alpha_A_algn)
    alpha_B_bij = jax.lax.stop_gradient(traced.alpha_B_bij)
    alpha_C_jac = jax.lax.stop_gradient(traced.alpha_C_jac)
    alpha_D_cp_norm = jax.lax.stop_gradient(traced.alpha_D_cp_norm)
    alpha_E_imposed_disp = jax.lax.stop_gradient(traced.alpha_E_imposed_disp)
    alpha_F_cp_pos_reg = jax.lax.stop_gradient(traced.alpha_F_cp_pos_reg)
    alpha_G_got_w2 = jax.lax.stop_gradient(traced.alpha_G_got_w2)
    alpha_J_point_registration = jax.lax.stop_gradient(
        traced.alpha_J_point_registration
    )
    barrier_epsilon = jax.lax.stop_gradient(traced.barrier_config.barrier_epsilon)
    barrier_fn_name = traced.barrier_config.barrier_fn_name

    interior_centers_w = traced.interior_centers_w
    interior_centers_t = traced.interior_centers_t
    displacements_w = traced.displacements_w
    displacements_t = traced.displacements_t

    centers_w = _build_full_centers(config.boundary_centers, interior_centers_w)
    centers_t = _build_full_centers(config.boundary_centers, interior_centers_t)

    mapping_rbf_kwargs = _build_mapping_rbf_kwargs(config)
    mapping_w = partial(
        _rbf_mapping,
        centers=centers_w,
        displacements=displacements_w,
        rbf_kwargs=mapping_rbf_kwargs,
    )
    mapping_t = partial(
        _rbf_mapping,
        centers=centers_t,
        displacements=displacements_t,
        rbf_kwargs=mapping_rbf_kwargs,
    )

    mesh_points = config.mesh.points
    n_mesh = mesh_points.shape[0]

    u0_comp_w = _interpolate_field_free(
        jnp.asarray(mapping_w(mesh_points)),
        config.u0_values,
        config,
    )
    u1_comp_t = _interpolate_field_free(
        jnp.asarray(mapping_t(mesh_points)),
        config.u1_values,
        config,
    )

    alignment_0, alignment_1 = _compute_alignment(
        u0_comp_w, u1_comp_t, config.u0_values, config.u1_values, alpha_A_algn
    )

    jacobian_barrier_costs = _compute_jacobian_barrier(
        displacements_w,
        displacements_t,
        mesh_points,
        centers_w,
        centers_t,
        mapping_rbf_kwargs,
        alpha_C_jac,
        barrier_epsilon,
        barrier_fn_name,
        n_mesh,
    )

    jacobian_violations = _compute_jacobian_violations(
        displacements_w,
        displacements_t,
        mesh_points,
        centers_w,
        centers_t,
        mapping_rbf_kwargs,
        alpha_C_jac,
    )

    mapping_sync_costs = _compute_mapping_sync(
        mapping_w, mapping_t, mesh_points, alpha_B_bij, n_mesh
    )

    control_reg_costs = _compute_control_reg(
        displacements_w, displacements_t, alpha_D_cp_norm
    )

    imposed_disp_costs = _compute_imposed_disp_cost(
        mapping_w,
        mapping_t,
        config.imposed_disp_points,
        config.imposed_disp_targets,
        alpha_E_imposed_disp,
    )

    interior_pos_reg_costs = _compute_interior_pos_reg(
        interior_centers_w,
        interior_centers_t,
        config.initial_interior_centers_w,
        config.initial_interior_centers_t,
        alpha_F_cp_pos_reg,
        config.n_interior,
    )

    w2_costs = _compute_w2_cost(
        u0_comp_w,
        u1_comp_t,
        config.u0_values,
        config.u1_values,
        mesh_points,
        alpha_G_got_w2,
    )

    point_registration_costs = _compute_point_registration(
        mapping_w,
        mapping_t,
        config.point_registration_points_0,
        config.point_registration_points_1,
        alpha_J_point_registration,
    )

    barrier_cost = (
        jacobian_barrier_costs
        + mapping_sync_costs
        + control_reg_costs
        + imposed_disp_costs
        + interior_pos_reg_costs
        + w2_costs
        + point_registration_costs
    )
    total_cost = alignment_0 + alignment_1 + barrier_cost

    _warn_if_non_finite("total_cost", total_cost)
    total_cost = _sanitize_cost(total_cost)

    _warn_if_non_finite("alignment_0", alignment_0)
    _warn_if_non_finite("alignment_1", alignment_1)
    _warn_if_non_finite("jacobian_barrier", jacobian_barrier_costs)
    _warn_if_non_finite("mapping_sync", mapping_sync_costs)
    _warn_if_non_finite("control_reg", control_reg_costs)
    _warn_if_non_finite("imposed_disp", imposed_disp_costs)
    _warn_if_non_finite("interior_pos_reg", interior_pos_reg_costs)
    _warn_if_non_finite("w2", w2_costs)
    _warn_if_non_finite("point_registration", point_registration_costs)

    aux: dict[str, Float[Array, ""] | Array] = {
        "alignment_0": _sanitize_cost(alignment_0),
        "alignment_1": _sanitize_cost(alignment_1),
        "jacobian_barrier": _sanitize_cost(jacobian_barrier_costs),
        "mapping_sync": _sanitize_cost(mapping_sync_costs),
        "control_reg": _sanitize_cost(control_reg_costs),
        "imposed_disp": _sanitize_cost(imposed_disp_costs),
        "interior_pos_reg": _sanitize_cost(interior_pos_reg_costs),
        "w2": _sanitize_cost(w2_costs),
        "point_registration": _sanitize_cost(point_registration_costs),
        "jacobian_violations": jacobian_violations,
    }

    if config.record_cp_history:
        aux["cp_centers_w"] = centers_w
        aux["cp_disp_w"] = displacements_w
        aux["cp_centers_t"] = centers_t
        aux["cp_disp_t"] = displacements_t

    return total_cost, aux
