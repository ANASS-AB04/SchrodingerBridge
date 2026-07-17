"""Main RBF cost function implementation with pure functional API.

Uses interpax RBFInterpolator for the deformation mapping. Field values at
deformed points are interpolated via RegularGridInterpolator (grid meshes),
LinearNDInterpolator (scattered / PointCloud meshes by default), an optional
NearestNDInterpolator, or an optional RBFInterpolator over the full mesh.
"""

from collections.abc import Callable
from functools import partial
import logging
from typing import Any

import jax
import jax.numpy as jnp
from interpax import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator
from jax import Array
from jax.scipy.interpolate import RegularGridInterpolator
from jaxtyping import Float

from phdtruel.mappings.cost_functional.rbf.types import RBFStaticConfig, RBFTracedParams
from phdtruel.mappings.cost_functional.utils import (
    BarrierFnName,
    get_barrier_fn,
)
from phdtruel.mappings.ot_gaussian import wasserstein_distance_analytical_jax

logger = logging.getLogger(__name__)

NON_FINITE_PENALTY = 1.0e30


def _squared_l2(x: Any) -> Float[Array, ""]:
    """Compute squared L2 norm without sqrt to avoid NaN gradients at zero."""
    x_arr = jnp.asarray(x)
    return jnp.sum(jnp.square(x_arr))


def _warn_if_non_finite(name: str, value: Float[Array, ""]) -> None:
    """Emit a warning when a scalar/tensor contains NaN or Inf values."""

    has_non_finite = jnp.any(~jnp.isfinite(value))

    def _callback(flag: Array) -> None:
        if bool(flag):
            logger.warning(
                "Detected non-finite values in %s; applying nan_to_num safeguard.",
                name,
            )

    jax.debug.callback(_callback, has_non_finite)


def _warn_if_negative(name: str, value: Float[Array, ""]) -> None:
    """Emit a warning when a scalar/tensor contains negative values."""

    has_negative = jnp.any(value < 0.0)

    def _callback(flag: Array) -> None:
        if bool(flag):
            logger.warning("Detected negative values in %s.", name)

    jax.debug.callback(_callback, has_negative)


def _build_mapping_rbf_kwargs(config: RBFStaticConfig) -> dict[str, Any]:
    """Build kwargs for mapping RBFInterpolator (all centers, neighbors=None ok)."""
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


def _build_field_rbf_kwargs(config: RBFStaticConfig) -> dict[str, Any]:
    """Build kwargs for field RBFInterpolator (full mesh, finite neighbors)."""
    kwargs: dict[str, Any] = {
        "kernel": config.field_rbf_kernel,
        "neighbors": config.field_rbf_neighbors,
        "smoothing": config.rbf_smoothing,
    }
    if config.rbf_epsilon is not None:
        kwargs["epsilon"] = config.rbf_epsilon
    if config.rbf_degree is not None:
        kwargs["degree"] = config.rbf_degree
    return kwargs


def _rbf_mapping(
    points: Float[Array, "n_points 2"],
    centers: Float[Array, "n_c 2"],
    displacements: Float[Array, "n_c 2"],
    rbf_kwargs: dict[str, Any],
) -> Float[Array, "n_points 2"]:
    """Apply RBF displacement field: mapped = points + RBF(centers, disp)(points)."""
    rbf_x = RBFInterpolator(centers, displacements[:, 0], **rbf_kwargs)
    rbf_y = RBFInterpolator(centers, displacements[:, 1], **rbf_kwargs)
    disp = jnp.column_stack((rbf_x(points), rbf_y(points)))
    return points + disp


def _interpolate_field(
    points: Float[Array, "n_points 2"],
    field_values: Float[Array, " n"],
    config: RBFStaticConfig,
) -> Float[Array, " n_points"]:
    """Interpolate field values at given points."""
    if config.field_interp_kind == "grid":
        shape = tuple(ax.shape[0] for ax in config.mesh.axes)
        interp = RegularGridInterpolator(
            config.mesh.axes,
            field_values.reshape(shape),
            fill_value=0.0,
        )
        return interp(points)

    if config.field_interp_kind == "linear":
        interp = LinearNDInterpolator(
            config.mesh.points,
            field_values,
            fill_value=0.0,
            frozen_points=config.field_linear_frozen_points,
        )
        return interp(points)

    if config.field_interp_kind == "nearest":
        interp = NearestNDInterpolator(config.mesh.points, field_values)
        return interp(points)

    field_kwargs = _build_field_rbf_kwargs(config)
    interp = RBFInterpolator(config.mesh.points, field_values, **field_kwargs)
    return interp(points)


def _compute_alignment(
    u0_aligned: Float[Array, " n"],
    u1_aligned: Float[Array, " n"],
    u0_values: Float[Array, " n"],
    u1_values: Float[Array, " n"],
    alpha_A_algn: Float[Array, ""],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Compute field alignment costs for both directions."""
    norm_u0 = jnp.linalg.norm(u0_values.ravel())
    norm_u1 = jnp.linalg.norm(u1_values.ravel())
    normalization = norm_u0**2 + norm_u1**2
    normalization = jnp.maximum(normalization, jnp.finfo(u0_values.dtype).eps)

    alignment_0 = (
        alpha_A_algn
        * _squared_l2(u0_aligned.flatten() - u1_values.flatten())
        / normalization
    )
    alignment_1 = (
        alpha_A_algn
        * _squared_l2(u1_aligned.flatten() - u0_values.flatten())
        / normalization
    )

    return alignment_0, alignment_1


def _jacobian_det_2d(
    points: Float[Array, "n_points 2"],
    centers: Float[Array, "n_c 2"],
    displacements: Float[Array, "n_c 2"],
    rbf_kwargs: dict[str, Any],
) -> Float[Array, " n_points"]:
    """Compute Jacobian determinant of the RBF mapping at mesh points."""

    def map_point(point: Float[Array, " 2"]) -> Float[Array, " 2"]:
        return _rbf_mapping(
            point.reshape(1, 2),
            centers,
            displacements,
            rbf_kwargs,
        )[0]

    jacobians = jax.vmap(jax.jacfwd(map_point))(points)
    return (
        jacobians[:, 0, 0] * jacobians[:, 1, 1]
        - jacobians[:, 0, 1] * jacobians[:, 1, 0]
    )


def _compute_jacobian_barrier(
    displacements_w: Float[Array, "n_c 2"],
    displacements_t: Float[Array, "n_c 2"],
    mesh_points: Float[Array, "n_points 2"],
    centers: Float[Array, "n_c 2"],
    rbf_kwargs: dict[str, Any],
    alpha_C_jac: Float[Array, ""],
    barrier_epsilon: Float[Array, ""],
    barrier_fn_name: BarrierFnName,
    n_mesh: int,
) -> Float[Array, ""]:
    """Compute Jacobian barrier costs for positivity constraint."""

    def barrier_disabled():
        return jnp.array(0.0)

    def barrier_enabled():
        barrier_fn = get_barrier_fn(barrier_fn_name)

        jacobian_w = _jacobian_det_2d(mesh_points, centers, displacements_w, rbf_kwargs)
        jacobian_t = _jacobian_det_2d(mesh_points, centers, displacements_t, rbf_kwargs)

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
    centers: Float[Array, "n_c 2"],
    rbf_kwargs: dict[str, Any],
    alpha_C_jac: Float[Array, ""],
) -> Float[Array, ""]:
    """Count non-positive Jacobian determinants when barrier is active."""

    def barrier_disabled():
        return jnp.array(-1)

    def barrier_enabled():
        jacobian_w = _jacobian_det_2d(mesh_points, centers, displacements_w, rbf_kwargs)
        jacobian_t = _jacobian_det_2d(mesh_points, centers, displacements_t, rbf_kwargs)

        n_violations_w = jnp.sum(jacobian_w <= 0.0)
        n_violations_t = jnp.sum(jacobian_t <= 0.0)
        return n_violations_w + n_violations_t

    return jax.lax.cond(alpha_C_jac == 0.0, barrier_disabled, barrier_enabled)


def _compute_mapping_sync(
    mapping_w: Callable,
    mapping_t: Callable,
    mesh_points: Float[Array, "n_points 2"],
    alpha_B_bij: Float[Array, ""],
    n_mesh: int,
) -> Float[Array, ""]:
    """Compute mapping synchronization costs (bidirectional composition error)."""
    mapped_w_t = mapping_w(mapping_t(mesh_points))
    mapped_t_w = mapping_t(mapping_w(mesh_points))

    map_composition_w_t = (mapped_w_t - mesh_points).reshape(-1, 2)
    map_composition_t_w = (mapped_t_w - mesh_points).reshape(-1, 2)

    sync_cost = (
        alpha_B_bij
        * (_squared_l2(map_composition_w_t) + _squared_l2(map_composition_t_w))
        / n_mesh
    )

    return sync_cost


def _compute_control_reg(
    displacements_w: Float[Array, "n_c 2"],
    displacements_t: Float[Array, "n_c 2"],
    alpha_D_cp_norm: Float[Array, ""],
) -> Float[Array, ""]:
    """Compute control point displacement regularization cost."""
    n_cp = displacements_w.shape[0]
    total_cost = _squared_l2(displacements_w) + _squared_l2(displacements_t)
    return alpha_D_cp_norm * total_cost / n_cp


def _compute_imposed_disp_cost(
    mapping_w: Callable,
    mapping_t: Callable,
    imposed_points: Float[Array, "n_bc 2"],
    imposed_targets: Float[Array, "n_bc 2"],
    alpha_E_imposed_disp: Float[Array, ""],
) -> Float[Array, ""]:
    """Penalize deviation from imposed displacements at boundary points."""

    def imposed_disabled():
        return jnp.array(0.0)

    def imposed_enabled():
        n_pts = imposed_points.shape[0]
        if n_pts == 0:
            return jnp.array(0.0)

        target_positions = imposed_points + imposed_targets
        mapped_w = mapping_w(imposed_points)
        mapped_t = mapping_t(imposed_points)

        diff_w = _squared_l2(mapped_w - target_positions)
        diff_t = _squared_l2(mapped_t - target_positions)

        return alpha_E_imposed_disp * (diff_w + diff_t) / n_pts

    return jax.lax.cond(alpha_E_imposed_disp == 0.0, imposed_disabled, imposed_enabled)


def _compute_w2_cost(
    u0_aligned: Float[Array, " n"],
    u1_aligned: Float[Array, " n"],
    u0_values: Float[Array, " n"],
    u1_values: Float[Array, " n"],
    mesh_points: Float[Array, "n_points 2"],
    alpha_G_got_w2: Float[Array, ""],
) -> Float[Array, ""]:
    """Compute Wasserstein-2 distance costs."""

    def w2_disabled():
        return jnp.array(0.0)

    def _weighted_moments(
        points: Float[Array, "n_points 2"],
        weights: Float[Array, " n_points"],
    ) -> tuple[Float[Array, " 2"], Float[Array, " 2 2"]]:
        mu = jnp.average(points, axis=0, weights=weights.ravel())
        sigma = jnp.cov(points.T, aweights=weights.ravel())
        return mu, sigma

    def w2_enabled():
        mu0, cov0 = _weighted_moments(mesh_points, u0_values)
        mu0_aligned, cov0_aligned = _weighted_moments(mesh_points, u0_aligned)
        mu1, cov1 = _weighted_moments(mesh_points, u1_values)
        mu1_aligned, cov1_aligned = _weighted_moments(mesh_points, u1_aligned)

        dist_sq_0 = wasserstein_distance_analytical_jax(
            mu1, mu0_aligned, cov1, cov0_aligned
        )
        _warn_if_negative("dist_sq_0", dist_sq_0)

        dist_sq_1 = wasserstein_distance_analytical_jax(
            mu0, mu1_aligned, cov0, cov1_aligned
        )
        _warn_if_negative("dist_sq_1", dist_sq_1)

        return alpha_G_got_w2 * (dist_sq_0 + dist_sq_1)

    return jax.lax.cond(alpha_G_got_w2 == 0.0, w2_disabled, w2_enabled)


def _sanitize_cost(value: Float[Array, ""]) -> Float[Array, ""]:
    return jnp.nan_to_num(
        value,
        nan=NON_FINITE_PENALTY,
        posinf=NON_FINITE_PENALTY,
        neginf=NON_FINITE_PENALTY,
    )


def _compute_point_registration(
    mapping_w: Callable,
    mapping_t: Callable,
    point_registration_points_0: Float[Array, "n_registration 2"],
    point_registration_points_1: Float[Array, "n_registration 2"],
    alpha_J_point_registration: Float[Array, ""],
) -> Float[Array, ""]:
    """Compute bidirectional point registration cost for RBF mappings."""

    def point_registration_disabled():
        return jnp.array(0.0)

    def point_registration_enabled():
        n_pts = point_registration_points_0.shape[0]
        if n_pts == 0:
            return jnp.array(0.0)

        mapped_w = mapping_w(point_registration_points_0)
        mapped_t = mapping_t(point_registration_points_1)

        diff_w = _squared_l2(mapped_w - point_registration_points_1)
        diff_t = _squared_l2(mapped_t - point_registration_points_0)

        return alpha_J_point_registration * (diff_w + diff_t) / n_pts

    return jax.lax.cond(
        alpha_J_point_registration == 0.0,
        point_registration_disabled,
        point_registration_enabled,
    )


def rbf_cost_function(
    traced: RBFTracedParams,
    config: RBFStaticConfig,
) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
    """Pure functional RBF cost function.

    All hyperparameters are traced but frozen via stop_gradient, allowing
    barrier epsilon to be updated without recompilation.

    Args:
        traced: Traced parameters (displacements and hyperparameters).
        config: Static configuration (meshes, fields, RBF params, BC points).

    Returns:
        Tuple of (total cost, auxiliary dict with cost components).
    """
    alpha_A_algn = jax.lax.stop_gradient(traced.alpha_A_algn)
    alpha_B_bij = jax.lax.stop_gradient(traced.alpha_B_bij)
    alpha_C_jac = jax.lax.stop_gradient(traced.alpha_C_jac)
    alpha_D_cp_norm = jax.lax.stop_gradient(traced.alpha_D_cp_norm)
    alpha_E_imposed_disp = jax.lax.stop_gradient(traced.alpha_E_imposed_disp)
    alpha_G_got_w2 = jax.lax.stop_gradient(traced.alpha_G_got_w2)
    alpha_J_point_registration = jax.lax.stop_gradient(
        traced.alpha_J_point_registration
    )
    barrier_epsilon = jax.lax.stop_gradient(traced.barrier_config.barrier_epsilon)
    barrier_fn_name = traced.barrier_config.barrier_fn_name

    displacements_w = traced.displacements_w
    displacements_t = traced.displacements_t

    mapping_rbf_kwargs = _build_mapping_rbf_kwargs(config)
    mapping_w = partial(
        _rbf_mapping,
        centers=config.centers,
        displacements=displacements_w,
        rbf_kwargs=mapping_rbf_kwargs,
    )
    mapping_t = partial(
        _rbf_mapping,
        centers=config.centers,
        displacements=displacements_t,
        rbf_kwargs=mapping_rbf_kwargs,
    )

    mesh_points = config.mesh.points
    n_mesh = mesh_points.shape[0]

    u0_comp_w = _interpolate_field(
        jnp.asarray(mapping_w(mesh_points)),
        config.u0_values,
        config,
    )
    u1_comp_t = _interpolate_field(
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
        config.centers,
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
        config.centers,
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
    _warn_if_non_finite("w2", w2_costs)
    _warn_if_non_finite("point_registration", point_registration_costs)

    aux = {
        "alignment_0": _sanitize_cost(alignment_0),
        "alignment_1": _sanitize_cost(alignment_1),
        "jacobian_barrier": _sanitize_cost(jacobian_barrier_costs),
        "mapping_sync": _sanitize_cost(mapping_sync_costs),
        "control_reg": _sanitize_cost(control_reg_costs),
        "imposed_disp": _sanitize_cost(imposed_disp_costs),
        "w2": _sanitize_cost(w2_costs),
        "point_registration": _sanitize_cost(point_registration_costs),
        "jacobian_violations": jacobian_violations,
    }

    return total_cost, aux
