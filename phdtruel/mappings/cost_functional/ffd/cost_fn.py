"""Main FFD cost function implementation with pure functional API.

Matches the logic of the original PiecewiseFFDMapSyncCostFn class:
- Only the mapping handles the pieces (via piecewise_ffd_mapping_2d).
- Interpolation uses per-region interpolators but evaluates on the
  **global mesh** points.
- Alignment, jacobian barrier, mapping sync, and W2 are all computed
  once on the full global mesh.
"""

from collections.abc import Callable
from functools import partial
import logging
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.interpolate import RegularGridInterpolator
from jaxtyping import Float

from phdtruel.mappings.cost_functional.ffd.types import (
    FFDStaticConfig,
    FFDTracedParams,
)
from phdtruel.mappings.cost_functional.types import MeshJaxed
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


def _interpolate_field_piecewise(
    points: Float[Array, "n_points 2"],
    region_meshes: tuple[MeshJaxed, ...],
    field_values_per_region: tuple[Float[Array, " n"], ...],
) -> Float[Array, " n_points"]:
    """Interpolate field values at given points using per-region interpolators.

    Matches the old _u0/_u1 logic: for each region, interpolate using that
    region's mesh and field values. Points outside a region get NaN from
    that region's interpolator; we take the first valid (non-NaN) value.

    Args:
        points: Points to interpolate at, shape (n_points, 2)
        region_meshes: Tuple of per-region meshes
        field_values_per_region: Tuple of field values per region

    Returns:
        Interpolated values at points
    """
    # Start with zeros (like old code)
    result = jnp.zeros(points.shape[0])

    for mesh, field_vals in zip(region_meshes, field_values_per_region):
        shape = tuple(ax.shape[0] for ax in mesh.axes)
        interp = RegularGridInterpolator(
            mesh.axes,
            field_vals.reshape(shape),
            fill_value=jnp.nan,  # NaN for out-of-bounds
        )
        region_values = interp(points)
        region_values_clean = jnp.nan_to_num(region_values, nan=0.0)
        # Use region values where valid (not NaN), like old code
        result = jnp.where(jnp.isnan(region_values), result, region_values_clean)

    return result


def _compute_alignment(
    u0_aligned: Float[Array, " n"],
    u1_aligned: Float[Array, " n"],
    u0_values: Float[Array, " n"],
    u1_values: Float[Array, " n"],
    alpha_A_algn: Float[Array, ""],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Compute field alignment costs for both directions.

    Args:
        u0_aligned: Field 0 mapped by W
        u1_aligned: Field 1 mapped by T
        u0_values: Original field 0 values
        u1_values: Original field 1 values
        alpha_A_algn: Alignment weight (frozen via stop_gradient)

    Returns:
        Tuple of (alignment_0, alignment_1) costs
    """
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


def _compute_jacobian_barrier(
    displacements_w: tuple[Float[Array, "n_cp_x n_cp_y 2"], ...],
    displacements_t: tuple[Float[Array, "n_cp_x n_cp_y 2"], ...],
    mesh_points: Float[Array, "n_points 2"],
    all_ffd_parameters: tuple[FFDParameters, ...],
    alpha_C_jac: Float[Array, ""],
    barrier_epsilon: Float[Array, ""],
    barrier_fn_name: BarrierFnName,
    n_mesh: int,
) -> Float[Array, ""]:
    """Compute Jacobian barrier costs for positivity constraint.

    Args:
        displacements_w: W mapping displacements per region
        displacements_t: T mapping displacements per region
        mesh_points: Mesh points to evaluate Jacobian at
        all_ffd_parameters: FFD parameters for each region
        alpha_C_jac: Jacobian barrier weight (frozen)
        barrier_epsilon: Barrier steepness parameter (traced, can be updated)
        barrier_fn_name: Name of the barrier function (resolved at trace time)
        n_mesh: Number of mesh points (for normalization)

    Returns:
        Total Jacobian barrier cost
    """

    def barrier_disabled():
        return jnp.array(0.0)

    def barrier_enabled():
        barrier_fn = get_barrier_fn(barrier_fn_name)

        jacobian_w = jnp.asarray(
            piecewise_jacobian_det_2d(mesh_points, displacements_w, all_ffd_parameters)
        )
        jacobian_t = jnp.asarray(
            piecewise_jacobian_det_2d(mesh_points, displacements_t, all_ffd_parameters)
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
    displacements_w: tuple[Float[Array, "n_cp_x n_cp_y 2"], ...],
    displacements_t: tuple[Float[Array, "n_cp_x n_cp_y 2"], ...],
    mesh_points: Float[Array, "n_points 2"],
    all_ffd_parameters: tuple[FFDParameters, ...],
    alpha_C_jac: Float[Array, ""],
) -> Float[Array, ""]:
    """Count non-positive Jacobian determinants when barrier is active.

    Args:
        displacements_w: W mapping displacements per region
        displacements_t: T mapping displacements per region
        mesh_points: Mesh points to evaluate Jacobian at
        all_ffd_parameters: FFD parameters for each region
        alpha_C_jac: Jacobian barrier weight; if 0, barrier not active

    Returns:
        Count of non-positive determinants (det <= 0) as a JAX array,
        or -1 (sentinel) if barrier is disabled (alpha_C_jac == 0)
    """

    def barrier_disabled():
        # Return -1 as sentinel when barrier is not active
        return jnp.array(-1)

    def barrier_enabled():
        jacobian_w = jnp.asarray(
            piecewise_jacobian_det_2d(mesh_points, displacements_w, all_ffd_parameters)
        )
        jacobian_t = jnp.asarray(
            piecewise_jacobian_det_2d(mesh_points, displacements_t, all_ffd_parameters)
        )

        n_violations_w = jnp.sum(jacobian_w <= 0.0)
        n_violations_t = jnp.sum(jacobian_t <= 0.0)
        total_violations = n_violations_w + n_violations_t

        return total_violations

    # Only compute when barrier is active; otherwise return -1 (sentinel)
    # The check is based on alpha_C_jac which tells us if the barrier cost is being used
    return jax.lax.cond(
        alpha_C_jac == 0.0,
        barrier_disabled,
        barrier_enabled,
    )


def _compute_mapping_sync(
    mapping_w: Callable,
    mapping_t: Callable,
    mesh_points: Float[Array, "n_points 2"],
    alpha_B_bij: Float[Array, ""],
    n_mesh: int,
) -> Float[Array, ""]:
    """Compute mapping synchronization costs (bidirectional composition error).

    Args:
        mapping_w: W mapping function
        mapping_t: T mapping function
        mesh_points: Mesh points to evaluate at
        alpha_B_bij: Mapping sync weight
        n_mesh: Number of mesh points (for normalization)

    Returns:
        Mapping synchronization cost
    """
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
    displacements_w: tuple[Float[Array, "n_cp_x n_cp_y 2"], ...],
    displacements_t: tuple[Float[Array, "n_cp_x n_cp_y 2"], ...],
    alpha_D_cp_norm: Float[Array, ""],
) -> Float[Array, ""]:
    """Compute control point displacement regularization cost.

    Args:
        displacements_w: W mapping displacements per region
        displacements_t: T mapping displacements per region
        alpha_D_cp_norm: Regularization weight

    Returns:
        Regularization cost
    """
    total_cost = 0.0
    total_n_cp = 0

    for disp_w, disp_t in zip(displacements_w, displacements_t):
        n_cp = disp_w.shape[0] * disp_w.shape[1]
        total_n_cp += n_cp

        cost_w = _squared_l2(disp_w)
        cost_t = _squared_l2(disp_t)

        total_cost += cost_w + cost_t

    return alpha_D_cp_norm * total_cost / total_n_cp


def _compute_w2_cost(
    u0_aligned: Float[Array, " n"],
    u1_aligned: Float[Array, " n"],
    u0_values: Float[Array, " n"],
    u1_values: Float[Array, " n"],
    mesh_points: Float[Array, "n_points 2"],
    alpha_G_got_w2: Float[Array, ""],
) -> Float[Array, ""]:
    """Compute Wasserstein-2 distance costs.

    Args:
        u0_aligned: Field 0 mapped by W
        u1_aligned: Field 1 mapped by T
        u0_values: Original field 0 values
        u1_values: Original field 1 values
        mesh_points: Mesh point coordinates
        alpha_G_got_w2: W2 distance weight

    Returns:
        W2 distance cost (0.0 if disabled)
    """

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

        weighted_w2 = alpha_G_got_w2 * (dist_sq_0 + dist_sq_1)

        return weighted_w2

    return jax.lax.cond(alpha_G_got_w2 == 0.0, w2_disabled, w2_enabled)


def _compute_inter_region_sync(
    displacements_w: tuple[Float[Array, "n_cp_x n_cp_y 2"], ...],
    displacements_t: tuple[Float[Array, "n_cp_x n_cp_y 2"], ...],
    all_ffd_parameters: tuple[FFDParameters, ...],
    inter_region_sync_indices: tuple[tuple[int, int], ...],
    inter_region_sync_points: tuple[Float[Array, "n_sync 2"], ...],
    alpha_H_inter_sync: Float[Array, ""],
) -> Float[Array, ""]:
    """Compute inter-region boundary synchronization costs.

    Args:
        displacements_w: W mapping displacements per region
        displacements_t: T mapping displacements per region
        all_ffd_parameters: FFD parameters for each region
        inter_region_sync_indices: Indices of regions to sync
        inter_region_sync_points: Points to synchronize at
        alpha_H_inter_sync: Inter-region sync weight

    Returns:
        Inter-region sync cost (0.0 if disabled)
    """

    def inter_sync_disabled():
        return jnp.array(0.0)

    def inter_sync_enabled():
        total_cost = 0.0
        total_n_pts = 0

        for (region_i, region_j), sync_points in zip(
            inter_region_sync_indices, inter_region_sync_points
        ):
            n_pts = sync_points.shape[0]
            if n_pts == 0:
                continue

            total_n_pts += n_pts

            params_i = all_ffd_parameters[region_i]
            params_j = all_ffd_parameters[region_j]

            mapped_w_i = ffd_mapping_2d(
                sync_points, displacements_w[region_i], params_i
            )
            mapped_t_i = ffd_mapping_2d(
                sync_points, displacements_t[region_i], params_i
            )

            mapped_w_j = ffd_mapping_2d(
                sync_points, displacements_w[region_j], params_j
            )
            mapped_t_j = ffd_mapping_2d(
                sync_points, displacements_t[region_j], params_j
            )

            diff_w = _squared_l2(mapped_w_i - mapped_w_j)
            diff_t = _squared_l2(mapped_t_i - mapped_t_j)

            total_cost += diff_w + diff_t

        if total_n_pts == 0:
            return jnp.array(0.0)

        return alpha_H_inter_sync * total_cost / total_n_pts

    return jax.lax.cond(
        alpha_H_inter_sync == 0.0, inter_sync_disabled, inter_sync_enabled
    )


def _compute_inter_region_jacobian_sync(
    displacements_w: tuple[Float[Array, "n_cp_x n_cp_y 2"], ...],
    displacements_t: tuple[Float[Array, "n_cp_x n_cp_y 2"], ...],
    all_ffd_parameters: tuple[FFDParameters, ...],
    inter_region_sync_indices: tuple[tuple[int, int], ...],
    inter_region_sync_points: tuple[Float[Array, "n_sync 2"], ...],
    alpha_I_inter_jac_sync: Float[Array, ""],
) -> Float[Array, ""]:
    """Compute inter-region Jacobian synchronization costs (C1 continuity).

    Args:
        displacements_w: W mapping displacements per region
        displacements_t: T mapping displacements per region
        all_ffd_parameters: FFD parameters for each region
        inter_region_sync_indices: Indices of regions to sync
        inter_region_sync_points: Shared boundary points per region pair
        alpha_I_inter_jac_sync: Inter-region Jacobian sync weight

    Returns:
        Inter-region Jacobian sync cost (0.0 if disabled)
    """

    def inter_jac_sync_disabled():
        return jnp.array(0.0)

    def inter_jac_sync_enabled():
        total_cost = 0.0
        total_n_pts = 0

        for (region_i, region_j), sync_points in zip(
            inter_region_sync_indices, inter_region_sync_points
        ):
            n_pts = sync_points.shape[0]
            if n_pts == 0:
                continue

            total_n_pts += n_pts

            params_i = all_ffd_parameters[region_i]
            params_j = all_ffd_parameters[region_j]

            jac_w_i = jacobian_matrix_2d(
                sync_points, displacements_w[region_i], params_i
            )
            jac_t_i = jacobian_matrix_2d(
                sync_points, displacements_t[region_i], params_i
            )

            jac_w_j = jacobian_matrix_2d(
                sync_points, displacements_w[region_j], params_j
            )
            jac_t_j = jacobian_matrix_2d(
                sync_points, displacements_t[region_j], params_j
            )

            # Penalize mismatch of Jacobian matrices across interfaces.
            diff_w = _squared_l2(jac_w_i - jac_w_j)
            diff_t = _squared_l2(jac_t_i - jac_t_j)

            total_cost += diff_w + diff_t

        if total_n_pts == 0:
            return jnp.array(0.0)

        return alpha_I_inter_jac_sync * total_cost / total_n_pts

    return jax.lax.cond(
        alpha_I_inter_jac_sync == 0.0,
        inter_jac_sync_disabled,
        inter_jac_sync_enabled,
    )


def _compute_point_registration(
    mapping_w: Callable,
    mapping_t: Callable,
    point_registration_points_0: Float[Array, "n_registration 2"],
    point_registration_points_1: Float[Array, "n_registration 2"],
    alpha_J_point_registration: Float[Array, ""],
) -> Float[Array, ""]:
    """Compute bidirectional point registration cost.

    Penalizes deviations between mapped source landmarks and target landmarks:
    ||W(pts_0) - pts_1||^2 + ||T(pts_1) - pts_0||^2, normalized by n_pts.

    Args:
        mapping_w: Forward mapping (field 0 -> field 1).
        mapping_t: Inverse mapping (field 1 -> field 0).
        point_registration_points_0: Source landmarks in field 0 domain.
        point_registration_points_1: Target landmarks in field 1 domain.
        alpha_J_point_registration: Weight for point registration cost.

    Returns:
        Point registration cost (0.0 if disabled or no points).
    """

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


def piecewise_ffd_cost_function(
    traced: FFDTracedParams,
    config: FFDStaticConfig,
) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
    """Pure functional piecewise FFD cost function.

    Matches the logic of the original PiecewiseFFDMapSyncCostFn:
    - Only the mapping handles the pieces (via piecewise_ffd_mapping_2d).
    - Interpolation uses per-region interpolators but evaluates on the
      **global mesh** points.
    - Alignment, jacobian barrier, mapping sync, and W2 are all computed
      once on the full global mesh.

    All hyperparameters are traced but frozen via stop_gradient, allowing
    barrier epsilon to be updated without recompilation.

    Args:
        traced: Traced parameters (displacements and hyperparameters)
        config: Static configuration (meshes, fields, FFD params, BC masks)

    Returns:
        Tuple of (total cost, auxiliary dict with cost components)
    """
    # Freeze hyperparameters via stop_gradient
    # This ensures gradients only flow to displacements
    alpha_A_algn = jax.lax.stop_gradient(traced.alpha_A_algn)
    alpha_B_bij = jax.lax.stop_gradient(traced.alpha_B_bij)
    alpha_C_jac = jax.lax.stop_gradient(traced.alpha_C_jac)
    alpha_D_cp_norm = jax.lax.stop_gradient(traced.alpha_D_cp_norm)
    alpha_G_got_w2 = jax.lax.stop_gradient(traced.alpha_G_got_w2)
    alpha_H_inter_sync = jax.lax.stop_gradient(traced.alpha_H_inter_sync)
    alpha_I_inter_jac_sync = jax.lax.stop_gradient(traced.alpha_I_inter_jac_sync)
    alpha_J_point_registration = jax.lax.stop_gradient(
        traced.alpha_J_point_registration
    )
    # Barrier config: epsilon is a traced array (frozen), fn_name is structural
    barrier_epsilon = jax.lax.stop_gradient(traced.barrier_config.barrier_epsilon)
    barrier_fn_name = traced.barrier_config.barrier_fn_name

    # Apply boundary conditions to displacements
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

    # Create piecewise mapping functions (partial applications)
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

    # Use the global mesh for all computations (like the old code)
    mesh_points = config.mesh.points
    n_mesh = mesh_points.shape[0]

    # Interpolate fields at deformed points on the global mesh using
    # per-region interpolators (matches old _u0/_u1 logic)
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

    # Compute all cost components on the global mesh (not per-region)
    alignment_0, alignment_1 = _compute_alignment(
        u0_comp_w, u1_comp_t, config.u0_values, config.u1_values, alpha_A_algn
    )

    jacobian_barrier_costs = _compute_jacobian_barrier(
        displacements_w,
        displacements_t,
        mesh_points,
        config.all_ffd_parameters,
        alpha_C_jac,
        barrier_epsilon,
        barrier_fn_name,
        n_mesh,
    )

    jacobian_violations = _compute_jacobian_violations(
        displacements_w,
        displacements_t,
        mesh_points,
        config.all_ffd_parameters,
        alpha_C_jac,
    )

    mapping_sync_costs = _compute_mapping_sync(
        mapping_w, mapping_t, mesh_points, alpha_B_bij, n_mesh
    )

    control_reg_costs = _compute_control_reg(
        displacements_w, displacements_t, alpha_D_cp_norm
    )

    w2_costs = _compute_w2_cost(
        u0_comp_w,
        u1_comp_t,
        config.u0_values,
        config.u1_values,
        mesh_points,
        alpha_G_got_w2,
    )

    inter_region_sync_costs = _compute_inter_region_sync(
        displacements_w,
        displacements_t,
        config.all_ffd_parameters,
        config.inter_region_sync_indices,
        config.inter_region_sync_points,
        alpha_H_inter_sync,
    )

    inter_region_jac_sync_costs = _compute_inter_region_jacobian_sync(
        displacements_w,
        displacements_t,
        config.all_ffd_parameters,
        config.inter_region_sync_indices,
        config.inter_region_sync_points,
        alpha_I_inter_jac_sync,
    )

    point_registration_costs = _compute_point_registration(
        mapping_w,
        mapping_t,
        config.point_registration_points_0,
        config.point_registration_points_1,
        alpha_J_point_registration,
    )

    # Compute total cost (like old code: alignment + barrier_cost)
    barrier_cost = (
        jacobian_barrier_costs
        + mapping_sync_costs
        + control_reg_costs
        + w2_costs
        + inter_region_sync_costs
        + inter_region_jac_sync_costs
        + point_registration_costs
    )
    total_cost = alignment_0 + alignment_1 + barrier_cost

    # Keep the objective finite for line-search trial points that produce
    # invalid mappings; this avoids poisoning LBFGS state with NaNs/Infs.
    _warn_if_non_finite("total_cost", total_cost)
    total_cost = jnp.nan_to_num(
        total_cost,
        nan=NON_FINITE_PENALTY,
        posinf=NON_FINITE_PENALTY,
        neginf=NON_FINITE_PENALTY,
    )

    _warn_if_non_finite("alignment_0", alignment_0)
    _warn_if_non_finite("alignment_1", alignment_1)
    _warn_if_non_finite("jacobian_barrier", jacobian_barrier_costs)
    _warn_if_non_finite("mapping_sync", mapping_sync_costs)
    _warn_if_non_finite("control_reg", control_reg_costs)
    _warn_if_non_finite("w2", w2_costs)
    _warn_if_non_finite("inter_region_sync", inter_region_sync_costs)
    _warn_if_non_finite("inter_region_jac_sync", inter_region_jac_sync_costs)
    _warn_if_non_finite("point_registration", point_registration_costs)

    aux = {
        "alignment_0": jnp.nan_to_num(
            alignment_0,
            nan=NON_FINITE_PENALTY,
            posinf=NON_FINITE_PENALTY,
            neginf=NON_FINITE_PENALTY,
        ),
        "alignment_1": jnp.nan_to_num(
            alignment_1,
            nan=NON_FINITE_PENALTY,
            posinf=NON_FINITE_PENALTY,
            neginf=NON_FINITE_PENALTY,
        ),
        "jacobian_barrier": jnp.nan_to_num(
            jacobian_barrier_costs,
            nan=NON_FINITE_PENALTY,
            posinf=NON_FINITE_PENALTY,
            neginf=NON_FINITE_PENALTY,
        ),
        "mapping_sync": jnp.nan_to_num(
            mapping_sync_costs,
            nan=NON_FINITE_PENALTY,
            posinf=NON_FINITE_PENALTY,
            neginf=NON_FINITE_PENALTY,
        ),
        "control_reg": jnp.nan_to_num(
            control_reg_costs,
            nan=NON_FINITE_PENALTY,
            posinf=NON_FINITE_PENALTY,
            neginf=NON_FINITE_PENALTY,
        ),
        "w2": jnp.nan_to_num(
            w2_costs,
            nan=NON_FINITE_PENALTY,
            posinf=NON_FINITE_PENALTY,
            neginf=NON_FINITE_PENALTY,
        ),
        "inter_region_sync": jnp.nan_to_num(
            inter_region_sync_costs,
            nan=NON_FINITE_PENALTY,
            posinf=NON_FINITE_PENALTY,
            neginf=NON_FINITE_PENALTY,
        ),
        "inter_region_jac_sync": jnp.nan_to_num(
            inter_region_jac_sync_costs,
            nan=NON_FINITE_PENALTY,
            posinf=NON_FINITE_PENALTY,
            neginf=NON_FINITE_PENALTY,
        ),
        "point_registration": jnp.nan_to_num(
            point_registration_costs,
            nan=NON_FINITE_PENALTY,
            posinf=NON_FINITE_PENALTY,
            neginf=NON_FINITE_PENALTY,
        ),
        "jacobian_violations": jacobian_violations,
    }

    return total_cost, aux
