"""Wasserstein-like barycenter cost with piecewise FFD mappings."""

from __future__ import annotations

from functools import partial
import logging
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.interpolate import RegularGridInterpolator
from jaxtyping import Float

from phdtruel.mappings.cost_functional.baricenter_ffd.types import (
    BariFFDStaticConfig,
    BariFFDTracedParams,
)
from phdtruel.mappings.cost_functional.utils import (
    BarrierFnName,
    get_barrier_fn,
    impose_control_displacements,
)
from phdtruel.mappings.ffd import (
    FFDParameters,
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


def _is_positive_weight(weight: Float[Array, ""]) -> Array:
    """Return True when a scalar weight is strictly positive."""
    return jnp.greater(weight, jnp.asarray(0.0, dtype=weight.dtype))


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


def _normalize_gamma(gamma: Float[Array, " n_fields"]) -> Float[Array, " n_fields"]:
    """Normalize non-negative gamma weights and warn when renormalization occurs."""
    gamma_clean = jnp.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)
    gamma_pos = jnp.clip(gamma_clean, a_min=0.0)

    sum_gamma = jnp.sum(gamma_pos)
    eps = jnp.finfo(gamma_pos.dtype).eps
    n_fields = gamma_pos.shape[0]

    gamma_normalized = jax.lax.cond(
        sum_gamma > eps,
        lambda _: gamma_pos / sum_gamma,
        lambda _: jnp.full_like(gamma_pos, 1.0 / n_fields),
        operand=None,
    )

    needs_warning = (
        jnp.any(~jnp.isfinite(gamma))
        | jnp.any(gamma < 0.0)
        | (jnp.abs(jnp.sum(gamma) - 1.0) > 1.0e-6)
    )

    def _callback(flag: Array) -> None:
        if bool(flag):
            logger.warning(
                "Gamma weights were normalized internally to a valid simplex."
            )

    jax.debug.callback(_callback, needs_warning)

    return gamma_normalized


def _interpolate_field_piecewise(
    points: Float[Array, "n_points 2"],
    region_meshes: tuple,
    field_values_per_region: tuple[Float[Array, " n_region"], ...],
) -> Float[Array, " n_points"]:
    """Interpolate a field on deformed points using per-region interpolators."""
    result = jnp.zeros(points.shape[0])

    for mesh, field_vals in zip(region_meshes, field_values_per_region):
        shape = tuple(ax.shape[0] for ax in mesh.axes)
        interp = RegularGridInterpolator(
            mesh.axes,
            field_vals.reshape(shape),
            fill_value=jnp.nan,
        )
        region_values = interp(points)
        region_values_clean = jnp.nan_to_num(region_values, nan=0.0)
        result = jnp.where(jnp.isnan(region_values), result, region_values_clean)

    return result


def _weighted_moments(
    points: Float[Array, "n_points 2"],
    weights: Float[Array, " n_points"],
) -> tuple[Float[Array, " 2"], Float[Array, " 2 2"]]:
    """Compute weighted Gaussian moments with finite-safe weights."""
    w = jnp.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    w = jnp.clip(w, a_min=0.0)
    w = jnp.ravel(w)  # Ensure 1D
    eps = jnp.finfo(points.dtype).eps

    # Add small uniform weight to prevent all-zero weights
    w = w + eps
    w_sum = jnp.maximum(jnp.sum(w), eps)

    mu = jnp.sum(points * w[:, None], axis=0) / w_sum

    centered = points - mu[None, :]
    cov = jnp.einsum("n,ni,nj->ij", w, centered, centered) / w_sum
    # Add larger regularization for numerical stability in eigendecomposition
    cov = cov + 1e-8 * jnp.eye(points.shape[1], dtype=points.dtype)

    return mu, cov


def _compute_identity_regularization(
    mapped_points_per_field: tuple[Float[Array, "n_points 2"], ...],
    mesh_points: Float[Array, "n_points 2"],
    gamma: Float[Array, " n_fields"],
    alpha_A_w_id: Float[Array, ""],
) -> Float[Array, ""]:
    """Compute alpha_A * sum_i gamma_i ||W_i - Id||^2 / n_fields."""
    zero_scalar = jnp.asarray(0.0, dtype=mesh_points.dtype)
    n_fields = len(mapped_points_per_field)

    def _compute(_: None) -> Float[Array, ""]:
        n_mesh = jnp.asarray(mesh_points.shape[0], dtype=mesh_points.dtype)
        total = jnp.asarray(0.0, dtype=mesh_points.dtype)

        for i, mapped in enumerate(mapped_points_per_field):
            total = total + gamma[i] * _squared_l2(mapped - mesh_points)

        return alpha_A_w_id * total / (n_mesh * n_fields)

    return jax.lax.cond(
        _is_positive_weight(alpha_A_w_id),
        _compute,
        lambda _: zero_scalar,
        operand=None,
    )


def _compute_log_barrier_jacobian(
    displacements_per_field: tuple[tuple[Float[Array, "n_cp_x n_cp_y 2"], ...], ...],
    mesh_points: Float[Array, "n_points 2"],
    all_ffd_parameters: tuple[FFDParameters, ...],
    alpha_C_log_barrier_jac: Float[Array, ""],
    barrier_epsilon: Float[Array, ""],
    barrier_fn_name: BarrierFnName,
) -> Float[Array, ""]:
    """Compute alpha_C * sum_i ||log_barrier(det(Jac(W_i)))||^2 / n_fields.

    This is a log-barrier on the Jacobian determinant to enforce positivity,
    different from the standard Jacobian barrier which penalizes negative values.
    """
    n_fields = len(displacements_per_field)

    def _compute(_: None) -> Float[Array, ""]:
        barrier_fn = get_barrier_fn(barrier_fn_name)
        total = jnp.asarray(0.0, dtype=mesh_points.dtype)

        for field_disps in displacements_per_field:
            jacobian_dets = jnp.asarray(
                piecewise_jacobian_det_2d(mesh_points, field_disps, all_ffd_parameters)
            )
            # Apply barrier function to the determinant
            barrier_vals = barrier_fn(jacobian_dets, barrier_epsilon)
            total = total + _squared_l2(barrier_vals)

        return alpha_C_log_barrier_jac * total / n_fields

    zero_scalar = jnp.asarray(0.0, dtype=mesh_points.dtype)
    return jax.lax.cond(
        _is_positive_weight(alpha_C_log_barrier_jac),
        _compute,
        lambda _: zero_scalar,
        operand=None,
    )


def _compute_pairwise_alignment(
    aligned_fields: tuple[Float[Array, " n_points"], ...],
    gamma: Float[Array, " n_fields"],
    alpha_B_pairwise_align: Float[Array, ""],
) -> Float[Array, ""]:
    """Compute alpha_B * sum_{i,j} gamma_i * gamma_j * ||u_i o W_i - u_j o W_j||^2 / n_fields."""
    zero_scalar = jnp.asarray(0.0, dtype=alpha_B_pairwise_align.dtype)
    n_fields = len(aligned_fields)
    if n_fields < 2:
        return zero_scalar

    def _compute(_: None) -> Float[Array, ""]:
        pairwise = jnp.asarray(0.0, dtype=aligned_fields[0].dtype)
        for i in range(n_fields):
            for j in range(n_fields):
                pairwise = pairwise + gamma[i] * gamma[j] * _squared_l2(
                    aligned_fields[i] - aligned_fields[j]
                )

        return alpha_B_pairwise_align * pairwise / n_fields

    return jax.lax.cond(
        _is_positive_weight(alpha_B_pairwise_align),
        _compute,
        lambda _: zero_scalar,
        operand=None,
    )


def _compute_pairwise_w2(
    mapped_moments: tuple[tuple[Float[Array, " 2"], Float[Array, " 2 2"]], ...],
    gamma: Float[Array, " n_fields"],
    alpha_D_pairwise_w2: Float[Array, ""],
) -> Float[Array, ""]:
    """Compute alpha_D * sum_{i,j} gamma_i * gamma_j * W2(g_i_mapped, g_j_mapped)^2 / n_fields."""
    zero_scalar = jnp.asarray(0.0, dtype=alpha_D_pairwise_w2.dtype)
    n_fields = len(mapped_moments)
    if n_fields < 2:
        return zero_scalar

    def _compute(_: None) -> Float[Array, ""]:
        total = jnp.asarray(0.0, dtype=mapped_moments[0][0].dtype)
        for i in range(n_fields):
            mu_i, cov_i = mapped_moments[i]
            for j in range(n_fields):
                mu_j, cov_j = mapped_moments[j]
                dist_sq = wasserstein_distance_analytical_jax(mu_i, mu_j, cov_i, cov_j)
                total = total + (gamma[i] + gamma[j]) * dist_sq

        return alpha_D_pairwise_w2 * total / n_fields

    return jax.lax.cond(
        _is_positive_weight(alpha_D_pairwise_w2),
        _compute,
        lambda _: zero_scalar,
        operand=None,
    )


def _compute_barycenter_w2(
    mapped_moments: tuple[tuple[Float[Array, " 2"], Float[Array, " 2 2"]], ...],
    original_moments: tuple[tuple[Float[Array, " 2"], Float[Array, " 2 2"]], ...],
    gamma: Float[Array, " n_fields"],
    alpha_E_bary_w2: Float[Array, ""],
) -> Float[Array, ""]:
    """Compute alpha_E * sum_i W2(g_i_mapped, g_bar_original)^2 / n_fields.

    The barycenter g_bar is computed from the original (unmapped) field Gaussians,
    while each g_i is the Gaussian fitted on the mapped field u_i o W_i.
    """
    zero_scalar = jnp.asarray(0.0, dtype=alpha_E_bary_w2.dtype)
    n_fields = len(mapped_moments)
    if n_fields == 0:
        return zero_scalar

    is_positive = _is_positive_weight(alpha_E_bary_w2)

    def _compute(_: None) -> Float[Array, ""]:
        # Compute barycenter from ORIGINAL field moments (not mapped)
        mus_orig = jnp.stack([mu for mu, _ in original_moments], axis=0)
        covs_orig = jnp.stack([cov for _, cov in original_moments], axis=0)

        mu_bar = jnp.sum(gamma[:, None] * mus_orig, axis=0)
        cov_bar = jnp.sum(gamma[:, None, None] * covs_orig, axis=0)

        # Compute W2 distance from each MAPPED field Gaussian to the original barycenter
        total = jnp.asarray(0.0, dtype=mus_orig.dtype)
        for mu_i_mapped, cov_i_mapped in mapped_moments:
            total = total + wasserstein_distance_analytical_jax(
                mu_i_mapped, mu_bar, cov_i_mapped, cov_bar
            )

        return alpha_E_bary_w2 * total / n_fields

    def _zero(_: None) -> Float[Array, ""]:
        return zero_scalar

    return jax.lax.cond(
        is_positive,
        _compute,
        _zero,
        operand=None,
    )


def wasserstein_barycenter_cost_function(
    traced: BariFFDTracedParams,
    config: BariFFDStaticConfig,
) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
    """Compute Wasserstein-like barycenter objective for N mapped fields.

    Implements the formulation:
    - alpha_A: sum_i gamma_i ||W_i - Id||^2 / n_fields
    - alpha_B: sum_{i,j} gamma_i * gamma_j * ||u_i o W_i - u_j o W_j||^2 / n_fields
    - alpha_C: sum_i ||log_barrier(det(Jac(W_i)))||^2 / n_fields
    - alpha_D: sum_{i,j} gamma_i * gamma_j * W2(g_i_mapped, g_j_mapped)^2 / n_fields
    - alpha_E: sum_i W2(g_i_mapped, g_bar_original)^2 / n_fields
    """
    alpha_A_w_id = jax.lax.stop_gradient(traced.alpha_A_w_id)
    alpha_B_pairwise_align = jax.lax.stop_gradient(traced.alpha_B_pairwise_align)
    alpha_C_log_barrier_jac = jax.lax.stop_gradient(traced.alpha_C_log_barrier_jac)
    alpha_D_pairwise_w2 = jax.lax.stop_gradient(traced.alpha_D_pairwise_w2)
    alpha_E_bary_w2 = jax.lax.stop_gradient(traced.alpha_E_bary_w2)
    log_barrier_epsilon = jax.lax.stop_gradient(
        traced.log_barrier_config.barrier_epsilon
    )
    log_barrier_fn_name = traced.log_barrier_config.barrier_fn_name

    if len(traced.displacements_w) != len(config.all_u_values):
        raise ValueError(
            "Number of displacement blocks must match number of input fields."
        )

    gamma = _normalize_gamma(config.gamma)
    mesh_points = config.mesh.points
    n_fields = len(config.all_u_values)

    # Compute constrained displacements for all fields (needed for alpha_C)
    constrained_displacements = tuple(
        tuple(
            impose_control_displacements(disp, mask)
            for disp, mask in zip(field_disps, config.all_imposed_displacements_masks)
        )
        for field_disps in traced.displacements_w
    )

    requires_mapping = (
        _is_positive_weight(alpha_A_w_id)
        | _is_positive_weight(alpha_B_pairwise_align)
        | _is_positive_weight(alpha_D_pairwise_w2)
        | _is_positive_weight(alpha_E_bary_w2)
    )

    def _compute_mapped_points(_: None) -> tuple[Float[Array, "n_points 2"], ...]:
        mapping_w = tuple(
            partial(
                piecewise_ffd_mapping_2d,
                all_control_points_displacement=field_disps,
                all_params=config.all_ffd_parameters,
            )
            for field_disps in constrained_displacements
        )

        return tuple(jnp.asarray(map_i(mesh_points)) for map_i in mapping_w)

    mapped_points_per_field = jax.lax.cond(
        requires_mapping,
        _compute_mapped_points,
        lambda _: tuple(mesh_points for _ in range(n_fields)),
        operand=None,
    )

    requires_alignment = (
        _is_positive_weight(alpha_B_pairwise_align)
        | _is_positive_weight(alpha_D_pairwise_w2)
        | _is_positive_weight(alpha_E_bary_w2)
    )

    n_points = mesh_points.shape[0]
    aligned_fields = jax.lax.cond(
        requires_alignment,
        lambda _: tuple(
            _interpolate_field_piecewise(
                mapped_points_per_field[i],
                config.region_meshes,
                config.all_u_per_region[i],
            )
            for i in range(n_fields)
        ),
        lambda _: tuple(
            jnp.zeros(n_points, dtype=mesh_points.dtype) for _ in range(n_fields)
        ),
        operand=None,
    )

    # Compute moments for both mapped and original fields
    requires_moments = _is_positive_weight(alpha_D_pairwise_w2) | _is_positive_weight(
        alpha_E_bary_w2
    )

    def _compute_moments_pair(
        _: None,
    ) -> tuple[
        tuple[tuple[Float[Array, " 2"], Float[Array, " 2 2"]], ...],
        tuple[tuple[Float[Array, " 2"], Float[Array, " 2 2"]], ...],
    ]:
        """Compute moments for both mapped and original fields."""
        mapped_moms = tuple(
            _weighted_moments(mesh_points, aligned) for aligned in aligned_fields
        )
        original_moms = tuple(
            _weighted_moments(mesh_points, u_vals) for u_vals in config.all_u_values
        )
        return mapped_moms, original_moms

    def _zero_moments_pair(
        _: None,
    ) -> tuple[
        tuple[tuple[Float[Array, " 2"], Float[Array, " 2 2"]], ...],
        tuple[tuple[Float[Array, " 2"], Float[Array, " 2 2"]], ...],
    ]:
        zero_mom = (
            jnp.zeros((2,), dtype=mesh_points.dtype),
            jnp.eye(2, dtype=mesh_points.dtype),
        )
        return (
            tuple(zero_mom for _ in range(n_fields)),
            tuple(zero_mom for _ in range(n_fields)),
        )

    mapped_moments, original_moments = jax.lax.cond(
        requires_moments,
        _compute_moments_pair,
        _zero_moments_pair,
        operand=None,
    )

    w_identity = _compute_identity_regularization(
        mapped_points_per_field,
        mesh_points,
        gamma,
        alpha_A_w_id,
    )
    pairwise_alignment = _compute_pairwise_alignment(
        aligned_fields,
        gamma,
        alpha_B_pairwise_align,
    )

    log_barrier_jac = _compute_log_barrier_jacobian(
        constrained_displacements,
        mesh_points,
        config.all_ffd_parameters,
        alpha_C_log_barrier_jac,
        log_barrier_epsilon,
        log_barrier_fn_name,
    )

    pairwise_w2 = _compute_pairwise_w2(
        mapped_moments,
        gamma,
        alpha_D_pairwise_w2,
    )
    barycenter_w2 = _compute_barycenter_w2(
        mapped_moments,
        original_moments,
        gamma,
        alpha_E_bary_w2,
    )

    total_cost = (
        w_identity + pairwise_alignment + log_barrier_jac + pairwise_w2 + barycenter_w2
    )

    _warn_if_non_finite("total_cost", total_cost)
    _warn_if_non_finite("w_identity", w_identity)
    _warn_if_non_finite("pairwise_alignment", pairwise_alignment)
    _warn_if_non_finite("log_barrier_jac", log_barrier_jac)
    _warn_if_non_finite("pairwise_w2", pairwise_w2)
    _warn_if_non_finite("barycenter_w2", barycenter_w2)

    total_cost = jnp.nan_to_num(
        total_cost,
        nan=NON_FINITE_PENALTY,
        posinf=NON_FINITE_PENALTY,
        neginf=NON_FINITE_PENALTY,
    )

    aux = {
        "w_identity": jnp.nan_to_num(
            w_identity,
            nan=NON_FINITE_PENALTY,
            posinf=NON_FINITE_PENALTY,
            neginf=NON_FINITE_PENALTY,
        ),
        "pairwise_alignment": jnp.nan_to_num(
            pairwise_alignment,
            nan=NON_FINITE_PENALTY,
            posinf=NON_FINITE_PENALTY,
            neginf=NON_FINITE_PENALTY,
        ),
        "log_barrier_jac": jnp.nan_to_num(
            log_barrier_jac,
            nan=NON_FINITE_PENALTY,
            posinf=NON_FINITE_PENALTY,
            neginf=NON_FINITE_PENALTY,
        ),
        "pairwise_w2": jnp.nan_to_num(
            pairwise_w2,
            nan=NON_FINITE_PENALTY,
            posinf=NON_FINITE_PENALTY,
            neginf=NON_FINITE_PENALTY,
        ),
        "barycenter_w2": jnp.nan_to_num(
            barycenter_w2,
            nan=NON_FINITE_PENALTY,
            posinf=NON_FINITE_PENALTY,
            neginf=NON_FINITE_PENALTY,
        ),
    }

    return total_cost, aux
