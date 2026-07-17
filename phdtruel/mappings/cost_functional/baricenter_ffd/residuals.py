"""Per-term residual vectors for barycenter FFD (Gauss-Newton / natural gradient).

Each returned vector ``r_k`` satisfies ``||r_k||^2 = cost_k`` for the
corresponding squared term in :mod:`cost_fn`.  The W2 (Gaussian OT) terms
are omitted because they are not plain sum-of-squares residuals.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Float

from phdtruel.mappings.cost_functional.baricenter_ffd.cost_fn import (
    _interpolate_field_piecewise,
    _normalize_gamma,
)
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


def _identity_residuals(
    mapped_points_per_field: tuple[Float[jax.Array, "n_points 2"], ...],
    mesh_points: Float[jax.Array, "n_points 2"],
    gamma: Float[jax.Array, " n_fields"],
    alpha_A_w_id: Float[jax.Array, ""],
    n_mesh: int,
    n_fields: int,
) -> dict[str, Float[jax.Array, " n"]]:
    def disabled() -> dict[str, Float[jax.Array, " n"]]:
        z = jnp.zeros((n_mesh * 2,), dtype=mesh_points.dtype)
        return {f"identity_{i}": z for i in range(n_fields)}

    def enabled() -> dict[str, Float[jax.Array, " n"]]:
        out: dict[str, Float[jax.Array, " n"]] = {}
        scale_base = jnp.sqrt(alpha_A_w_id / (n_mesh * n_fields))
        for i, mapped in enumerate(mapped_points_per_field):
            scale = scale_base * jnp.sqrt(gamma[i])
            out[f"identity_{i}"] = scale * (mapped - mesh_points).ravel()
        return out

    return jax.lax.cond(alpha_A_w_id == 0.0, disabled, enabled)


def _pairwise_alignment_residuals(
    aligned_fields: tuple[Float[jax.Array, " n_points"], ...],
    gamma: Float[jax.Array, " n_fields"],
    alpha_B_pairwise_align: Float[jax.Array, ""],
    n_fields: int,
) -> dict[str, Float[jax.Array, " n"]]:
    if n_fields < 2:
        return {}

    out: dict[str, Float[jax.Array, " n"]] = {}
    scale_base = jnp.where(
        alpha_B_pairwise_align == 0.0,
        0.0,
        jnp.sqrt(alpha_B_pairwise_align / n_fields),
    )
    for i in range(n_fields):
        for j in range(n_fields):
            scale = scale_base * jnp.sqrt(gamma[i] * gamma[j])
            out[f"pairwise_align_{i}_{j}"] = scale * (
                aligned_fields[i] - aligned_fields[j]
            ).ravel()
    return out


def _log_barrier_jacobian_residuals(
    displacements_per_field: tuple[
        tuple[Float[jax.Array, "n_cp_x n_cp_y 2"], ...], ...
    ],
    mesh_points: Float[jax.Array, "n_points 2"],
    all_ffd_parameters: tuple[FFDParameters, ...],
    alpha_C_log_barrier_jac: Float[jax.Array, ""],
    barrier_epsilon: Float[jax.Array, ""],
    barrier_fn_name: BarrierFnName,
    n_fields: int,
    n_mesh: int,
) -> dict[str, Float[jax.Array, " n_points"]]:
    def disabled() -> dict[str, Float[jax.Array, " n_points"]]:
        z = jnp.zeros((n_mesh,), dtype=mesh_points.dtype)
        return {f"log_barrier_jac_{i}": z for i in range(n_fields)}

    def enabled() -> dict[str, Float[jax.Array, " n_points"]]:
        barrier_fn = get_barrier_fn(barrier_fn_name)
        scale = jnp.sqrt(alpha_C_log_barrier_jac / n_fields)
        out: dict[str, Float[jax.Array, " n_points"]] = {}
        for i, field_disps in enumerate(displacements_per_field):
            jacobian_dets = jnp.asarray(
                piecewise_jacobian_det_2d(mesh_points, field_disps, all_ffd_parameters)
            )
            barrier_vals = barrier_fn(jacobian_dets, barrier_epsilon)
            out[f"log_barrier_jac_{i}"] = scale * barrier_vals.ravel()
        return out

    return jax.lax.cond(alpha_C_log_barrier_jac == 0.0, disabled, enabled)


def wasserstein_barycenter_residual_function(
    traced: BariFFDTracedParams,
    config: BariFFDStaticConfig,
) -> dict[str, Float[jax.Array, " n"]]:
    """Return unsquared residual vectors for each barycenter cost term.

    W2 terms (``alpha_D_pairwise_w2``, ``alpha_E_bary_w2``) are excluded:
    they are not sum-of-squares residuals.

    Args:
        traced: Traced barycenter parameters.
        config: Static barycenter configuration.

    Returns:
        Dictionary mapping term names to 1-D residual vectors.
    """
    alpha_A_w_id = jax.lax.stop_gradient(traced.alpha_A_w_id)
    alpha_B_pairwise_align = jax.lax.stop_gradient(traced.alpha_B_pairwise_align)
    alpha_C_log_barrier_jac = jax.lax.stop_gradient(traced.alpha_C_log_barrier_jac)
    barrier_epsilon = jax.lax.stop_gradient(traced.log_barrier_config.barrier_epsilon)
    barrier_fn_name = traced.log_barrier_config.barrier_fn_name

    if len(traced.displacements_w) != len(config.all_u_values):
        raise ValueError(
            "Number of displacement blocks must match number of input fields."
        )

    gamma = _normalize_gamma(config.gamma)
    mesh_points = config.mesh.points
    n_fields = len(config.all_u_values)
    n_mesh = mesh_points.shape[0]

    constrained_displacements = tuple(
        tuple(
            impose_control_displacements(disp, mask)
            for disp, mask in zip(field_disps, config.all_imposed_displacements_masks)
        )
        for field_disps in traced.displacements_w
    )

    requires_mapping = (alpha_A_w_id != 0.0) | (alpha_B_pairwise_align != 0.0)

    def _compute_mapped_points(_: None) -> tuple[Float[jax.Array, "n_points 2"], ...]:
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

    requires_alignment = alpha_B_pairwise_align != 0.0

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
            jnp.zeros(n_mesh, dtype=mesh_points.dtype) for _ in range(n_fields)
        ),
        operand=None,
    )

    residuals: dict[str, Float[jax.Array, " n"]] = {}
    residuals.update(
        _identity_residuals(
            mapped_points_per_field,
            mesh_points,
            gamma,
            alpha_A_w_id,
            n_mesh,
            n_fields,
        )
    )
    residuals.update(
        _pairwise_alignment_residuals(
            aligned_fields,
            gamma,
            alpha_B_pairwise_align,
            n_fields,
        )
    )
    residuals.update(
        _log_barrier_jacobian_residuals(
            constrained_displacements,
            mesh_points,
            config.all_ffd_parameters,
            alpha_C_log_barrier_jac,
            barrier_epsilon,
            barrier_fn_name,
            n_fields,
            n_mesh,
        )
    )

    return residuals
