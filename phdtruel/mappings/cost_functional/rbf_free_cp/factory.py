"""Factory function for creating RBF free-CP cost function and initial parameters."""

from __future__ import annotations

from typing import Any, Callable, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float

from phdtruel.mappings.cost_functional.rbf.cost_fn import _rbf_mapping
from phdtruel.mappings.cost_functional.rbf_free_cp.cost_fn import (
    _build_mapping_rbf_kwargs,
)
from phdtruel.mappings.cost_functional.rbf.factory import _resolve_field_interp_kind
from phdtruel.mappings.cost_functional.rbf_free_cp.cost_fn import (
    _build_full_centers,
    rbf_free_cp_cost_function,
)
from phdtruel.mappings.cost_functional.rbf_free_cp.types import (
    FieldInterpKind,
    RBFFreeStaticConfig,
    RBFFreeTracedParams,
)
from phdtruel.mappings.cost_functional.ffd.types import BarrierFnConfig
from phdtruel.mappings.cost_functional.types import MeshJaxed
from phdtruel.mappings.cost_functional.utils import BarrierFnName
from phdtruel.mappings.mappings import Mapping


class RBFFreeFactoryResult(NamedTuple):
    """Result tuple from the free-CP factory function.

    Attributes:
        cost_fn: The cost function (traced params only).
        init_params: Initial traced parameters.
        make_mappings: Function to create W and T mappings from optimized params.
        get_displacements: Extract displacements from solved params.
        get_centers_and_displacements: Extract full centers and displacements.
    """

    cost_fn: Callable
    init_params: RBFFreeTracedParams
    make_mappings: Callable[[RBFFreeTracedParams], tuple[Callable, Callable]]
    get_displacements: Callable[
        [Any],
        tuple[Float[Array, "n_c 2"], Float[Array, "n_c 2"]],
    ]
    get_centers_and_displacements: Callable[
        [Any],
        tuple[
            Float[Array, "n_c 2"],
            Float[Array, "n_c 2"],
            Float[Array, "n_c 2"],
            Float[Array, "n_c 2"],
        ],
    ]


def _build_static_config(
    mesh: MeshJaxed,
    u0_values: Float[Array, " n"],
    u1_values: Float[Array, " n"],
    boundary_centers: Float[Array, "n_b 2"],
    interior_centers: Float[Array, "n_i 2"],
    imposed_disp_points: Float[Array, "n_bc 2"],
    imposed_disp_targets: Float[Array, "n_bc 2"],
    point_registration_points_0: Float[Array, "n_registration 2"],
    point_registration_points_1: Float[Array, "n_registration 2"],
    rbf_kernel: str,
    rbf_neighbors: int | None,
    rbf_epsilon: float | None,
    rbf_smoothing: float,
    rbf_degree: int | None,
    field_interp_kind: FieldInterpKind,
    field_linear_frozen_points: bool,
    field_rbf_neighbors: int,
    field_rbf_kernel: str | None,
    alpha_A_algn: float,
    alpha_B_bij: float,
    alpha_C_jac: float,
    alpha_D_cp_norm: float,
    alpha_E_imposed_disp: float,
    alpha_F_cp_pos_reg: float,
    alpha_G_got_w2: float,
    alpha_J_point_registration: float,
    record_cp_history: bool,
) -> RBFFreeStaticConfig:
    """Build the static configuration for RBF free-CP cost function."""
    enable_w2_cost = alpha_G_got_w2 > 0.0
    enable_point_registration = (
        alpha_J_point_registration > 0.0 and point_registration_points_0.shape[0] > 0
    )
    interior_centers_jax = jnp.asarray(interior_centers)

    return RBFFreeStaticConfig(
        mesh=mesh,
        u0_values=u0_values,
        u1_values=u1_values,
        boundary_centers=boundary_centers,
        n_interior=int(interior_centers_jax.shape[0]),
        imposed_disp_points=imposed_disp_points,
        imposed_disp_targets=imposed_disp_targets,
        point_registration_points_0=point_registration_points_0,
        point_registration_points_1=point_registration_points_1,
        initial_interior_centers_w=interior_centers_jax,
        initial_interior_centers_t=interior_centers_jax,
        rbf_kernel=rbf_kernel,
        rbf_neighbors=rbf_neighbors,
        rbf_epsilon=rbf_epsilon,
        rbf_smoothing=rbf_smoothing,
        rbf_degree=rbf_degree,
        field_interp_kind=field_interp_kind,
        field_linear_frozen_points=field_linear_frozen_points,
        field_rbf_neighbors=field_rbf_neighbors,
        field_rbf_kernel=field_rbf_kernel or rbf_kernel,
        alpha_A_algn=alpha_A_algn,
        alpha_B_bij=alpha_B_bij,
        alpha_C_jac=alpha_C_jac,
        alpha_D_cp_norm=alpha_D_cp_norm,
        alpha_E_imposed_disp=alpha_E_imposed_disp,
        alpha_F_cp_pos_reg=alpha_F_cp_pos_reg,
        alpha_G_got_w2=alpha_G_got_w2,
        alpha_J_point_registration=alpha_J_point_registration,
        enable_w2_cost=enable_w2_cost,
        enable_point_registration=enable_point_registration,
        record_cp_history=record_cp_history,
    )


def _build_initial_params(
    boundary_centers: Float[Array, "n_b 2"],
    interior_centers: Float[Array, "n_i 2"],
    initial_mapping_function_w: Mapping,
    initial_mapping_function_t: Mapping,
    initialisation_noise_factor: float,
    alpha_A_algn: float,
    alpha_B_bij: float,
    alpha_C_jac: float,
    alpha_D_cp_norm: float,
    alpha_E_imposed_disp: float,
    alpha_F_cp_pos_reg: float,
    alpha_G_got_w2: float,
    alpha_J_point_registration: float,
    barrier_epsilon: float,
    barrier_fn_name: BarrierFnName = "logarithmic",
) -> RBFFreeTracedParams:
    """Build initial traced parameters for free-CP optimization."""
    boundary_np = np.asarray(boundary_centers)
    interior_np = np.asarray(interior_centers)
    full_centers_np = np.vstack([boundary_np, interior_np])

    mapped_w = np.asarray(initial_mapping_function_w(full_centers_np))
    mapped_t = np.asarray(initial_mapping_function_t(full_centers_np))
    disp_w = jnp.array(mapped_w - full_centers_np)
    disp_t = jnp.array(mapped_t - full_centers_np)

    key = jax.random.PRNGKey(42)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    noise_w = initialisation_noise_factor * jax.random.normal(subkey1, disp_w.shape)
    noise_t = initialisation_noise_factor * jax.random.normal(subkey2, disp_t.shape)
    disp_w = disp_w + noise_w
    disp_t = disp_t + noise_t

    return RBFFreeTracedParams(
        interior_centers_w=jnp.array(interior_np),
        displacements_w=disp_w,
        interior_centers_t=jnp.array(interior_np),
        displacements_t=disp_t,
        barrier_config=BarrierFnConfig(
            barrier_fn_name=barrier_fn_name,
            barrier_epsilon=jnp.array(barrier_epsilon),
        ),
        alpha_A_algn=jnp.array(alpha_A_algn),
        alpha_B_bij=jnp.array(alpha_B_bij),
        alpha_C_jac=jnp.array(alpha_C_jac),
        alpha_D_cp_norm=jnp.array(alpha_D_cp_norm),
        alpha_E_imposed_disp=jnp.array(alpha_E_imposed_disp),
        alpha_F_cp_pos_reg=jnp.array(alpha_F_cp_pos_reg),
        alpha_G_got_w2=jnp.array(alpha_G_got_w2),
        alpha_J_point_registration=jnp.array(alpha_J_point_registration),
    )


def _create_mappings(
    static_config: RBFFreeStaticConfig,
) -> Callable[[RBFFreeTracedParams], tuple[Callable, Callable]]:
    """Create a function that generates W and T mappings from optimized params."""

    def make_mappings(opt_params: RBFFreeTracedParams) -> tuple[Callable, Callable]:
        mapping_rbf_kwargs = _build_mapping_rbf_kwargs(static_config)
        centers_w = _build_full_centers(
            static_config.boundary_centers,
            opt_params.interior_centers_w,
        )
        centers_t = _build_full_centers(
            static_config.boundary_centers,
            opt_params.interior_centers_t,
        )

        def mapping_w(points: Float[Array, "n_points 2"], s: float = 1.0):
            """W mapping (field 0 -> field 1)."""
            deformed = _rbf_mapping(
                points,
                centers_w,
                opt_params.displacements_w,
                mapping_rbf_kwargs,
            )
            return (1 - s) * points + s * deformed

        def mapping_t(points: Float[Array, "n_points 2"], s: float = 1.0):
            """T mapping (field 1 -> field 0)."""
            deformed = _rbf_mapping(
                points,
                centers_t,
                opt_params.displacements_t,
                mapping_rbf_kwargs,
            )
            return (1 - s) * points + s * deformed

        return mapping_w, mapping_t

    return make_mappings


def _create_displacement_extractor() -> Callable[
    [Any],
    tuple[Float[Array, "n_c 2"], Float[Array, "n_c 2"]],
]:
    """Create helper extracting displacements from params or SolveResult."""

    def get_displacements(
        result_or_params: Any,
    ) -> tuple[Float[Array, "n_c 2"], Float[Array, "n_c 2"]]:
        params = _extract_params(result_or_params)
        return params.displacements_w, params.displacements_t

    return get_displacements


def _create_centers_and_displacement_extractor(
    static_config: RBFFreeStaticConfig,
) -> Callable[
    [Any],
    tuple[
        Float[Array, "n_c 2"],
        Float[Array, "n_c 2"],
        Float[Array, "n_c 2"],
        Float[Array, "n_c 2"],
    ],
]:
    """Create helper extracting full centers and displacements."""

    def get_centers_and_displacements(
        result_or_params: Any,
    ) -> tuple[
        Float[Array, "n_c 2"],
        Float[Array, "n_c 2"],
        Float[Array, "n_c 2"],
        Float[Array, "n_c 2"],
    ]:
        params = _extract_params(result_or_params)
        centers_w = _build_full_centers(
            static_config.boundary_centers,
            params.interior_centers_w,
        )
        centers_t = _build_full_centers(
            static_config.boundary_centers,
            params.interior_centers_t,
        )
        return (
            centers_w,
            params.displacements_w,
            centers_t,
            params.displacements_t,
        )

    return get_centers_and_displacements


def _extract_params(result_or_params: Any) -> RBFFreeTracedParams:
    if isinstance(result_or_params, RBFFreeTracedParams):
        return result_or_params
    if hasattr(result_or_params, "params") and isinstance(
        result_or_params.params, RBFFreeTracedParams
    ):
        return result_or_params.params
    raise TypeError(
        "Expected RBFFreeTracedParams or an object with a 'params' attribute "
        "of type RBFFreeTracedParams."
    )


def make_rbf_free_cp_cost_function(
    mesh: MeshJaxed,
    u0_values: Float[Array, " n"],
    u1_values: Float[Array, " n"],
    boundary_centers: Float[Array, "n_b 2"],
    interior_centers: Float[Array, "n_i 2"],
    imposed_disp_points: Float[Array, "n_bc 2"],
    initial_mapping_function_w: Mapping,
    initial_mapping_function_t: Mapping,
    alpha_A_algn: float,
    alpha_B_bij: float,
    alpha_C_jac: float,
    alpha_D_cp_norm: float,
    alpha_E_imposed_disp: float,
    barrier_epsilon: float,
    imposed_disp_targets: Float[Array, "n_bc 2"] | None = None,
    alpha_F_cp_pos_reg: float = 0.0,
    alpha_G_got_w2: float = 0.0,
    alpha_J_point_registration: float = 0.0,
    point_registration_points_0: Float[Array, "n_registration 2"] | None = None,
    point_registration_points_1: Float[Array, "n_registration 2"] | None = None,
    rbf_kernel: str = "thin_plate_spline",
    rbf_neighbors: int | None = None,
    rbf_epsilon: float | None = None,
    rbf_smoothing: float = 0.0,
    rbf_degree: int | None = None,
    field_interp_kind: Literal["auto", "grid", "rbf", "nearest", "linear"] = "auto",
    field_linear_frozen_points: bool = True,
    field_rbf_neighbors: int = 64,
    field_rbf_kernel: str | None = None,
    initialisation_noise_factor: float = 0.0,
    barrier_fn_name: BarrierFnName = "logarithmic",
    record_cp_history: bool = True,
) -> RBFFreeFactoryResult:
    """Create an RBF free-CP cost function with initial parameters.

    Interior control-point positions are optimized separately for W and T
    mappings.  Boundary center positions remain pinned.

    Args:
        mesh: Global domain mesh for interpolation, alignment, etc.
        u0_values: Field 0 values on the global mesh.
        u1_values: Field 1 values on the global mesh.
        boundary_centers: Pinned RBF center coordinates on the domain boundary.
        interior_centers: Initial interior center coordinates (optimizable).
        imposed_disp_points: Points where displacement is imposed (boundary).
        initial_mapping_function_w: Initial W mapping guess.
        initial_mapping_function_t: Initial T mapping guess.
        alpha_A_algn: Weight for field alignment cost.
        alpha_B_bij: Weight for mapping synchronization cost.
        alpha_C_jac: Weight for Jacobian barrier cost.
        alpha_D_cp_norm: Weight for control point displacement regularization.
        alpha_E_imposed_disp: Weight for imposed displacement cost.
        barrier_epsilon: Initial barrier parameter.
        imposed_disp_targets: Target displacements at imposed points (default: 0).
        alpha_F_cp_pos_reg: Weight for interior center position regularization.
        alpha_G_got_w2: Weight for W2 distance cost (default: 0).
        alpha_J_point_registration: Weight for point registration cost (default: 0).
        point_registration_points_0: Source landmarks in field 0 domain (default: None).
        point_registration_points_1: Target landmarks in field 1 domain (default: None).
        rbf_kernel: RBF kernel for mapping (default: thin_plate_spline).
        rbf_neighbors: Neighbors for mapping RBF (default: None = all centers).
        rbf_epsilon: Shape parameter for RBF kernel.
        rbf_smoothing: Smoothing for RBF interpolators.
        rbf_degree: Polynomial degree for RBF.
        field_interp_kind: Field interpolation method
            ("auto", "grid", "rbf", "nearest", "linear").
        field_linear_frozen_points: Reuse triangulation transforms in linear ND
            interpolation when possible.
        field_rbf_neighbors: Neighbors for field RBF on full mesh (default: 64).
        field_rbf_kernel: Kernel for field RBF (defaults to rbf_kernel).
        initialisation_noise_factor: Noise for initial params (default: 0).
        barrier_fn_name: Barrier function to use (default: "logarithmic").
        record_cp_history: Record CP snapshots in aux each iteration.

    Returns:
        RBFFreeFactoryResult with cost_fn, init_params, and helper functions.
    """
    if imposed_disp_targets is None:
        imposed_disp_targets = jnp.zeros_like(imposed_disp_points)

    # Normalize missing point registration inputs to empty arrays.
    if point_registration_points_0 is None:
        point_registration_points_0 = jnp.zeros((0, 2), dtype=jnp.float64)
    if point_registration_points_1 is None:
        point_registration_points_1 = jnp.zeros((0, 2), dtype=jnp.float64)

    resolved_field_kind = _resolve_field_interp_kind(mesh, field_interp_kind)

    static_config = _build_static_config(
        mesh=mesh,
        u0_values=u0_values,
        u1_values=u1_values,
        boundary_centers=boundary_centers,
        interior_centers=interior_centers,
        imposed_disp_points=imposed_disp_points,
        imposed_disp_targets=imposed_disp_targets,
        point_registration_points_0=point_registration_points_0,
        point_registration_points_1=point_registration_points_1,
        rbf_kernel=rbf_kernel,
        rbf_neighbors=rbf_neighbors,
        rbf_epsilon=rbf_epsilon,
        rbf_smoothing=rbf_smoothing,
        rbf_degree=rbf_degree,
        field_interp_kind=resolved_field_kind,
        field_linear_frozen_points=field_linear_frozen_points,
        field_rbf_neighbors=field_rbf_neighbors,
        field_rbf_kernel=field_rbf_kernel,
        alpha_A_algn=alpha_A_algn,
        alpha_B_bij=alpha_B_bij,
        alpha_C_jac=alpha_C_jac,
        alpha_D_cp_norm=alpha_D_cp_norm,
        alpha_E_imposed_disp=alpha_E_imposed_disp,
        alpha_F_cp_pos_reg=alpha_F_cp_pos_reg,
        alpha_G_got_w2=alpha_G_got_w2,
        alpha_J_point_registration=alpha_J_point_registration,
        record_cp_history=record_cp_history,
    )

    init_params = _build_initial_params(
        boundary_centers=boundary_centers,
        interior_centers=interior_centers,
        initial_mapping_function_w=initial_mapping_function_w,
        initial_mapping_function_t=initial_mapping_function_t,
        initialisation_noise_factor=initialisation_noise_factor,
        alpha_A_algn=alpha_A_algn,
        alpha_B_bij=alpha_B_bij,
        alpha_C_jac=alpha_C_jac,
        alpha_D_cp_norm=alpha_D_cp_norm,
        alpha_E_imposed_disp=alpha_E_imposed_disp,
        alpha_F_cp_pos_reg=alpha_F_cp_pos_reg,
        alpha_G_got_w2=alpha_G_got_w2,
        alpha_J_point_registration=alpha_J_point_registration,
        barrier_epsilon=barrier_epsilon,
        barrier_fn_name=barrier_fn_name,
    )

    make_mappings = _create_mappings(static_config)
    get_displacements = _create_displacement_extractor()
    get_centers_and_displacements = _create_centers_and_displacement_extractor(
        static_config
    )

    def traced_only_cost_fn(opt_params: RBFFreeTracedParams):
        return rbf_free_cp_cost_function(opt_params, static_config)

    return RBFFreeFactoryResult(
        cost_fn=traced_only_cost_fn,
        init_params=init_params,
        make_mappings=make_mappings,
        get_displacements=get_displacements,
        get_centers_and_displacements=get_centers_and_displacements,
    )
