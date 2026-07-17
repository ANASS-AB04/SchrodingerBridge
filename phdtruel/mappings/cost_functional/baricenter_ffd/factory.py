"""Factory helpers for Wasserstein-like barycenter FFD objective."""

from __future__ import annotations

from typing import Any, Callable, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float

from phdtruel.mappings.cost_functional.baricenter_ffd.cost_fn import (
    wasserstein_barycenter_cost_function,
)
from phdtruel.mappings.cost_functional.baricenter_ffd.residuals import (
    wasserstein_barycenter_residual_function,
)
from phdtruel.mappings.cost_functional.baricenter_ffd.types import (
    BariFFDStaticConfig,
    BariFFDTracedParams,
    BarrierFnConfig,
)
from phdtruel.mappings.cost_functional.types import MeshJaxed
from phdtruel.mappings.cost_functional.utils import impose_control_displacements
from phdtruel.mappings.ffd import (
    FFDParameters,
    get_control_points_from_ffd_params,
    piecewise_ffd_mapping_2d,
)
from phdtruel.mappings.mappings import Mapping, identity_mapping


class BariFFDFactoryResult(NamedTuple):
    """Result tuple from barycenter factory."""

    cost_fn: Callable
    residual_fn: Callable
    init_params: BariFFDTracedParams
    make_mappings: Callable[[BariFFDTracedParams], tuple[Callable, ...]]
    get_displacements: Callable[
        [Any], tuple[tuple[Float[Array, "n_cp_x n_cp_y 2"], ...], ...]
    ]


def _get_constrained_displacements(
    params: BariFFDTracedParams,
    static_config: BariFFDStaticConfig,
) -> tuple[tuple[Float[Array, "n_cp_x n_cp_y 2"], ...], ...]:
    """Apply BC masks to all field/region displacement blocks."""
    return tuple(
        tuple(
            impose_control_displacements(disp, mask)
            for disp, mask in zip(
                field_disps, static_config.all_imposed_displacements_masks
            )
        )
        for field_disps in params.displacements_w
    )


def _create_displacement_extractor(
    static_config: BariFFDStaticConfig,
) -> Callable[[Any], tuple[tuple[Float[Array, "n_cp_x n_cp_y 2"], ...], ...]]:
    """Create helper extracting constrained displacements from params/result."""

    def get_displacements(
        result_or_params: Any,
    ) -> tuple[tuple[Float[Array, "n_cp_x n_cp_y 2"], ...], ...]:
        if isinstance(result_or_params, BariFFDTracedParams):
            params = result_or_params
        elif hasattr(result_or_params, "params") and isinstance(
            result_or_params.params, BariFFDTracedParams
        ):
            params = result_or_params.params
        else:
            raise TypeError(
                "Expected BariFFDTracedParams or an object with a 'params' attribute "
                "of type BariFFDTracedParams."
            )

        return _get_constrained_displacements(params, static_config)

    return get_displacements


def _build_static_config(
    mesh: MeshJaxed,
    region_meshes: list[MeshJaxed],
    all_ffd_parameters: tuple[FFDParameters, ...],
    all_imposed_displacements_masks: list[Float[Array, "n_cp_x n_cp_y 2"]],
    all_u_values: tuple[Float[Array, " n_points"], ...],
    all_u_per_region: tuple[tuple[Float[Array, " n_region"], ...], ...],
    gamma: Float[Array, " n_fields"],
    alpha_A_w_id: float,
    alpha_B_pairwise_align: float,
    alpha_C_log_barrier_jac: float,
    alpha_D_pairwise_w2: float,
    alpha_E_bary_w2: float,
) -> BariFFDStaticConfig:
    return BariFFDStaticConfig(
        mesh=mesh,
        region_meshes=tuple(region_meshes),
        all_ffd_parameters=all_ffd_parameters,
        all_imposed_displacements_masks=tuple(all_imposed_displacements_masks),
        all_u_values=all_u_values,
        all_u_per_region=all_u_per_region,
        gamma=gamma,
        alpha_A_w_id=alpha_A_w_id,
        alpha_B_pairwise_align=alpha_B_pairwise_align,
        alpha_C_log_barrier_jac=alpha_C_log_barrier_jac,
        alpha_D_pairwise_w2=alpha_D_pairwise_w2,
        alpha_E_bary_w2=alpha_E_bary_w2,
    )


def _build_initial_params(
    all_ffd_parameters: tuple[FFDParameters, ...],
    all_n_cp: list[tuple[int, int]],
    initial_mapping_functions_w: list[Mapping],
    initialisation_noise_factor: float,
    alpha_A_w_id: float,
    alpha_B_pairwise_align: float,
    alpha_C_log_barrier_jac: float,
    alpha_D_pairwise_w2: float,
    alpha_E_bary_w2: float,
    log_barrier_config: BarrierFnConfig,
) -> BariFFDTracedParams:
    """Build initial displacement blocks for all fields and regions."""
    all_initial_displacements_w: list[tuple[Float[Array, "n_cp_x n_cp_y 2"], ...]] = []

    key = jax.random.PRNGKey(42)

    for mapping_w in initial_mapping_functions_w:
        field_displacements: list[Float[Array, "n_cp_x n_cp_y 2"]] = []
        for ffd_params, (n_cp_x, n_cp_y) in zip(all_ffd_parameters, all_n_cp):
            ctrl_pts = get_control_points_from_ffd_params(ffd_params)
            ctrl_pts_flat = np.asarray(ctrl_pts.reshape(-1, 2))

            mapped_w = np.asarray(mapping_w(ctrl_pts_flat))
            disp_w = mapped_w.reshape(n_cp_x, n_cp_y, 2) - ctrl_pts

            key, subkey = jax.random.split(key)
            noise = initialisation_noise_factor * jax.random.normal(
                subkey, disp_w.shape
            )
            field_displacements.append(jnp.array(disp_w) + noise)

        all_initial_displacements_w.append(tuple(field_displacements))

    return BariFFDTracedParams(
        displacements_w=tuple(all_initial_displacements_w),
        alpha_A_w_id=jnp.array(alpha_A_w_id),
        alpha_B_pairwise_align=jnp.array(alpha_B_pairwise_align),
        alpha_C_log_barrier_jac=jnp.array(alpha_C_log_barrier_jac),
        alpha_D_pairwise_w2=jnp.array(alpha_D_pairwise_w2),
        alpha_E_bary_w2=jnp.array(alpha_E_bary_w2),
        log_barrier_config=log_barrier_config,
    )


def _create_mappings(
    static_config: BariFFDStaticConfig,
) -> Callable[[BariFFDTracedParams], tuple[Callable, ...]]:
    """Create a function that generates all W mappings from optimized params."""

    def make_mappings(opt_params: BariFFDTracedParams) -> tuple[Callable, ...]:
        all_ffd_params_np = tuple(
            FFDParameters(
                np.array(p.box_origin),
                np.array(p.box_length),
                np.array(p.n_control_points),
            )
            for p in static_config.all_ffd_parameters
        )

        constrained_displacements = _get_constrained_displacements(
            opt_params,
            static_config,
        )

        mappings: list[Callable] = []
        for field_disps in constrained_displacements:

            def mapping(
                points: Float[Array, "n_points 2"], s: float = 1.0, _disp=field_disps
            ):
                deformed = piecewise_ffd_mapping_2d(points, _disp, all_ffd_params_np)
                return (1 - s) * points + s * deformed

            mappings.append(mapping)

        return tuple(mappings)

    return make_mappings


def make_wasserstein_barycenter_ffd_cost_function(
    region_meshes: list[MeshJaxed],
    all_n_cp: list[tuple[int, int]],
    all_u_values: tuple[Float[Array, " n_points"], ...],
    gamma: Float[Array, " n_fields"],
    alpha_A_w_id: float,
    alpha_B_pairwise_align: float,
    alpha_C_log_barrier_jac: float,
    alpha_D_pairwise_w2: float,
    alpha_E_bary_w2: float,
    all_imposed_displacements_masks: list[Float[Array, "n_cp_x n_cp_y 2"]]
    | None = None,
    initial_mapping_functions_w: list[Mapping] | None = None,
    full_domain_mesh: MeshJaxed | None = None,
    initialisation_noise_factor: float = 0.0,
    boundary_conditions: (
        Literal["slip_on_boundaries", "slip_on_external_boundaries"] | None
    ) = None,
    log_barrier_config: BarrierFnConfig | None = None,
) -> BariFFDFactoryResult:
    """Create barycenter cost function and initialization artifacts."""
    n_fields = len(all_u_values)
    if n_fields == 0:
        raise ValueError("all_u_values must contain at least one field.")

    if initial_mapping_functions_w is None:
        initial_mapping_functions_w = [identity_mapping for _ in range(n_fields)]
    if len(initial_mapping_functions_w) != n_fields:
        raise ValueError("initial_mapping_functions_w must match number of fields.")

    if all_imposed_displacements_masks is None and boundary_conditions is None:
        raise ValueError(
            "Either all_imposed_displacements_masks or boundary_conditions must be provided"
        )

    masks_for_building: list[Float[Array, "n_cp_x n_cp_y 2"]]
    if all_imposed_displacements_masks is not None:
        masks_for_building = all_imposed_displacements_masks
    else:
        from phdtruel.fields.meshes import RegularGrid
        from phdtruel.mappings.boundary_conditions import (
            create_slip_on_external_boundaries_masks,
            initialize_all_slip_masks,
        )

        region_meshes_regular = [
            RegularGrid([np.array(ax) for ax in mesh.axes]) for mesh in region_meshes
        ]

        if boundary_conditions == "slip_on_boundaries":
            numpy_masks = initialize_all_slip_masks(all_n_cp)
        elif boundary_conditions == "slip_on_external_boundaries":
            numpy_masks = create_slip_on_external_boundaries_masks(
                all_n_cp,
                region_meshes_regular,
                inter_region_sync=None,
                tolerance=None,
            )
        else:
            raise ValueError(
                f"Unknown boundary_conditions: {boundary_conditions}. "
                "Supported: 'slip_on_boundaries', 'slip_on_external_boundaries'"
            )

        masks_for_building = [jnp.array(mask) for mask in numpy_masks]

    all_ffd_parameters = []
    for mesh, (n_cp_x, n_cp_y) in zip(region_meshes, all_n_cp):
        points_np = np.array(mesh.points)
        bounds_min = points_np.min(axis=0)
        bounds_max = points_np.max(axis=0)
        lengths = bounds_max - bounds_min

        all_ffd_parameters.append(
            FFDParameters(
                box_origin=jnp.array(bounds_min),
                box_length=jnp.array(lengths),
                n_control_points=jnp.array([n_cp_x, n_cp_y]),
            )
        )

    all_ffd_parameters_tuple = tuple(all_ffd_parameters)
    global_mesh = full_domain_mesh if full_domain_mesh is not None else region_meshes[0]

    all_u_per_region_list: list[tuple[Float[Array, " n_region"], ...]] = []
    if full_domain_mesh is not None:
        from scipy.spatial import KDTree

        full_pts = np.array(full_domain_mesh.points)
        tree = KDTree(full_pts)
        for field_values in all_u_values:
            per_region = []
            for region_mesh in region_meshes:
                region_pts = np.array(region_mesh.points)
                _, idx = tree.query(region_pts)
                per_region.append(field_values[idx])
            all_u_per_region_list.append(tuple(per_region))
    else:
        for field_values in all_u_values:
            per_region = []
            for region_mesh in region_meshes:
                n_pts = np.array(region_mesh.points).shape[0]
                idx = jnp.arange(n_pts)
                per_region.append(field_values[idx])
            all_u_per_region_list.append(tuple(per_region))

    # Default barrier config if not provided
    if log_barrier_config is None:
        log_barrier_config = BarrierFnConfig(
            barrier_fn_name="logarithmic",
            barrier_epsilon=jnp.array(0.1),
        )

    static_config = _build_static_config(
        mesh=global_mesh,
        region_meshes=region_meshes,
        all_ffd_parameters=all_ffd_parameters_tuple,
        all_imposed_displacements_masks=masks_for_building,
        all_u_values=all_u_values,
        all_u_per_region=tuple(all_u_per_region_list),
        gamma=gamma,
        alpha_A_w_id=alpha_A_w_id,
        alpha_B_pairwise_align=alpha_B_pairwise_align,
        alpha_C_log_barrier_jac=alpha_C_log_barrier_jac,
        alpha_D_pairwise_w2=alpha_D_pairwise_w2,
        alpha_E_bary_w2=alpha_E_bary_w2,
    )

    init_params = _build_initial_params(
        all_ffd_parameters=all_ffd_parameters_tuple,
        all_n_cp=all_n_cp,
        initial_mapping_functions_w=initial_mapping_functions_w,
        initialisation_noise_factor=initialisation_noise_factor,
        alpha_A_w_id=alpha_A_w_id,
        alpha_B_pairwise_align=alpha_B_pairwise_align,
        alpha_C_log_barrier_jac=alpha_C_log_barrier_jac,
        alpha_D_pairwise_w2=alpha_D_pairwise_w2,
        alpha_E_bary_w2=alpha_E_bary_w2,
        log_barrier_config=log_barrier_config,
    )

    make_mappings = _create_mappings(static_config)
    get_displacements = _create_displacement_extractor(static_config)

    def traced_only_cost_fn(opt_params: BariFFDTracedParams):
        return wasserstein_barycenter_cost_function(opt_params, static_config)

    def traced_only_residual_fn(opt_params: BariFFDTracedParams):
        """Residual vectors for natural-gradient / Gauss-Newton optimization."""
        return wasserstein_barycenter_residual_function(opt_params, static_config)

    return BariFFDFactoryResult(
        cost_fn=traced_only_cost_fn,
        residual_fn=traced_only_residual_fn,
        init_params=init_params,
        make_mappings=make_mappings,
        get_displacements=get_displacements,
    )
