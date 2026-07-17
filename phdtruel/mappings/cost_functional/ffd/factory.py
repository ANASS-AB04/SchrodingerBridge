"""Factory function for creating FFD cost function and initial parameters."""

from __future__ import annotations

from typing import Any, Callable, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float

from phdtruel.mappings.cost_functional.ffd.cost_fn import piecewise_ffd_cost_function
from phdtruel.mappings.cost_functional.ffd.residuals import (
    piecewise_ffd_residual_function,
)
from phdtruel.mappings.cost_functional.ffd.types import (
    BarrierFnConfig,
    FFDStaticConfig,
    FFDTracedParams,
)
from phdtruel.mappings.cost_functional.types import MeshJaxed
from phdtruel.mappings.cost_functional.utils import (
    BarrierFnName,
    impose_control_displacements,
)
from phdtruel.mappings.ffd import (
    FFDParameters,
    get_control_points_from_ffd_params,
    piecewise_ffd_mapping_2d,
)
from phdtruel.mappings.mappings import Mapping


class FFDFactoryResult(NamedTuple):
    """Result tuple from factory function.

    Attributes:
        cost_fn: The jitted cost function
        init_params: Initial traced parameters
        make_mappings: Function to create W and T mappings from optimized params
        get_displacements: Function to extract constrained displacements from
            solved params (or a SolveResult carrying them)
    """

    cost_fn: Callable
    residual_fn: Callable
    init_params: FFDTracedParams
    make_mappings: Callable[[FFDTracedParams], tuple[Callable, Callable]]
    get_displacements: Callable[
        [Any],
        tuple[
            tuple[Float[Array, "n_cp_x n_cp_y 2"], ...],
            tuple[Float[Array, "n_cp_x n_cp_y 2"], ...],
        ],
    ]


def _get_constrained_displacements(
    params: FFDTracedParams,
    static_config: FFDStaticConfig,
) -> tuple[
    tuple[Float[Array, "n_cp_x n_cp_y 2"], ...],
    tuple[Float[Array, "n_cp_x n_cp_y 2"], ...],
]:
    """Apply BC masks to traced displacements.

    Args:
        params: Traced FFD parameters.
        static_config: Static config containing BC masks.

    Returns:
        Tuple ``(constrained_displacements_w, constrained_displacements_t)``.
    """
    constrained_displacements_w = tuple(
        impose_control_displacements(disp, mask)
        for disp, mask in zip(
            params.displacements_w,
            static_config.all_imposed_displacements_masks,
        )
    )
    constrained_displacements_t = tuple(
        impose_control_displacements(disp, mask)
        for disp, mask in zip(
            params.displacements_t,
            static_config.all_imposed_displacements_masks,
        )
    )

    return constrained_displacements_w, constrained_displacements_t


def _create_displacement_extractor(
    static_config: FFDStaticConfig,
) -> Callable[
    [Any],
    tuple[
        tuple[Float[Array, "n_cp_x n_cp_y 2"], ...],
        tuple[Float[Array, "n_cp_x n_cp_y 2"], ...],
    ],
]:
    """Create helper extracting constrained displacements from params/result."""

    def get_displacements(
        result_or_params: Any,
    ) -> tuple[
        tuple[Float[Array, "n_cp_x n_cp_y 2"], ...],
        tuple[Float[Array, "n_cp_x n_cp_y 2"], ...],
    ]:
        """Return constrained displacements from ``FFDTracedParams`` or ``SolveResult``.

        Args:
            result_or_params: Either ``FFDTracedParams`` directly, or an object
                exposing a ``params`` attribute containing ``FFDTracedParams``
                (e.g. ``SolveResult``).

        Returns:
            Tuple ``(displacements_w, displacements_t)`` with BCs enforced.
        """
        if isinstance(result_or_params, FFDTracedParams):
            params = result_or_params
        elif hasattr(result_or_params, "params") and isinstance(
            result_or_params.params, FFDTracedParams
        ):
            params = result_or_params.params
        else:
            raise TypeError(
                "Expected FFDTracedParams or an object with a 'params' attribute "
                "of type FFDTracedParams."
            )

        return _get_constrained_displacements(params, static_config)

    return get_displacements


def _build_static_config(
    mesh: MeshJaxed,
    u0_values: Float[Array, " n"],
    u1_values: Float[Array, " n"],
    u0_per_region: tuple[Float[Array, " n_region"], ...],
    u1_per_region: tuple[Float[Array, " n_region"], ...],
    region_meshes: list[MeshJaxed],
    all_ffd_parameters: tuple[FFDParameters, ...],
    all_imposed_displacements_masks: list[Float[Array, "n_cp_x n_cp_y 2"]],
    inter_region_sync_indices: tuple[tuple[int, int], ...],
    inter_region_sync_points: tuple[Float[Array, "n_sync 2"], ...],
    point_registration_points_0: Float[Array, "n_registration 2"],
    point_registration_points_1: Float[Array, "n_registration 2"],
    alpha_A_algn: float,
    alpha_B_bij: float,
    alpha_C_jac: float,
    alpha_D_cp_norm: float,
    alpha_G_got_w2: float,
    alpha_H_inter_sync: float,
    alpha_I_inter_jac_sync: float,
    alpha_J_point_registration: float,
) -> FFDStaticConfig:
    """Build the static configuration for FFD cost function.

    Args:
        mesh: Global domain mesh for interpolation, alignment, etc.
        u0_values: Field 0 values on the global mesh
        u1_values: Field 1 values on the global mesh
        u0_per_region: Field 0 values per region (for piecewise interpolation)
        u1_per_region: Field 1 values per region (for piecewise interpolation)
        region_meshes: List of meshes for each region
        all_ffd_parameters: FFD parameters for each region
        all_imposed_displacements_masks: BC masks per region
        inter_region_sync_indices: Indices of regions to synchronize
        inter_region_sync_points: Shared boundary points per region pair
        point_registration_points_0: Source landmarks in field 0 domain
        point_registration_points_1: Target landmarks in field 1 domain
        alpha_A_algn: Weight for alignment cost
        alpha_B_bij: Weight for mapping sync cost
        alpha_C_jac: Weight for Jacobian barrier cost
        alpha_D_cp_norm: Weight for control point regularization
        alpha_G_got_w2: Weight for W2 distance cost
        alpha_H_inter_sync: Weight for inter-region sync cost
        alpha_I_inter_jac_sync: Weight for inter-region Jacobian sync
        alpha_J_point_registration: Weight for point registration cost

    Returns:
        Frozen static configuration
    """
    enable_w2_cost = alpha_G_got_w2 > 0.0
    enable_inter_sync = alpha_H_inter_sync > 0.0 and len(inter_region_sync_indices) > 0
    enable_point_registration = (
        alpha_J_point_registration > 0.0 and point_registration_points_0.shape[0] > 0
    )

    return FFDStaticConfig(
        mesh=mesh,
        u0_values=u0_values,
        u1_values=u1_values,
        u0_per_region=u0_per_region,
        u1_per_region=u1_per_region,
        region_meshes=tuple(region_meshes),
        all_ffd_parameters=all_ffd_parameters,
        all_imposed_displacements_masks=tuple(all_imposed_displacements_masks),
        inter_region_sync_indices=inter_region_sync_indices,
        inter_region_sync_points=inter_region_sync_points,
        point_registration_points_0=point_registration_points_0,
        point_registration_points_1=point_registration_points_1,
        alpha_A_algn=alpha_A_algn,
        alpha_B_bij=alpha_B_bij,
        alpha_C_jac=alpha_C_jac,
        alpha_D_cp_norm=alpha_D_cp_norm,
        alpha_G_got_w2=alpha_G_got_w2,
        alpha_H_inter_sync=alpha_H_inter_sync,
        alpha_I_inter_jac_sync=alpha_I_inter_jac_sync,
        alpha_J_point_registration=alpha_J_point_registration,
        enable_w2_cost=enable_w2_cost,
        enable_inter_sync=enable_inter_sync,
        enable_point_registration=enable_point_registration,
    )


def _build_initial_params(
    all_ffd_parameters: tuple[FFDParameters, ...],
    all_n_cp: list[tuple[int, int]],
    initial_mapping_function_w: Mapping,
    initial_mapping_function_t: Mapping,
    initialisation_noise_factor: float,
    alpha_A_algn: float,
    alpha_B_bij: float,
    alpha_C_jac: float,
    alpha_D_cp_norm: float,
    alpha_G_got_w2: float,
    alpha_H_inter_sync: float,
    alpha_I_inter_jac_sync: float,
    alpha_J_point_registration: float,
    barrier_epsilon: float,
    barrier_fn_name: BarrierFnName = "logarithmic",
) -> FFDTracedParams:
    """Build initial traced parameters for optimization.

    Args:
        all_ffd_parameters: FFD parameters for each region
        all_n_cp: Number of control points (n_cp_x, n_cp_y) per region
        initial_mapping_function_w: Initial W mapping guess
        initial_mapping_function_t: Initial T mapping guess
        initialisation_noise_factor: Noise level to add to initial params
        alpha_A_algn: Field alignment weight
        alpha_B_bij: Mapping sync weight
        alpha_C_jac: Jacobian barrier weight
        alpha_D_cp_norm: Control point reg weight
        alpha_G_got_w2: W2 distance weight
        alpha_H_inter_sync: Inter-region sync weight
        alpha_I_inter_jac_sync: Inter-region Jacobian sync weight
        alpha_J_point_registration: Point registration weight
        barrier_epsilon: Initial barrier steepness parameter
        barrier_fn_name: Barrier function to use (default: "logarithmic")

    Returns:
        Initial traced parameters
    """
    # Compute initial displacements for each region
    all_initial_displacements_w = []
    all_initial_displacements_t = []

    key = jax.random.PRNGKey(42)

    for idx, (ffd_params, (n_cp_x, n_cp_y)) in enumerate(
        zip(all_ffd_parameters, all_n_cp)
    ):
        # Get control points for this region
        ctrl_pts = get_control_points_from_ffd_params(ffd_params)
        ctrl_pts_flat = np.asarray(ctrl_pts.reshape(-1, 2))

        # Initial displacement for mapping W (field 0 -> field 1)
        mapped_w = np.asarray(initial_mapping_function_w(ctrl_pts_flat))
        disp_w = mapped_w.reshape(n_cp_x, n_cp_y, 2) - ctrl_pts

        # Initial displacement for mapping T (field 1 -> field 0)
        mapped_t = np.asarray(initial_mapping_function_t(ctrl_pts_flat))
        disp_t = mapped_t.reshape(n_cp_x, n_cp_y, 2) - ctrl_pts

        # Add noise
        key, subkey1, subkey2 = jax.random.split(key, 3)
        noise_w = initialisation_noise_factor * jax.random.normal(subkey1, disp_w.shape)
        noise_t = initialisation_noise_factor * jax.random.normal(subkey2, disp_t.shape)
        disp_w = jnp.array(disp_w) + noise_w
        disp_t = jnp.array(disp_t) + noise_t

        all_initial_displacements_w.append(disp_w)
        all_initial_displacements_t.append(disp_t)

    return FFDTracedParams(
        displacements_w=tuple(all_initial_displacements_w),
        displacements_t=tuple(all_initial_displacements_t),
        barrier_config=BarrierFnConfig(
            barrier_fn_name=barrier_fn_name,
            barrier_epsilon=jnp.array(barrier_epsilon),
        ),
        alpha_A_algn=jnp.array(alpha_A_algn),
        alpha_B_bij=jnp.array(alpha_B_bij),
        alpha_C_jac=jnp.array(alpha_C_jac),
        alpha_D_cp_norm=jnp.array(alpha_D_cp_norm),
        alpha_G_got_w2=jnp.array(alpha_G_got_w2),
        alpha_H_inter_sync=jnp.array(alpha_H_inter_sync),
        alpha_I_inter_jac_sync=jnp.array(alpha_I_inter_jac_sync),
        alpha_J_point_registration=jnp.array(alpha_J_point_registration),
    )


def _create_mappings(
    static_config: FFDStaticConfig,
) -> Callable[[FFDTracedParams], tuple[Callable, Callable]]:
    """Create a function that generates W and T mappings from optimized params.

    Args:
        static_config: Static configuration containing FFD parameters

    Returns:
        Function that takes optimized params and returns (W_mapping, T_mapping)
    """

    def make_mappings(opt_params: FFDTracedParams) -> tuple[Callable, Callable]:
        """Create W and T mapping functions from optimized parameters.

        Args:
            opt_params: Optimized traced parameters

        Returns:
            Tuple of (W_mapping, T_mapping) functions
        """
        # Convert FFD parameters to numpy for mapping functions
        all_ffd_params_np = tuple(
            FFDParameters(
                np.array(p.box_origin),
                np.array(p.box_length),
                np.array(p.n_control_points),
            )
            for p in static_config.all_ffd_parameters
        )

        # Keep mapping outputs consistent with optimization BC constraints.
        constrained_displacements_w, constrained_displacements_t = (
            _get_constrained_displacements(opt_params, static_config)
        )

        def mapping_w(points: Float[Array, "n_points 2"], s: float = 1.0):
            """W mapping (field 0 -> field 1).

            Args:
                points: Points to map
                s: Interpolation parameter (0 = identity, 1 = full deformation)

            Returns:
                Mapped points
            """
            deformed = piecewise_ffd_mapping_2d(
                points, constrained_displacements_w, all_ffd_params_np
            )
            return (1 - s) * points + s * deformed

        def mapping_t(points: Float[Array, "n_points 2"], s: float = 1.0):
            """T mapping (field 1 -> field 0).

            Args:
                points: Points to map
                s: Interpolation parameter (0 = identity, 1 = full deformation)

            Returns:
                Mapped points
            """
            deformed = piecewise_ffd_mapping_2d(
                points, constrained_displacements_t, all_ffd_params_np
            )
            return (1 - s) * points + s * deformed

        return mapping_w, mapping_t

    return make_mappings


def make_piecewise_ffd_cost_function(
    region_meshes: list[MeshJaxed],
    all_n_cp: list[tuple[int, int]],
    all_imposed_displacements_masks: list[Float[Array, "n_cp_x n_cp_y 2"]] | None,
    initial_mapping_function_w: Mapping,
    initial_mapping_function_t: Mapping,
    alpha_A_algn: float,
    alpha_B_bij: float,
    alpha_C_jac: float,
    alpha_D_cp_norm: float,
    barrier_epsilon: float,
    u0_values: Float[Array, " n"],
    u1_values: Float[Array, " n"],
    full_domain_mesh: MeshJaxed | None = None,
    alpha_G_got_w2: float = 0.0,
    alpha_H_inter_sync: float = 0.0,
    alpha_I_inter_jac_sync: float = 0.0,
    alpha_J_point_registration: float = 0.0,
    inter_region_sync_indices: tuple[tuple[int, int], ...] = (),
    inter_region_sync_points: tuple[Float[Array, "n_sync 2"], ...] = (),
    point_registration_points_0: Float[Array, "n_registration 2"] | None = None,
    point_registration_points_1: Float[Array, "n_registration 2"] | None = None,
    initialisation_noise_factor: float = 0.0,
    boundary_conditions: (
        Literal["slip_on_boundaries", "slip_on_external_boundaries"] | None
    ) = None,
    barrier_fn_name: BarrierFnName = "logarithmic",
) -> FFDFactoryResult:
    """Create a piecewise FFD cost function with initial parameters.

    This factory function creates a complete optimization setup including:
    - A jitted cost function with explicit static/traced boundaries
    - Initial traced parameters for optimization
    - A function to create mapping closures from optimized params

    The cost function uses stop_gradient to freeze hyperparameters while
    keeping them traced, allowing barrier method iterations without
    recompilation.

    Args:
        region_meshes: List of meshes for each FFD region
        all_n_cp: List of (n_cp_x, n_cp_y) per region
        all_imposed_displacements_masks: BC masks per region. If None,
            boundary_conditions must be specified to auto-generate masks.
        initial_mapping_function_w: Initial W mapping guess
        initial_mapping_function_t: Initial T mapping guess
        alpha_A_algn: Weight for field alignment cost
        alpha_B_bij: Weight for mapping synchronization cost
        alpha_C_jac: Weight for Jacobian barrier cost
        alpha_D_cp_norm: Weight for control point regularization
        barrier_epsilon: Initial barrier parameter
        u0_values: Field 0 values on main mesh
        u1_values: Field 1 values on main mesh
        alpha_G_got_w2: Weight for W2 distance cost (default: 0)
        alpha_H_inter_sync: Weight for inter-region sync (default: 0)
        alpha_I_inter_jac_sync: Weight for inter-region Jacobian sync (default: 0)
        alpha_J_point_registration: Weight for point registration cost (default: 0)
        inter_region_sync_indices: Region pairs to synchronize (default: ())
        inter_region_sync_points: Sync points for each pair (default: ())
        point_registration_points_0: Source landmarks in field 0 domain (default: None)
        point_registration_points_1: Target landmarks in field 1 domain (default: None)
        initialisation_noise_factor: Noise for initial params (default: 0)
        boundary_conditions: Type of boundary conditions to apply:
            - "slip_on_boundaries": Apply slip conditions to all region boundaries
            - "slip_on_external_boundaries": Apply slip only to external boundaries,
              leaving internal boundaries free for continuity
            - None: Must provide all_imposed_displacements_masks manually

    Returns:
        FFDFactoryResult with cost_fn, init_params, and make_mappings

    Example:
        >>> result = make_piecewise_ffd_cost_function(
        ...     region_meshes=[mesh1, mesh2],
        ...     all_n_cp=[(4, 4), (4, 4)],
        ...     all_imposed_displacements_masks=None,
        ...     initial_mapping_function_w=identity_mapping,
        ...     initial_mapping_function_t=identity_mapping,
        ...     alpha_A_algn=1.0,
        ...     alpha_B_bij=0.1,
        ...     alpha_C_jac=1.0,
        ...     alpha_D_cp_norm=0.01,
        ...     barrier_epsilon=1.0,
        ...     u0_values=u0,
        ...     u1_values=u1,
        ...     boundary_conditions="slip_on_external_boundaries",
        ... )
        >>>
        >>> # Optimize
        >>> from phdtruel.mappings.functional_solver import solve
        >>> solve_result = solve(result.cost_fn, result.init_params)
        >>>
        >>> # Get mappings
        >>> w_map, t_map = result.make_mappings(solve_result.params)
    """
    # Validate and generate boundary condition masks if needed
    if all_imposed_displacements_masks is None and boundary_conditions is None:
        raise ValueError(
            "Either all_imposed_displacements_masks or boundary_conditions must be provided"
        )

    # Initialize with provided masks or generate from boundary_conditions
    masks_for_building: list[Float[Array, "n_cp_x n_cp_y 2"]]

    if all_imposed_displacements_masks is not None:
        # Use provided masks directly
        masks_for_building = all_imposed_displacements_masks
    else:
        # Generate masks from boundary_conditions string
        from phdtruel.fields.meshes import RegularGrid
        from phdtruel.mappings.boundary_conditions import (
            create_slip_on_external_boundaries_masks,
            initialize_all_slip_masks,
        )

        # Convert MeshJaxed to RegularGrid for boundary condition utilities
        region_meshes_regular = []
        for mesh_jaxed in region_meshes:
            region_meshes_regular.append(
                RegularGrid([np.array(ax) for ax in mesh_jaxed.axes])
            )

        # Build inter-region sync data from indices and points
        inter_region_sync = None
        if len(inter_region_sync_indices) > 0:
            inter_region_sync = [
                (i, j, np.array(pts))
                for (i, j), pts in zip(
                    inter_region_sync_indices, inter_region_sync_points
                )
            ]

        # Generate numpy masks based on boundary condition type
        if boundary_conditions == "slip_on_boundaries":
            numpy_masks = initialize_all_slip_masks(all_n_cp)
        elif boundary_conditions == "slip_on_external_boundaries":
            numpy_masks = create_slip_on_external_boundaries_masks(
                all_n_cp,
                region_meshes_regular,
                inter_region_sync,
                tolerance=None,  # Will auto-compute from mesh
            )
        else:
            raise ValueError(
                f"Unknown boundary_conditions: {boundary_conditions}. "
                "Supported: 'slip_on_boundaries', 'slip_on_external_boundaries'"
            )

        # Convert numpy masks to JAX arrays
        masks_for_building = [jnp.array(mask) for mask in numpy_masks]

    # Build FFD parameters for each region
    all_ffd_parameters = []
    for mesh, (n_cp_x, n_cp_y) in zip(region_meshes, all_n_cp):
        # Get bounds from mesh points
        points_np = np.array(mesh.points)
        bounds_min = points_np.min(axis=0)
        bounds_max = points_np.max(axis=0)
        lengths = bounds_max - bounds_min

        ffd_params = FFDParameters(
            box_origin=jnp.array(bounds_min),
            box_length=jnp.array(lengths),
            n_control_points=jnp.array([n_cp_x, n_cp_y]),
        )
        all_ffd_parameters.append(ffd_params)

    all_ffd_parameters = tuple(all_ffd_parameters)

    # Determine the global mesh.
    # If a full_domain_mesh is provided, use it; otherwise fall back to the
    # first region mesh (valid only for single-region setups).
    if full_domain_mesh is not None:
        global_mesh = full_domain_mesh
    else:
        global_mesh = region_meshes[0]

    # Compute per-region field values by indexing into the full-domain arrays.
    # For each region, find which global mesh points fall within that region
    # and extract the corresponding field values.
    u0_per_region_list: list[Float[Array, " n_region"]] = []
    u1_per_region_list: list[Float[Array, " n_region"]] = []

    if full_domain_mesh is not None:
        full_pts = np.array(full_domain_mesh.points)  # (N, 2)
        from scipy.spatial import KDTree

        tree = KDTree(full_pts)
        for region_mesh in region_meshes:
            region_pts = np.array(region_mesh.points)  # (M, 2)
            _, idx = tree.query(region_pts)
            u0_per_region_list.append(u0_values[idx])
            u1_per_region_list.append(u1_values[idx])
    else:
        # Fallback: use sequential indices (valid when each region mesh
        # uses the same indexing as u0_values/u1_values).
        for region_mesh in region_meshes:
            n_pts = np.array(region_mesh.points).shape[0]
            idx = jnp.arange(n_pts)
            u0_per_region_list.append(u0_values[idx])
            u1_per_region_list.append(u1_values[idx])

    u0_per_region = tuple(u0_per_region_list)
    u1_per_region = tuple(u1_per_region_list)

    # Normalize missing point registration inputs to empty arrays so the cost
    # term is safely disabled (n_pts == 0) when no points are provided.
    if point_registration_points_0 is None:
        point_registration_points_0 = jnp.zeros((0, 2), dtype=jnp.float64)
    if point_registration_points_1 is None:
        point_registration_points_1 = jnp.zeros((0, 2), dtype=jnp.float64)

    # Build static config
    static_config = _build_static_config(
        mesh=global_mesh,
        u0_values=u0_values,
        u1_values=u1_values,
        u0_per_region=u0_per_region,
        u1_per_region=u1_per_region,
        region_meshes=region_meshes,
        all_ffd_parameters=all_ffd_parameters,
        all_imposed_displacements_masks=masks_for_building,
        inter_region_sync_indices=inter_region_sync_indices,
        inter_region_sync_points=inter_region_sync_points,
        point_registration_points_0=point_registration_points_0,
        point_registration_points_1=point_registration_points_1,
        alpha_A_algn=alpha_A_algn,
        alpha_B_bij=alpha_B_bij,
        alpha_C_jac=alpha_C_jac,
        alpha_D_cp_norm=alpha_D_cp_norm,
        alpha_G_got_w2=alpha_G_got_w2,
        alpha_H_inter_sync=alpha_H_inter_sync,
        alpha_I_inter_jac_sync=alpha_I_inter_jac_sync,
        alpha_J_point_registration=alpha_J_point_registration,
    )

    # Build initial params
    init_params = _build_initial_params(
        all_ffd_parameters=all_ffd_parameters,
        all_n_cp=all_n_cp,
        initial_mapping_function_w=initial_mapping_function_w,
        initial_mapping_function_t=initial_mapping_function_t,
        initialisation_noise_factor=initialisation_noise_factor,
        alpha_A_algn=alpha_A_algn,
        alpha_B_bij=alpha_B_bij,
        alpha_C_jac=alpha_C_jac,
        alpha_D_cp_norm=alpha_D_cp_norm,
        alpha_G_got_w2=alpha_G_got_w2,
        alpha_H_inter_sync=alpha_H_inter_sync,
        alpha_I_inter_jac_sync=alpha_I_inter_jac_sync,
        alpha_J_point_registration=alpha_J_point_registration,
        barrier_epsilon=barrier_epsilon,
        barrier_fn_name=barrier_fn_name,
    )

    # Create make_mappings function
    make_mappings = _create_mappings(static_config)
    get_displacements = _create_displacement_extractor(static_config)

    def traced_only_cost_fn(opt_params: FFDTracedParams):
        """Cost function that only depends on traced parameters, with static config frozen."""
        return piecewise_ffd_cost_function(
            opt_params,
            static_config,
        )

    def traced_only_residual_fn(opt_params: FFDTracedParams):
        """Residual vectors for natural-gradient / Gauss-Newton optimization."""
        return piecewise_ffd_residual_function(
            opt_params,
            static_config,
        )

    # Return result tuple
    return FFDFactoryResult(
        cost_fn=traced_only_cost_fn,  # returns (total_cost, aux_dict)
        residual_fn=traced_only_residual_fn,
        init_params=init_params,
        make_mappings=make_mappings,
        get_displacements=get_displacements,
    )
