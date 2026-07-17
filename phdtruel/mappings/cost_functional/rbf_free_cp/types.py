"""RBF free control-point cost function types and configuration."""

from dataclasses import dataclass
from typing import NamedTuple

from jax import Array
from jaxtyping import Float

from phdtruel.mappings.cost_functional.ffd.types import BarrierFnConfig
from phdtruel.mappings.cost_functional.rbf.types import FieldInterpKind
from phdtruel.mappings.cost_functional.types import MeshJaxed


@dataclass(frozen=True)
class RBFFreeStaticConfig:
    """Static configuration for RBF free-CP cost function (immutable).

    Interior control-point positions are traced and optimized separately for
    W and T mappings.  Boundary centers remain pinned in this static config.

    Attributes:
        mesh: Global domain mesh for interpolation, alignment, jacobian, sync.
        u0_values: Field 0 values on the global mesh.
        u1_values: Field 1 values on the global mesh.
        boundary_centers: Pinned RBF center coordinates on the domain boundary.
        n_interior: Number of optimizable interior centers per mapping.
        imposed_disp_points: Points where displacement is imposed (boundary).
        imposed_disp_targets: Target displacements at imposed points (zeros).
        initial_interior_centers_w: Initial interior centers for W (for reg).
        initial_interior_centers_t: Initial interior centers for T (for reg).
        rbf_kernel: Kernel for mapping RBF interpolators.
        rbf_neighbors: Neighbors for mapping RBF (None = all centers).
        rbf_epsilon: Shape parameter for mapping RBF kernel.
        rbf_smoothing: Smoothing for mapping RBF.
        rbf_degree: Polynomial degree for mapping RBF.
        field_interp_kind: How to interpolate field values at mapped points.
        field_rbf_neighbors: Neighbors for field RBF on the full mesh.
        field_rbf_kernel: Kernel for field RBF (defaults to rbf_kernel).
        alpha_*: Cost term weights (static reference values).
        alpha_F_cp_pos_reg: Weight for interior center position regularization.
        enable_w2_cost: Whether to compute W2 distance cost.
        record_cp_history: Whether to record CP snapshots in aux each iteration.
    """

    mesh: MeshJaxed
    u0_values: Float[Array, " n"]
    u1_values: Float[Array, " n"]

    boundary_centers: Float[Array, "n_b 2"]
    n_interior: int
    imposed_disp_points: Float[Array, "n_bc 2"]
    imposed_disp_targets: Float[Array, "n_bc 2"]

    # Point registration points
    point_registration_points_0: Float[Array, "n_registration 2"]
    point_registration_points_1: Float[Array, "n_registration 2"]

    initial_interior_centers_w: Float[Array, "n_i 2"]
    initial_interior_centers_t: Float[Array, "n_i 2"]

    # Mapping RBF configuration (small number of centers, neighbors=None ok)
    rbf_kernel: str
    rbf_neighbors: int | None
    rbf_epsilon: float | None
    rbf_smoothing: float
    rbf_degree: int | None

    # Field interpolation configuration
    field_interp_kind: FieldInterpKind
    field_linear_frozen_points: bool
    field_rbf_neighbors: int
    field_rbf_kernel: str

    # Cost weights
    alpha_A_algn: float
    alpha_B_bij: float
    alpha_C_jac: float
    alpha_D_cp_norm: float
    alpha_E_imposed_disp: float
    alpha_F_cp_pos_reg: float
    alpha_G_got_w2: float
    alpha_J_point_registration: float

    enable_w2_cost: bool
    enable_point_registration: bool
    record_cp_history: bool


class RBFFreeTracedParams(NamedTuple):
    """Traced parameters for RBF free-CP optimization (PyTree).

    Interior center positions and displacements are optimized separately for
    W and T mappings.  Hyperparameters (alphas) are traced but frozen via
    stop_gradient.

    Attributes:
        interior_centers_w: Optimizable interior centers for W mapping.
        displacements_w: Displacements at all W centers (boundary + interior).
        interior_centers_t: Optimizable interior centers for T mapping.
        displacements_t: Displacements at all T centers (boundary + interior).
        barrier_config: Barrier function configuration (name + epsilon).
        alpha_*: Cost weights (traced but frozen via stop_gradient).
    """

    interior_centers_w: Float[Array, "n_i 2"]
    displacements_w: Float[Array, "n_c 2"]
    interior_centers_t: Float[Array, "n_i 2"]
    displacements_t: Float[Array, "n_c 2"]

    barrier_config: BarrierFnConfig

    alpha_A_algn: Float[Array, ""]
    alpha_B_bij: Float[Array, ""]
    alpha_C_jac: Float[Array, ""]
    alpha_D_cp_norm: Float[Array, ""]
    alpha_E_imposed_disp: Float[Array, ""]
    alpha_F_cp_pos_reg: Float[Array, ""]
    alpha_G_got_w2: Float[Array, ""]
    alpha_J_point_registration: Float[Array, ""]
