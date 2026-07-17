"""RBF functional cost function types and configuration."""

from dataclasses import dataclass
from typing import Literal, NamedTuple

from jax import Array
from jaxtyping import Float

from phdtruel.mappings.cost_functional.ffd.types import BarrierFnConfig
from phdtruel.mappings.cost_functional.types import MeshJaxed

FieldInterpKind = Literal["grid", "rbf", "nearest", "linear"]


@dataclass(frozen=True)
class RBFStaticConfig:
    """Static configuration for RBF cost function (immutable).

    These values are passed as static arguments to the jitted cost function
    and never change during optimization.

    Attributes:
        mesh: Global domain mesh for interpolation, alignment, jacobian, sync.
        u0_values: Field 0 values on the global mesh.
        u1_values: Field 1 values on the global mesh.
        centers: Fixed RBF center coordinates for W and T mappings.
        imposed_disp_points: Points where displacement is imposed (boundary).
        imposed_disp_targets: Target displacements at imposed points (zeros).
        rbf_kernel: Kernel for mapping RBF interpolators.
        rbf_neighbors: Neighbors for mapping RBF (None = all centers).
        rbf_epsilon: Shape parameter for mapping RBF kernel.
        rbf_smoothing: Smoothing for mapping RBF.
        rbf_degree: Polynomial degree for mapping RBF.
        field_interp_kind: How to interpolate field values at mapped points.
        field_rbf_neighbors: Neighbors for field RBF on the full mesh.
        field_rbf_kernel: Kernel for field RBF (defaults to rbf_kernel).
        alpha_*: Cost term weights (static reference values).
        enable_w2_cost: Whether to compute W2 distance cost.
    """

    mesh: MeshJaxed
    u0_values: Float[Array, " n"]
    u1_values: Float[Array, " n"]

    centers: Float[Array, "n_c 2"]
    imposed_disp_points: Float[Array, "n_bc 2"]
    imposed_disp_targets: Float[Array, "n_bc 2"]

    # Point registration points
    point_registration_points_0: Float[Array, "n_registration 2"]
    point_registration_points_1: Float[Array, "n_registration 2"]

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
    alpha_G_got_w2: float
    alpha_J_point_registration: float

    enable_w2_cost: bool
    enable_point_registration: bool


class RBFTracedParams(NamedTuple):
    """Traced parameters for RBF optimization (PyTree).

    These values are traced through JAX computations and include both
    optimized variables (displacements) and frozen hyperparameters (alphas).

    Attributes:
        displacements_w: Displacements at RBF centers for W mapping.
        displacements_t: Displacements at RBF centers for T mapping.
        barrier_config: Barrier function configuration (name + epsilon).
        alpha_*: Cost weights (traced but frozen via stop_gradient).
    """

    displacements_w: Float[Array, "n_c 2"]
    displacements_t: Float[Array, "n_c 2"]

    barrier_config: BarrierFnConfig

    alpha_A_algn: Float[Array, ""]
    alpha_B_bij: Float[Array, ""]
    alpha_C_jac: Float[Array, ""]
    alpha_D_cp_norm: Float[Array, ""]
    alpha_E_imposed_disp: Float[Array, ""]
    alpha_G_got_w2: Float[Array, ""]
    alpha_J_point_registration: Float[Array, ""]
