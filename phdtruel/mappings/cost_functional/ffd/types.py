"""FFD functional cost function types and configuration."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
from jax import Array
from jaxtyping import Float

from phdtruel.mappings.cost_functional.types import MeshJaxed
from phdtruel.mappings.cost_functional.utils import BarrierFnName
from phdtruel.mappings.ffd import FFDParameters


class BarrierFnConfig:
    """Configuration for the Jacobian barrier function.

    Groups all barrier-related parameters. Registered as a custom JAX pytree
    so that barrier_fn_name is treated as static auxiliary data (triggers
    recompilation on change) while barrier_epsilon is a traceable JAX array
    leaf that can be updated freely without recompilation.

    Attributes:
        barrier_fn_name: Name of the barrier function to use.
            One of "logarithmic" or "exponential".
        barrier_epsilon: Barrier steepness parameter. Smaller values produce
            a steeper barrier. Can be updated without recompilation.
    """

    def __init__(
        self,
        barrier_fn_name: BarrierFnName,
        barrier_epsilon: Float[Array, ""],
    ) -> None:
        self.barrier_fn_name: BarrierFnName = barrier_fn_name
        self.barrier_epsilon: Float[Array, ""] = barrier_epsilon

    def _replace(self, **kwargs: object) -> "BarrierFnConfig":
        """Return a new BarrierFnConfig with updated fields (NamedTuple-style)."""
        return BarrierFnConfig(
            barrier_fn_name=kwargs.get("barrier_fn_name", self.barrier_fn_name),  # type: ignore[arg-type]
            barrier_epsilon=kwargs.get("barrier_epsilon", self.barrier_epsilon),  # type: ignore[arg-type]
        )

    def __repr__(self) -> str:
        return (
            f"BarrierFnConfig("
            f"barrier_fn_name={self.barrier_fn_name!r}, "
            f"barrier_epsilon={self.barrier_epsilon})"
        )


# Register BarrierFnConfig as a custom JAX pytree.
# barrier_fn_name → auxiliary (static): changes trigger recompilation
# barrier_epsilon → leaf: a JAX array, updated freely without recompilation
jax.tree_util.register_pytree_node(
    BarrierFnConfig,
    flatten_func=lambda cfg: ([cfg.barrier_epsilon], cfg.barrier_fn_name),
    unflatten_func=lambda aux, leaves: BarrierFnConfig(
        barrier_fn_name=aux,  # type: ignore[assignment]
        barrier_epsilon=leaves[0],
    ),
)


@dataclass(frozen=True)
class FFDStaticConfig:
    """Static configuration for FFD cost function (immutable).

    These values are passed as static arguments to the jitted cost function
    and never change during optimization.

    Attributes:
        region_meshes: Tuple of meshes for each FFD region
        all_ffd_parameters: FFD parameters for each region
        all_imposed_displacements_masks: BC masks per region
        inter_region_sync_indices: Indices of regions to synchronize
        alpha_A_algn: Weight for field alignment cost
        alpha_B_bij: Weight for mapping synchronization cost
        alpha_C_jac: Weight for Jacobian barrier cost
        alpha_D_cp_norm: Weight for control point regularization
        alpha_G_got_w2: Weight for W2 distance cost
        alpha_H_inter_sync: Weight for inter-region sync cost
        alpha_I_inter_jac_sync: Weight for inter-region Jacobian sync
        enable_w2_cost: Whether to compute W2 distance cost
        enable_inter_sync: Whether to compute inter-region sync cost
    """

    # Global domain mesh (used for interpolation, alignment, jacobian, sync)
    mesh: MeshJaxed

    # Field values on the global mesh
    u0_values: Float[Array, " n"]
    u1_values: Float[Array, " n"]

    # Per-region field values (for piecewise interpolation)
    u0_per_region: tuple[Float[Array, " n_region"], ...]
    u1_per_region: tuple[Float[Array, " n_region"], ...]

    # Per-region meshes (used for piecewise interpolation in _u0/_u1)
    region_meshes: tuple[MeshJaxed, ...]
    all_ffd_parameters: tuple[FFDParameters, ...]
    all_imposed_displacements_masks: tuple[Float[Array, "n_cp_x n_cp_y 2"], ...]
    inter_region_sync_indices: tuple[tuple[int, int], ...]
    # Inter-region synchronization points (shared boundary points per region pair)
    inter_region_sync_points: tuple[Float[Array, "n_sync 2"], ...]

    # Point registration points
    point_registration_points_0: Float[Array, "n_registration 2"]
    point_registration_points_1: Float[Array, "n_registration 2"]

    # Cost weights
    alpha_A_algn: float
    alpha_B_bij: float
    alpha_C_jac: float
    alpha_D_cp_norm: float
    alpha_G_got_w2: float
    alpha_H_inter_sync: float
    alpha_I_inter_jac_sync: float
    alpha_J_point_registration: float

    # Feature flags
    enable_w2_cost: bool
    enable_inter_sync: bool
    enable_point_registration: bool


class FFDTracedParams(NamedTuple):
    """Traced parameters for FFD optimization (PyTree).

    These values are traced through JAX computations and include both
    optimized variables (displacements) and frozen hyperparameters (alphas).

    Attributes:
        displacements_w: Control point displacements for W mapping per region
        displacements_t: Control point displacements for T mapping per region
        barrier_config: Barrier function configuration (name + epsilon).
            barrier_epsilon can be updated without recompilation;
            barrier_fn_name change triggers recompilation.
        alpha_A_algn: Field alignment weight (traced but frozen via stop_gradient)
        alpha_B_bij: Mapping sync weight (traced but frozen)
        alpha_C_jac: Jacobian barrier weight (traced but frozen)
        alpha_D_cp_norm: Control point reg weight (traced but frozen)
        alpha_G_got_w2: W2 distance weight (traced but frozen)
        alpha_H_inter_sync: Inter-region sync weight (traced but frozen)
        alpha_I_inter_jac_sync: Inter-region Jacobian sync weight (traced but frozen)
        alpha_J_point_registration: Point registration weight (traced but frozen)
    """

    displacements_w: tuple[Float[Array, "n_cp_x n_cp_y 2"], ...]
    displacements_t: tuple[Float[Array, "n_cp_x n_cp_y 2"], ...]

    # Barrier configuration - epsilon can be updated without recompilation
    barrier_config: BarrierFnConfig

    # Hyperparameters - traced but frozen via stop_gradient
    alpha_A_algn: Float[Array, ""]
    alpha_B_bij: Float[Array, ""]
    alpha_C_jac: Float[Array, ""]
    alpha_D_cp_norm: Float[Array, ""]
    alpha_G_got_w2: Float[Array, ""]
    alpha_H_inter_sync: Float[Array, ""]
    alpha_I_inter_jac_sync: Float[Array, ""]
    alpha_J_point_registration: Float[Array, ""]
