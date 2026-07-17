"""Barycenter FFD functional cost function types and configuration."""

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
            One of "logarithmic", "exponential", etc.
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
class BariFFDStaticConfig:
    """Static configuration for Wasserstein-like barycenter FFD objective."""

    mesh: MeshJaxed
    region_meshes: tuple[MeshJaxed, ...]
    all_ffd_parameters: tuple[FFDParameters, ...]
    all_imposed_displacements_masks: tuple[Float[Array, "n_cp_x n_cp_y 2"], ...]

    # Per-field values on global mesh.
    all_u_values: tuple[Float[Array, " n_points"], ...]
    # Per-field, per-region values for piecewise interpolation.
    all_u_per_region: tuple[tuple[Float[Array, " n_region"], ...], ...]

    # Barycenter weights and objective coefficients.
    gamma: Float[Array, " n_fields"]
    alpha_A_w_id: float
    alpha_B_pairwise_align: float
    alpha_C_log_barrier_jac: float
    alpha_D_pairwise_w2: float
    alpha_E_bary_w2: float


class BariFFDTracedParams(NamedTuple):
    """Traced parameters for barycenter optimization.

    ``displacements_w`` is indexed as ``[field][region]``.
    """

    displacements_w: tuple[tuple[Float[Array, "n_cp_x n_cp_y 2"], ...], ...]

    # Hyperparameters are traced but frozen in the cost via stop_gradient.
    alpha_A_w_id: Float[Array, ""]
    alpha_B_pairwise_align: Float[Array, ""]
    alpha_C_log_barrier_jac: Float[Array, ""]
    alpha_D_pairwise_w2: Float[Array, ""]
    alpha_E_bary_w2: Float[Array, ""]

    # Barrier configuration for the log-barrier Jacobian term (alpha_C).
    log_barrier_config: BarrierFnConfig
