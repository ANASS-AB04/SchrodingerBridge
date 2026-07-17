"""RBF free control-point cost function implementation.

This module provides a variant of the RBF cost function where interior
control-point positions are optimized separately for W and T mappings.
Boundary center positions remain pinned.

**Key Components:**
- `make_rbf_free_cp_cost_function`: Factory function to create cost function
- `RBFFreeStaticConfig`: Frozen dataclass for static configuration
- `RBFFreeTracedParams`: NamedTuple for traced optimization parameters
- `rbf_free_cp_cost_function`: Core cost function (usually accessed via factory)

**Basic Usage:**
    >>> from phdtruel.mappings.cost_functional.rbf_free_cp import (
    ...     make_rbf_free_cp_cost_function,
    ... )
    >>>
    >>> result = make_rbf_free_cp_cost_function(
    ...     mesh=mesh_jaxed,
    ...     u0_values=u0,
    ...     u1_values=u1,
    ...     boundary_centers=boundary_centers,
    ...     interior_centers=interior_centers,
    ...     imposed_disp_points=boundary_centers,
    ...     initial_mapping_function_w=initial_w,
    ...     initial_mapping_function_t=initial_t,
    ...     alpha_A_algn=1.0,
    ...     alpha_B_bij=1.0,
    ...     alpha_C_jac=1.0,
    ...     alpha_D_cp_norm=0.01,
    ...     alpha_E_imposed_disp=1.0,
    ...     barrier_epsilon=1.0,
    ... )
    >>>
    >>> params, state = run_opt(result.init_params, result.cost_fn, opt=optax.lbfgs())
    >>> mapping_w, mapping_t = result.make_mappings(params)
"""

from phdtruel.mappings.cost_functional.rbf_free_cp.cost_fn import (
    rbf_free_cp_cost_function,
)
from phdtruel.mappings.cost_functional.rbf_free_cp.factory import (
    RBFFreeFactoryResult,
    make_rbf_free_cp_cost_function,
)
from phdtruel.mappings.cost_functional.rbf_free_cp.types import (
    RBFFreeStaticConfig,
    RBFFreeTracedParams,
)

__all__ = [
    "RBFFreeStaticConfig",
    "RBFFreeTracedParams",
    "RBFFreeFactoryResult",
    "make_rbf_free_cp_cost_function",
    "rbf_free_cp_cost_function",
]
