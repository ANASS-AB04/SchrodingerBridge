"""Radial Basis Function (RBF) functional cost function implementation.

This module provides a pure functional implementation of RBF-based cost functions
for optimizing mappings between fields using interpax RBFInterpolator.

**Key Components:**
- `make_rbf_cost_function`: Factory function to create cost function
- `RBFStaticConfig`: Frozen dataclass for static configuration
- `RBFTracedParams`: NamedTuple for traced optimization parameters
- `rbf_cost_function`: Core cost function (usually accessed via factory)

**Basic Usage:**
    >>> from phdtruel.mappings.cost_functional.rbf import make_rbf_cost_function
    >>>
    >>> result = make_rbf_cost_function(
    ...     mesh=mesh_jaxed,
    ...     u0_values=u0,
    ...     u1_values=u1,
    ...     centers=centers,
    ...     imposed_disp_points=boundary_points,
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

from phdtruel.mappings.cost_functional.rbf.cost_fn import rbf_cost_function
from phdtruel.mappings.cost_functional.rbf.factory import (
    RBFFactoryResult,
    make_rbf_cost_function,
)
from phdtruel.mappings.cost_functional.rbf.types import (
    RBFStaticConfig,
    RBFTracedParams,
)

__all__ = [
    "RBFStaticConfig",
    "RBFTracedParams",
    "RBFFactoryResult",
    "make_rbf_cost_function",
    "rbf_cost_function",
]
