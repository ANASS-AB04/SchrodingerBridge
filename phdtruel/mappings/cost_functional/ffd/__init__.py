"""Free-Form Deformation (FFD) functional cost function implementation.

This module provides a pure functional implementation of piecewise FFD cost functions
for optimizing mappings between fields.

**Key Components:**
- `make_piecewise_ffd_cost_function`: Factory function to create cost function
- `FFDStaticConfig`: Frozen dataclass for static configuration
- `FFDTracedParams`: NamedTuple for traced optimization parameters
- `piecewise_ffd_cost_function`: Core cost function (usually accessed via factory)

**Basic Usage:**
    >>> from phdtruel.mappings.cost_functional.ffd import make_piecewise_ffd_cost_function
    >>> from phdtruel.mappings.functional_solver import solve
    >>>
    >>> result = make_piecewise_ffd_cost_function(
    ...     region_meshes=[mesh1, mesh2],
    ...     all_n_cp=[(5, 5), (5, 5)],
    ...     all_imposed_displacements_masks=[mask1, mask2],
    ...     initial_mapping_function_w=initial_w,
    ...     initial_mapping_function_t=initial_t,
    ...     alpha_A_algn=1.0,
    ...     u0_values=u0,
    ...     u1_values=u1,
    ... )
    >>>
    >>> # Optimize
    >>> solve_result = solve(result.cost_fn, result.init_params)
    >>>
    >>> # Get mappings
    >>> mapping_w, mapping_t = result.make_mappings(solve_result.params)

**Barrier Method:**
    >>> from phdtruel.mappings.functional_solver import solve_barrier
    >>>
    >>> solve_result = solve_barrier(
    ...     cost_fn=result.cost_fn,
    ...     init_params=result.init_params,
    ...     updates=[
    ...         {"barrier_epsilon": 1.0},
    ...         {"barrier_epsilon": 0.1},
    ...         {"barrier_epsilon": 0.01},
    ...     ],
    ... )

See Also:
    - Factory implementation: :mod:`phdtruel.mappings.cost_functional.ffd.factory`
    - Type definitions: :mod:`phdtruel.mappings.cost_functional.ffd.types`
    - Cost function: :mod:`phdtruel.mappings.cost_functional.ffd.cost_fn`
"""

from phdtruel.mappings.cost_functional.ffd.cost_fn import piecewise_ffd_cost_function
from phdtruel.mappings.cost_functional.ffd.factory import (
    FFDFactoryResult,
    make_piecewise_ffd_cost_function,
)
from phdtruel.mappings.cost_functional.ffd.residuals import (
    piecewise_ffd_residual_function,
)
from phdtruel.mappings.cost_functional.ffd.types import (
    FFDStaticConfig,
    FFDTracedParams,
)

__all__ = [
    "FFDStaticConfig",
    "FFDTracedParams",
    "FFDFactoryResult",
    "make_piecewise_ffd_cost_function",
    "piecewise_ffd_cost_function",
    "piecewise_ffd_residual_function",
]
