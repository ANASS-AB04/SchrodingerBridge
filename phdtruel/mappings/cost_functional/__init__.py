"""Pure functional cost functions for field mapping optimization.

This module provides functional alternatives to equinox-based cost functions,
with explicit static/traced boundaries using JAX's jit and stop_gradient.

The main API is the factory pattern that returns a ready-to-use cost function:

    from phdtruel.mappings.cost_functional import make_piecewise_ffd_cost_function

    result = make_piecewise_ffd_cost_function(
        region_meshes=region_meshes,
        all_n_cp=[(5, 5), (5, 5)],  # control points per region
        all_imposed_displacements_masks=masks,
        initial_mapping_function_w=initial_w,
        initial_mapping_function_t=initial_t,
        alpha_A_algn=1.0,  # alignment weight
        alpha_B_bij=0.1,   # bijectivity weight
        alpha_C_jac=0.1,   # jacobian barrier weight
        barrier_epsilon=1.0,
        u0_values=u0,
        u1_values=u1,
    )

    # Run optimization
    from phdtruel.mappings.functional_solver import solve

    solve_result = solve(result.cost_fn, result.init_params)

    # Extract mappings
    mapping_w, mapping_t = result.make_mappings(solve_result.params)

**Design:**
- Static config is a frozen dataclass (FFDStaticConfig)
- Traced params are a NamedTuple (FFDTracedParams)
- Factory pattern instead of direct instantiation
- Barrier epsilon updates staged via `phdtruel.mappings.functional_solver.solve_barrier`

See Also:
    - scripts/minimized_mappings/piecewise_ffd_in_square.py: Working example
"""

from phdtruel.mappings.cost_functional.ffd import (
    make_piecewise_ffd_cost_function,
)
from phdtruel.mappings.cost_functional.baricenter_ffd import (
    BariFFDFactoryResult,
    BariFFDStaticConfig,
    BariFFDTracedParams,
    make_wasserstein_barycenter_ffd_cost_function,
)
from phdtruel.mappings.cost_functional.ffd.factory import (
    FFDFactoryResult,
)
from phdtruel.mappings.cost_functional.ffd.types import (
    FFDStaticConfig,
    FFDTracedParams,
)
from phdtruel.mappings.cost_functional.types import MeshJaxed

__all__ = [
    "MeshJaxed",
    "FFDStaticConfig",
    "FFDTracedParams",
    "make_piecewise_ffd_cost_function",
    "FFDFactoryResult",
    "BariFFDStaticConfig",
    "BariFFDTracedParams",
    "make_wasserstein_barycenter_ffd_cost_function",
    "BariFFDFactoryResult",
]
