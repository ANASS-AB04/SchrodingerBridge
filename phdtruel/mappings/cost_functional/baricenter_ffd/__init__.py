"""Functional Wasserstein-like barycenter FFD objective."""

from phdtruel.mappings.cost_functional.baricenter_ffd.cost_fn import (
    wasserstein_barycenter_cost_function,
)
from phdtruel.mappings.cost_functional.baricenter_ffd.factory import (
    BariFFDFactoryResult,
    make_wasserstein_barycenter_ffd_cost_function,
)
from phdtruel.mappings.cost_functional.baricenter_ffd.residuals import (
    wasserstein_barycenter_residual_function,
)
from phdtruel.mappings.cost_functional.baricenter_ffd.types import (
    BariFFDStaticConfig,
    BariFFDTracedParams,
)

__all__ = [
    "BariFFDStaticConfig",
    "BariFFDTracedParams",
    "BariFFDFactoryResult",
    "make_wasserstein_barycenter_ffd_cost_function",
    "wasserstein_barycenter_cost_function",
    "wasserstein_barycenter_residual_function",
]
