"""Utility functions for cost_functional module.

This module provides helper functions used by the functional cost function
implementation, including barrier functions and boundary condition utilities.

Functions:
    logarithmic_barrier_function: Logarithmic barrier for Jacobian positivity.
    exponential_barrier_function: Exponential barrier for Jacobian positivity.
    get_barrier_fn: Resolve a barrier function by name.
    impose_control_displacements: Apply BC masks to control point displacements.

Example:
    >>> from phdtruel.mappings.cost_functional.utils import (
    ...     get_barrier_fn,
    ...     impose_control_displacements,
    ... )
    >>> import jax.numpy as jnp
    >>>
    >>> # Resolve and apply barrier to Jacobian determinants
    >>> jacobians = jnp.array([0.5, 1.0, 2.0])
    >>> barrier_fn = get_barrier_fn("logarithmic")
    >>> barrier = barrier_fn(jacobians, barrier_epsilon=0.1)
    >>>
    >>> # Apply boundary conditions (NaN = free, value = constrained)
    >>> displacements = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    >>> mask = jnp.array([[0.0, jnp.nan], [jnp.nan, 0.0]])
    >>> constrained = impose_control_displacements(displacements, mask)
"""

from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

BarrierFnName = Literal[
    "logarithmic",
    "exponential",
    "poly_2",
    "null",
    "sigmoid",
    "linear_then_log",
    "relu",
]


def sigmoid_barrier_function(
    x: Float[Array, "..."],
    barrier_epsilon: float | Float[Array, ""],
    y_for_negatives: float = 1.0e12,
) -> Float[Array, "..."]:
    """
    Recommanded epsilon between 0.1 and 0.001
    """
    y = -1.0 / (1 + jnp.exp(-x / barrier_epsilon)) + 1.0
    y = y * y_for_negatives
    return y


def linear_then_log_barrier_function(
    x: Float[Array, "..."],
    barrier_epsilon: float | Float[Array, ""],
    y_for_negatives: float = 1.0e12,
) -> Float[Array, "..."]:
    x_transition = jnp.exp(-jnp.sqrt(y_for_negatives) / barrier_epsilon)
    y_linear = -x / barrier_epsilon + y_for_negatives
    y_log = (-barrier_epsilon * jnp.log(x)) ** 2
    y = jnp.where(x > x_transition, y_log, y_linear)
    return y


def logarithmic_barrier_function(
    x: Float[Array, "..."],
    barrier_epsilon: float | Float[Array, ""],
    y_for_negatives: float = 1.0e12,
) -> Float[Array, "..."]:
    """Logarithmic barrier function.

    Small epsilon means steep barrier.

    Args:
        x: Input values (typically Jacobian determinants)
        barrier_epsilon: Barrier parameter (smaller = steeper barrier)
        tol: Tolerance for checking positivity

    Returns:
        Barrier function values
    """
    # Avoid evaluating log on invalid entries to keep value/grad finite.
    safe_x = jnp.where(x > 0.0, x, 1.0)
    y = (-barrier_epsilon * jnp.log10(safe_x)) ** 2
    y = jnp.where(x <= 0.0, -x + y_for_negatives, y)
    return y


def exponential_barrier_function(
    x: Float[Array, "..."],
    barrier_epsilon: float | Float[Array, ""],
) -> Float[Array, "..."]:
    """Exponential barrier function.

    Small epsilon means steep barrier.

    Args:
        x: Input values (typically Jacobian determinants)
        barrier_epsilon: Barrier parameter (smaller = steeper barrier)
        tol: Tolerance for checking positivity

    Returns:
        Barrier function values
    """
    y = jnp.exp(-2 * x / barrier_epsilon)
    return y


def polynomial_deg_2_barrier_function(
    x: Float[Array, "..."],
    barrier_epsilon: float | Float[Array, ""],
) -> Float[Array, "..."]:
    """Polynomial barrier function of degree 2 centered at one.

    Small epsilon means steep barrier.

    Args:
        x: Input values (typically Jacobian determinants)
        barrier_epsilon: Barrier parameter (smaller = steeper barrier)
        tol: Tolerance for checking positivity

    Returns:
        Barrier function values
    """
    y = (x - 1) ** 2 / barrier_epsilon
    return y


def homographic_barrier_function(
    x: Float[Array, "..."],
    eps: float = 1.0,
) -> Float[Array, "..."]:
    """Homographic barrier function.

    Args:
        x: Input values (typically Jacobian determinants)
        c: Wall position (barrier is infinite at x <= c)
        eps: Barrier parameter

    Returns:
        Barrier function values
    """
    y = -eps * jnp.log(x)
    y = jnp.where(x <= 0.0, -x, y)
    return y


def relu_barrier_function(
    x: Float[Array, "..."],
    shift: float = 0.01,
) -> Float[Array, "..."]:
    """ReLU barrier function (zero for positive inputs, linear for negatives).

    Args:
        x: Input values (typically Jacobian determinants)

    Returns:
        Barrier function values
    """
    y = jnp.where(x > shift, 0.0, -x)
    return y


def get_barrier_fn(
    name: BarrierFnName,
) -> Callable[[Float[Array, "..."], float | Float[Array, ""]], Float[Array, "..."]]:
    """Resolve a barrier function by name.

    This is a pure Python dispatch — the name is resolved at trace time,
    so JAX sees a concrete callable. Changing the name triggers recompilation.

    Args:
        name: One of "logarithmic" or "exponential"

    Returns:
        The corresponding barrier function callable

    Raises:
        ValueError: If name is not a known barrier function
    """
    match name:
        case "logarithmic":
            return logarithmic_barrier_function
        case "exponential":
            return exponential_barrier_function
        case "poly_2":
            return polynomial_deg_2_barrier_function
        case "null":
            return lambda x, eps: jnp.zeros_like(x)
        case "sigmoid":
            return sigmoid_barrier_function
        case "linear_then_log":
            return linear_then_log_barrier_function
        case _:
            raise ValueError(
                f"Unknown barrier function name: {name!r}. "
                f"Expected one of: 'logarithmic', 'exponential', 'poly_2', 'sigmoid', 'linear_then_log'."
            )


def impose_control_displacements(
    control_displacement: Float[Array, "*"],
    imposed_displacements_where_not_nan: Float[Array, "*"],
) -> Float[Array, "*"]:
    """Apply boundary condition masks to control point displacements.

    Replaces values where mask is not NaN with the mask value.
    Keeps original values where mask is NaN.

    Args:
        control_displacement: Original control point displacements
        imposed_displacements_where_not_nan: Mask with NaN for free DOFs,
            actual values for constrained DOFs

    Returns:
        Displacements with boundary conditions applied
    """
    imposed_clean = jnp.nan_to_num(imposed_displacements_where_not_nan, nan=0.0)
    return jnp.where(
        jnp.isnan(imposed_displacements_where_not_nan),
        control_displacement,
        imposed_clean,
    )


if __name__ == "__main__":
    # Demo plot of barriers
    import os

    # CPU
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

    print(f"{float(logarithmic_barrier_function(-20, 0.1))}")
