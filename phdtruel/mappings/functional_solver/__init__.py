"""Functional solver for optimisation-based field mappings.

This subpackage provides a pure-functional API for iterative optimisation:

- :func:`make_solver` — compile-once factory returning a ``solve()`` callable
- :func:`solve` — one-shot convenience wrapper (default L-BFGS)
- :func:`solve_barrier` — barrier-method outer loop with staged updates
- :func:`solve_with_checkpointing` — resumable solver with on-disk checkpoints

- :func:`natural_gradient` — matrix-free Energy Natural Gradient (Gauss-Newton) optimizer
- :func:`conjugate_gradient` — PCG helper used by natural gradient

Core types:

- :class:`SolveResult` — result NamedTuple (params, history, stop_reason, …)
- :class:`StopCriteria` — convergence thresholds dataclass
- :class:`Checkpoint` — on-disk checkpoint state

Quick-start example::

    import jax.numpy as jnp
    from phdtruel.mappings.functional_solver import solve, StopCriteria

    def cost(params):
        value = jnp.sum(params ** 2)
        return value, {"loss": value}

    result = solve(cost, jnp.ones(5), stop_criteria=StopCriteria(max_iter=200))
    print(result.stop_reason)        # "grad_norm_abs"
    print(result.history["cost"])    # np.ndarray shape (n_iters,)
    print(result.n_iters)            # number of steps taken

For multi-call efficiency (avoids recompiling the JIT step)::

    from phdtruel.mappings.functional_solver import make_solver

    solver = make_solver(cost)
    result1 = solver(jnp.zeros(5))
    result2 = solver(result1.params, init_state=result1.opt_state)  # warm start

Barrier method::

    from phdtruel.mappings.functional_solver import solve_barrier

    result = solve_barrier(
        cost_fn=cost,
        init_params=jnp.zeros(n),
        updates=[
            {"barrier_config.barrier_epsilon": jnp.array(1.0)},
            {"barrier_config.barrier_epsilon": jnp.array(0.1)},
        ],
    )

Checkpointing::

    from pathlib import Path
    from phdtruel.mappings.functional_solver import solve_with_checkpointing

    result = solve_with_checkpointing(
        cost_fn=cost,
        init_params=jnp.zeros(n),
        checkpoint_path=Path("run.pkl"),
        checkpoint_freq=100,
        resume=True,
    )

Natural gradient (Gauss-Newton on sum-of-squared residuals)::

    from phdtruel.mappings.functional_solver import make_solver, natural_gradient

    def residual_fn(params):
        return {"r": params - target}

    solver = make_solver(
        cost_fn,
        opt=natural_gradient(residual_fn=residual_fn, linesearch="zoom"),
    )
    result = solver(init_params)
"""

from typing import Any, Callable

import optax

from .barrier_method_solver import solve_barrier
from .natural_gradient import conjugate_gradient, natural_gradient
from .opt_resolver import make_solver_opt_resolver
from .solver import check_stop_criteria, make_solver
from .types import Checkpoint, SolveResult, StopCriteria


def solve(
    cost_fn: Callable,
    init_params: Any,
    *,
    opt: optax.GradientTransformationExtraArgs | None = None,
    stop_criteria: StopCriteria | None = None,
    has_aux: bool = True,
    init_state: Any | None = None,
) -> SolveResult:
    """One-shot convenience wrapper around ``make_solver``."""
    solver = make_solver(
        cost_fn=cost_fn,
        opt=opt,
        stop_criteria=stop_criteria,
        has_aux=has_aux,
    )
    return solver(init_params, init_state=init_state)


__all__ = [
    # core
    "make_solver",
    "check_stop_criteria",
    # natural gradient
    "natural_gradient",
    "conjugate_gradient",
    # convenience
    "solve",
    # barrier
    "solve_barrier",
    "make_solver_opt_resolver",
    # checkpointing
    # types
    "SolveResult",
    "StopCriteria",
    "Checkpoint",
]
