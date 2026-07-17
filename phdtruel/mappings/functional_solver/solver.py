"""Core functional solver implementation.

This module provides the heart of the functional solver API:

- :func:`check_stop_criteria`: checks convergence at each step
- :func:`make_solver`: factory that JIT-compiles the optimisation step once
  and returns a ``solve(init_params, ...)`` callable

The solver supports the *has_aux* pattern where the cost function returns
``(value, aux_dict)`` alongside its scalar cost value.

Example::

    import jax.numpy as jnp
    import optax
    from phdtruel.mappings.functional_solver import make_solver, StopCriteria

    def my_cost(params):
        loss = jnp.sum(params ** 2)
        aux = {"loss": loss}
        return loss, aux

    solver = make_solver(
        cost_fn=my_cost,
        opt=optax.lbfgs(),
        stop_criteria=StopCriteria(grad_norm_abs=1e-6),
    )
    result = solver(jnp.ones(3))
    print(result.stop_reason, result.history["cost"])
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
import optax.tree_utils as otu

from .types import SolveResult, StopCriteria

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def check_stop_criteria(
    criteria: StopCriteria,
    *,
    grad_norm: float,
    grad_norm_0: float,
    value: float,
    prev_value: float,
    current_iter: int,
    aux: dict[str, Any] | None = None,
) -> str | None:
    """Check whether any stopping criterion is satisfied.

    Parameters
    ----------
    criteria:
        Convergence thresholds to test.
    grad_norm:
        L2 norm of the gradient at the current step.
    grad_norm_0:
        L2 norm of the gradient at the first step (for relative check).
    value:
        Cost function value at the current step.
    prev_value:
        Cost function value at the previous step.
    current_iter:
        Current iteration number (0-based).
    aux:
        Optional auxiliary values returned by the cost function.

    Returns
    -------
    str | None
        The name of the satisfied criterion (e.g. ``"grad_norm_abs"``),
        or ``None`` if no criterion is met yet.
    """
    return criteria.check(
        grad_norm=grad_norm,
        grad_norm_0=grad_norm_0,
        value=value,
        prev_value=prev_value,
        current_iter=current_iter,
        aux=aux,
    )


# ---------------------------------------------------------------------------
# _build_step
# ---------------------------------------------------------------------------


def _build_step(
    cost_fn: Callable,
    opt: optax.GradientTransformationExtraArgs,
    has_aux: bool,
) -> Callable:
    """Build and JIT-compile a single optimisation step.

    The returned function performs **one** gradient step and is compiled once
    regardless of how many times :func:`make_solver` is called with the
    result.

    Parameters
    ----------
    cost_fn:
        The cost function.  If *has_aux* is ``True`` it must return
        ``(value, aux_dict)``; otherwise just ``value``.
    opt:
        An Optax gradient transformation (e.g. ``optax.lbfgs()``).
    has_aux:
        Whether ``cost_fn`` returns ``(value, aux_dict)``.

    Returns
    -------
    Callable
        JIT-compiled ``step(params, opt_state) -> (params, opt_state, value, aux)``
        where ``aux`` is the ``aux_dict`` if *has_aux* else ``{}``.
    """
    # Normalise cost_fn to always return (value, aux_dict)
    if has_aux:
        _cost_fn_with_aux = cost_fn
    else:

        def _cost_fn_with_aux(params: Any) -> tuple[Any, dict]:
            return cost_fn(params), {}

    # Create a scalar wrapper for the value_fn callback in line-search.
    def _cost_scalar(params: Any) -> Any:
        v, _ = _cost_fn_with_aux(params)
        return v

    # Compute value, gradient, and auxiliary data in a single pass
    value_and_grad_fn = jax.value_and_grad(_cost_fn_with_aux, has_aux=True)

    @jax.jit
    def _step(params: Any, opt_state: Any) -> tuple[Any, Any, Any, dict]:
        (value, aux), grad = value_and_grad_fn(params)

        # Pass value_fn so that line-search optimizers (e.g. LBFGS) can
        # re-evaluate the function during their inner loop.
        updates, new_state = opt.update(
            grad, opt_state, params, value=value, grad=grad, value_fn=_cost_scalar
        )

        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, value, aux

    return _step


def _get_linesearch_steps(opt_state: Any) -> int | None:
    if isinstance(opt_state, tuple) and len(opt_state) >= 3:
        linesearch_state = opt_state[2]
        if hasattr(linesearch_state, "info"):
            return int(linesearch_state.info.num_linesearch_steps)
    return None


def _log_iteration(
    iteration: int,
    cost_value: float,
    grad_norm: float,
    elapsed_time: float,
    ls_steps: int | None = None,
) -> None:
    if ls_steps is None:
        logger.info(
            "iter %d: cost=%.6e  grad_norm=%.6g  time=%.4fs",
            iteration,
            cost_value,
            grad_norm,
            elapsed_time,
        )
    else:
        logger.info(
            "iter %d: cost=%.6e  grad_norm=%.6g  ls=%d  time=%.4fs",
            iteration,
            cost_value,
            grad_norm,
            ls_steps,
            elapsed_time,
        )


def _canonicalize_weak_scalar_leaves(tree: Any) -> Any:
    """Convert weakly-typed scalar JAX leaves to strongly typed scalars.

    This stabilizes JIT cache keys when optimizer states start with weak scalar
    leaves (e.g. from Python literals) that become strongly typed after updates.
    """

    def _canonicalize_leaf(leaf: Any) -> Any:
        if (
            isinstance(leaf, jax.Array)
            and leaf.shape == ()
            and getattr(leaf, "weak_type", False)
        ):
            return jnp.asarray(leaf, dtype=leaf.dtype)
        return leaf

    return jax.tree.map(_canonicalize_leaf, tree)


# ---------------------------------------------------------------------------
# make_solver
# ---------------------------------------------------------------------------


def make_solver(
    cost_fn: Callable,
    opt: optax.GradientTransformationExtraArgs | None = None,
    stop_criteria: StopCriteria | None = None,
    has_aux: bool = True,
) -> Callable[..., SolveResult]:
    """Create a solver from a cost function and optimizer.

    The optimisation step is JIT-compiled **once** at call time; subsequent
    calls to the returned ``solve`` function reuse the compiled artefact.

    Parameters
    ----------
    cost_fn:
        Scalar cost function.  If *has_aux* is ``True`` (default) it must
        return ``(value, aux_dict)`` where ``aux_dict`` is a flat
        ``dict[str, jax.Array]``.  If *has_aux* is ``False`` it must return
        a scalar.
    opt:
        Optax gradient transformation.  Defaults to ``optax.lbfgs()``.
    stop_criteria:
        Convergence thresholds.  Defaults to
        ``StopCriteria(grad_norm_abs=1e-5, max_iter=1000)``.
    has_aux:
        Whether ``cost_fn`` returns auxiliary data (default ``True``).

    Returns
    -------
    Callable[..., SolveResult]
        A ``solve(init_params, *, init_state=None) -> SolveResult`` function.
        Pass ``init_state=previous_result.opt_state`` to warm-start.

    Example::

        solver = make_solver(cost_fn=my_fn, stop_criteria=StopCriteria(max_iter=200))
        result = solver(jnp.zeros(10))
    """
    if opt is None:
        opt = optax.lbfgs(
            linesearch=optax.scale_by_zoom_linesearch(
                max_linesearch_steps=50,
            )
        )
    if stop_criteria is None:
        stop_criteria = StopCriteria()
    stop_criteria.validate()

    _step = _build_step(cost_fn, opt, has_aux)

    def solve(
        init_params: Any,
        *,
        init_state: Any | None = None,
    ) -> SolveResult:
        """Run the optimisation loop.

        Parameters
        ----------
        init_params:
            Initial parameter values (JAX pytree).
        init_state:
            Optional pre-existing optimizer state for warm-starting.  If
            ``None`` a fresh state is initialised from ``opt.init(init_params)``.

        Returns
        -------
        SolveResult
        """
        params = _canonicalize_weak_scalar_leaves(init_params)
        if init_state is None:
            opt_state = opt.init(params)
        else:
            opt_state = init_state
        opt_state = _canonicalize_weak_scalar_leaves(opt_state)

        # ---- history accumulation (Python lists for efficiency) ----
        cost_history: list[float] = []
        gradient_norm_history: list[float] = []
        aux_history: dict[str, list] = {}

        stop_reason: str = "max_iter"
        grad_norm_0: float | None = None
        prev_value: float = float("inf")
        value: float = float("nan")
        start_time: float = time.perf_counter()
        iter_start_time: float = time.perf_counter()
        n_iters: int = 0

        # Keep the last finite state so we can roll back safely if LBFGS
        # proposes a non-finite trial step.
        last_finite_params = params
        last_finite_state = opt_state

        for i in range(stop_criteria.max_iter):
            # Keep scalar leaves strongly typed to avoid signature drift that
            # triggers extra recompiles in early LBFGS iterations.
            params = _canonicalize_weak_scalar_leaves(params)
            opt_state = _canonicalize_weak_scalar_leaves(opt_state)
            params, opt_state, value, aux = _step(params, opt_state)

            # ----- collect grad norm and line search info from optimizer state -----
            grad = otu.tree_get(opt_state, "grad")
            grad_norm = float(otu.tree_norm(grad))

            ls_steps = _get_linesearch_steps(opt_state)

            value_f = float(value)

            if not (np.isfinite(value_f) and np.isfinite(grad_norm)):
                logger.warning(
                    "iter %d produced non-finite values (cost=%s, grad_norm=%s); "
                    "stopping and reverting to last finite state.",
                    i + 1,
                    value_f,
                    grad_norm,
                )
                params = last_finite_params
                opt_state = last_finite_state
                stop_reason = "non_finite"
                n_iters = i
                break

            cost_history.append(value_f)
            gradient_norm_history.append(grad_norm)

            # Collect aux
            if has_aux and isinstance(aux, dict):
                for k, v in aux.items():
                    if k not in aux_history:
                        aux_history[k] = []
                    aux_history[k].append(float(v) if np.ndim(v) == 0 else v)

            # Save the initial grad norm for relative stopping criteria.  If Nan fall back to 1.0
            if grad_norm_0 is None:
                grad_norm_0 = grad_norm if not np.isnan(grad_norm) else 1.0

            iter_time = time.perf_counter() - iter_start_time
            aux_history.setdefault("iter_time", []).append(iter_time)
            aux_history.setdefault("ls_steps", []).append(
                -1 if ls_steps is None else int(ls_steps)
            )
            _log_iteration(
                iteration=i + 1,
                cost_value=value_f,
                grad_norm=grad_norm,
                elapsed_time=iter_time,
                ls_steps=ls_steps,
            )
            iter_start_time = time.perf_counter()

            # ----- check stopping criteria -----
            reason = check_stop_criteria(
                stop_criteria,
                grad_norm=grad_norm,
                grad_norm_0=grad_norm_0,
                value=value_f,
                prev_value=prev_value,
                current_iter=i,
                aux=aux,
            )

            n_iters = i + 1
            prev_value = value_f
            last_finite_params = params
            last_finite_state = opt_state

            if reason is not None:
                stop_reason = reason
                break

        # ---- convert history lists to numpy arrays ----
        history: dict[str, np.ndarray] = {
            "cost": np.array(cost_history, dtype=np.float64),
            "gradient_norms": np.array(gradient_norm_history, dtype=np.float64),
        }
        for k, v_list in aux_history.items():
            try:
                history[k] = np.array(v_list, dtype=np.float64)
            except (ValueError, TypeError):
                history[k] = np.array(v_list, dtype=object)

        total_elapsed_time = time.perf_counter() - start_time
        logger.info(
            "Solver stopped after %d iterations: %s (final cost: %.6g, total time: %.4fs)",
            n_iters,
            stop_reason,
            float(value),
            total_elapsed_time,
        )

        return SolveResult(
            params=params,
            history=history,
            stop_reason=stop_reason,
            opt_state=opt_state,
            n_iters=n_iters,
        )

    return solve
