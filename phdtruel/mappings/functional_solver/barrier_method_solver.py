"""Barrier-method outer loop for the functional solver.

This module provides :func:`solve_barrier`, which repeatedly calls a solver
with a staged update schedule that modifies parameter fields between outer
iterations.

Per-stage solver configuration uses meta keys in each update dict (keys
starting with ``_``). Supported meta keys:

- ``_max_iter``: override iteration budget for this stage
- ``_grad_norm_rel``: override relative gradient tolerance
- ``_jacobian_violations_threshold``: override Jacobian violation stop threshold
- ``_solver``: ``"lbfgs"`` or ``"natural_gradient"`` (requires *opt_resolver*)

Example::

    from phdtruel.mappings.functional_solver import (
        solve_barrier,
        StopCriteria,
        make_solver_opt_resolver,
    )
    import jax.numpy as jnp

    updates = [
        {
            "epsilon": jnp.array(1.0),
            "_max_iter": 100,
            "_solver": "lbfgs",
        },
        {
            "epsilon": jnp.array(0.1),
            "_max_iter": 500,
            "_solver": "natural_gradient",
        },
    ]

    init_params = {"x": jnp.array([1.0, 1.0]), "epsilon": jnp.array(1.0)}
    result = solve_barrier(
        cost_fn=cost,
        init_params=init_params,
        updates=updates,
        opt_resolver=make_solver_opt_resolver(residual_fn=residual_fn),
        has_aux=False,
    )
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import Any, Callable

import jax
import numpy as np
import optax

from .solver import make_solver
from .types import SolveResult, StopCriteria

logger = logging.getLogger(__name__)


def _split_update_step(
    update_step: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Partition an update dict into param paths and stage meta keys."""
    param_updates: dict[str, Any] = {}
    meta: dict[str, Any] = {}
    for key, value in update_step.items():
        if key.startswith("_"):
            meta[key] = value
        else:
            param_updates[key] = value
    return param_updates, meta


def _merge_stop_criteria(
    base: StopCriteria | None,
    meta: dict[str, Any],
) -> StopCriteria:
    """Return stop criteria with per-stage meta overrides applied."""
    merged = base if base is not None else StopCriteria()
    overrides: dict[str, Any] = {}
    for meta_key, field_name in (
        ("_max_iter", "max_iter"),
        ("_grad_norm_rel", "grad_norm_rel"),
        ("_grad_norm_abs", "grad_norm_abs"),
        ("_value_change_abs", "value_change_abs"),
        ("_value_change_rel", "value_change_rel"),
        ("_jacobian_violations_threshold", "jacobian_violations_threshold"),
    ):
        if meta_key in meta:
            overrides[field_name] = meta[meta_key]
    if overrides:
        merged = replace(merged, **overrides)
    merged.validate()
    return merged


def _stop_criteria_cache_key(criteria: StopCriteria) -> tuple[Any, ...]:
    return (
        criteria.grad_norm_abs,
        criteria.grad_norm_rel,
        criteria.value_change_abs,
        criteria.value_change_rel,
        criteria.jacobian_violations_threshold,
        criteria.max_iter,
    )


def _resolve_stage_opt(
    *,
    meta: dict[str, Any],
    default_opt: optax.GradientTransformationExtraArgs | None,
    opt_resolver: Callable[[str], optax.GradientTransformationExtraArgs | None]
    | None,
) -> optax.GradientTransformationExtraArgs | None:
    solver_name = meta.get("_solver")
    if solver_name is None:
        return default_opt
    if opt_resolver is None:
        raise ValueError(
            "Update step specifies _solver but no opt_resolver was provided to "
            "solve_barrier."
        )
    if not isinstance(solver_name, str):
        raise ValueError(
            f"_solver must be a string solver name, got {type(solver_name).__name__}."
        )
    return opt_resolver(solver_name)


def _parse_update_path(path: str) -> list[str | int]:
    """Parse a dotted/indexed update path into tokens.

    Supported grammar is intentionally narrow:
    - dotted names: "a.b.c"
    - integer indices: "a[0].b" or "[0].a"
    """
    if not path:
        raise ValueError("Update path must be a non-empty string.")

    tokens: list[str | int] = []
    segments = path.split(".")
    for segment in segments:
        if segment == "":
            raise ValueError(f"Invalid update path {path!r}: empty path segment.")

        cursor = 0
        name_match = re.match(r"[A-Za-z_]\w*", segment)
        if name_match is not None:
            name = name_match.group(0)
            tokens.append(name)
            cursor = len(name)

        while cursor < len(segment):
            if segment[cursor] != "[":
                raise ValueError(
                    f"Invalid update path {path!r}: malformed index in segment "
                    f"{segment!r}."
                )
            end = segment.find("]", cursor + 1)
            if end == -1:
                raise ValueError(
                    f"Invalid update path {path!r}: missing closing ']' in segment "
                    f"{segment!r}."
                )
            index_str = segment[cursor + 1 : end]
            if not index_str.isdigit():
                raise ValueError(
                    f"Invalid update path {path!r}: index must be a non-negative "
                    f"integer, got {index_str!r}."
                )
            tokens.append(int(index_str))
            cursor = end + 1

    if not tokens:
        raise ValueError(f"Invalid update path {path!r}: no tokens found.")
    return tokens


def _set_path_value(tree: Any, tokens: list[str | int], value: Any, path: str) -> Any:
    """Return a copy of *tree* with *value* written at *tokens* path."""
    if not tokens:
        return value

    head, *tail = tokens
    if isinstance(head, str):
        if hasattr(tree, "_replace"):
            if not hasattr(tree, head):
                raise ValueError(
                    f"Cannot apply update for path {path!r}: field {head!r} does not "
                    "exist."
                )
            child = getattr(tree, head)
            new_child = _set_path_value(child, tail, value, path)
            return tree._replace(**{head: new_child})

        if isinstance(tree, dict):
            if head not in tree:
                raise ValueError(
                    f"Cannot apply update for path {path!r}: key {head!r} does not "
                    "exist."
                )
            new_tree = dict(tree)
            new_tree[head] = _set_path_value(tree[head], tail, value, path)
            return new_tree

        raise TypeError(
            f"Cannot apply update for path {path!r}: expected a NamedTuple-like "
            f"object or dict when traversing field {head!r}, got "
            f"{type(tree).__name__}."
        )

    if isinstance(tree, tuple):
        if head < 0 or head >= len(tree):
            raise ValueError(
                f"Cannot apply update for path {path!r}: tuple index {head} is out "
                f"of range for length {len(tree)}."
            )
        new_items = list(tree)
        new_items[head] = _set_path_value(tree[head], tail, value, path)
        return tuple(new_items)

    if isinstance(tree, list):
        if head < 0 or head >= len(tree):
            raise ValueError(
                f"Cannot apply update for path {path!r}: list index {head} is out "
                f"of range for length {len(tree)}."
            )
        new_items = list(tree)
        new_items[head] = _set_path_value(tree[head], tail, value, path)
        return new_items

    raise TypeError(
        f"Cannot apply update for path {path!r}: expected a list/tuple when "
        f"traversing index [{head}], got {type(tree).__name__}."
    )


def _apply_update_step(params: Any, update_step: dict[str, Any]) -> Any:
    """Apply one schedule step (dict of path -> value) to params."""
    updated_params = params
    for path, value in update_step.items():
        tokens = _parse_update_path(path)
        updated_params = _set_path_value(updated_params, tokens, value, path)
    return updated_params


def _get_or_create_solver(
    *,
    solver_cache: dict[tuple[Any, ...], Callable[..., SolveResult]],
    cost_fn: Callable,
    stage_opt: optax.GradientTransformationExtraArgs | None,
    stage_stop_criteria: StopCriteria,
    has_aux: bool,
) -> Callable[..., SolveResult]:
    cache_key = (id(stage_opt), _stop_criteria_cache_key(stage_stop_criteria))
    if cache_key not in solver_cache:
        solver_cache[cache_key] = make_solver(
            cost_fn=cost_fn,
            opt=stage_opt,
            stop_criteria=stage_stop_criteria,
            has_aux=has_aux,
        )
    return solver_cache[cache_key]


def solve_barrier(
    cost_fn: Callable,
    init_params: Any,
    updates: list[dict[str, Any]],
    *,
    opt: optax.GradientTransformationExtraArgs | None = None,
    stop_criteria: StopCriteria | None = None,
    opt_resolver: Callable[[str], optax.GradientTransformationExtraArgs | None]
    | None = None,
    has_aux: bool = True,
) -> SolveResult:
    """Run a staged outer loop with per-stage parameter updates.

    Each outer iteration:

    1. Applies one update dict from *updates* to the current params.
    2. Runs the solver from the updated params using ``cost_fn``.

    The cost function is compiled once per unique ``(opt, stop_criteria)`` pair
    and reused across stages that share the same configuration.
    The number of outer iterations is ``len(updates)``.

    Parameters
    ----------
    cost_fn:
        Cost function suitable for :func:`make_solver`.  Compiled once per
        unique stage configuration.
    init_params:
        Initial parameters for the first outer iteration.
    updates:
        Outer-loop schedule. Each element is a dict mapping path -> value,
        where paths support dotted fields and integer indices (for example,
        ``"barrier_config.barrier_epsilon"`` or ``"displacements_w[0]"``).
        Keys starting with ``_`` are stage meta configuration (not applied to
        params): ``_max_iter``, ``_grad_norm_rel``, ``_grad_norm_abs``,
        ``_value_change_abs``, ``_value_change_rel``,
        ``_jacobian_violations_threshold``, ``_solver``.
    opt:
        Default Optax gradient transformation when a stage omits ``_solver``
        (default ``optax.lbfgs()``).
    stop_criteria:
        Default convergence thresholds; per-stage meta keys override fields
        for that stage only.
    opt_resolver:
        Callable mapping ``_solver`` names (e.g. ``"lbfgs"``,
        ``"natural_gradient"``) to Optax optimizers. Required when any update
        specifies ``_solver``.
    has_aux:
        Whether the cost function returns ``(value, aux_dict)`` (default
        ``True``).

    Returns
    -------
    SolveResult
        Result with accumulated history from all completed outer iterations.
        The *history* contains concatenated data from all barrier stages,
        with *n_iters* being the total number of iterations across all stages.

    Raises
    ------
    ValueError
        If *updates* is empty, or if ``_solver`` is set without *opt_resolver*.

    Example::

        result = solve_barrier(
            cost_fn=my_cost_fn,
            init_params=init_params,
            updates=[
                {
                    "barrier_config.barrier_epsilon": jnp.array(1.0),
                    "_max_iter": 200,
                },
                {
                    "barrier_config.barrier_epsilon": jnp.array(0.1),
                    "_max_iter": 1000,
                    "_solver": "lbfgs",
                },
            ],
            opt_resolver=make_solver_opt_resolver(residual_fn=residual_fn),
            has_aux=False,
        )
        print(f"Stopped: {result.stop_reason}")
    """
    if len(updates) < 1:
        raise ValueError("updates must contain at least one outer-iteration step.")

    default_stop_criteria = (
        stop_criteria if stop_criteria is not None else StopCriteria()
    )
    default_stop_criteria.validate()

    solver_cache: dict[tuple[Any, ...], Callable[..., SolveResult]] = {}

    params = init_params
    opt_state: Any = None
    result: SolveResult | None = None

    accumulated_history: dict[str, list] = {}
    total_iters = 0
    final_stop_reason = ""

    prev_stage_config: tuple[Any, ...] | None = None

    n_barrier_iters = len(updates)
    for barrier_iter, update_step in enumerate(updates):
        param_updates, meta = _split_update_step(update_step)
        stage_stop_criteria = _merge_stop_criteria(default_stop_criteria, meta)
        stage_opt = _resolve_stage_opt(
            meta=meta,
            default_opt=opt,
            opt_resolver=opt_resolver,
        )
        stage_config = (id(stage_opt), _stop_criteria_cache_key(stage_stop_criteria))
        solver_name = meta.get("_solver", "default")

        logger.info(
            "solve_barrier: outer iteration %d / %d (solver=%s, max_iter=%d)",
            barrier_iter + 1,
            n_barrier_iters,
            solver_name,
            stage_stop_criteria.max_iter,
        )

        before_tree = jax.tree_util.tree_structure(params)
        params = _apply_update_step(params, param_updates)
        after_tree = jax.tree_util.tree_structure(params)

        if barrier_iter > 0:
            logger.info(
                "solve_barrier: starting new barrier stage %d; "
                "resetting optimizer state to prevent alpha weight corruption",
                barrier_iter + 1,
            )
            opt_state = None
        elif before_tree != after_tree:
            logger.info(
                "solve_barrier: params tree changed at outer iteration %d; "
                "resetting warm-start optimizer state",
                barrier_iter + 1,
            )
            opt_state = None
        elif prev_stage_config is not None and stage_config != prev_stage_config:
            logger.info(
                "solve_barrier: stage %d solver/stop-criteria changed; "
                "resetting optimizer state",
                barrier_iter + 1,
            )
            opt_state = None

        solver = _get_or_create_solver(
            solver_cache=solver_cache,
            cost_fn=cost_fn,
            stage_opt=stage_opt,
            stage_stop_criteria=stage_stop_criteria,
            has_aux=has_aux,
        )

        result = solver(params, init_state=opt_state)

        logger.info(
            "solve_barrier: iteration %d done — %s (%d steps)",
            barrier_iter,
            result.stop_reason,
            result.n_iters,
        )

        for key, values in result.history.items():
            if key not in accumulated_history:
                accumulated_history[key] = []
            accumulated_history[key].extend(values)

        total_iters += result.n_iters
        final_stop_reason = result.stop_reason
        params = result.params
        opt_state = result.opt_state
        prev_stage_config = stage_config

    assert result is not None

    full_history = {
        key: np.array(values) for key, values in accumulated_history.items()
    }

    return SolveResult(
        params=result.params,
        history=full_history,
        stop_reason=final_stop_reason,
        opt_state=result.opt_state,
        n_iters=total_iters,
    )
