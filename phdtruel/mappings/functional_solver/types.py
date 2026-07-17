"""Type definitions for the functional solver.

This module defines the core types used by the functional solver API:
- `SolveResult`: result returned from a solve call
- `StopCriteria`: configurable convergence criteria
- `Checkpoint`: on-disk checkpoint state for resumable solves

Example::

    from phdtruel.mappings.functional_solver import StopCriteria, SolveResult

    criteria = StopCriteria(grad_norm_abs=1e-6, value_change_rel=1e-8)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# SolveResult
# ---------------------------------------------------------------------------


class SolveResult(NamedTuple):
    """Result of a single solve call.

    Attributes:
        params: Final optimised parameters as a JAX/numpy pytree.
        history: Dictionary mapping string keys to 1-D ``np.ndarray`` of
            length ``n_iters`` (number of completed iterations).  Always
            contains the keys ``"cost"`` and ``"gradient_norms"``.  When the
            cost function returns ``(value, aux_dict)`` every key in *aux_dict*
            is also stored.
        stop_reason: Human-readable string describing why optimisation
            stopped (e.g. ``"grad_norm_abs"``, ``"max_iter"``).
        opt_state: Final optimax optimizer state.  Pass as *init_state* to a
            subsequent :func:`make_solver` call to warm-start.
        n_iters: Total number of gradient/function evaluations performed.

    Example::

        result = solver(init_params)
        print(result.stop_reason)   # "grad_norm_abs"
        print(result.history["cost"].shape)  # (n_iters,)
    """

    params: Any
    history: dict[str, np.ndarray]
    stop_reason: str
    opt_state: Any
    n_iters: int


# ---------------------------------------------------------------------------
# StopCriteria
# ---------------------------------------------------------------------------


@dataclass
class StopCriteria:
    """Convergence criteria for the solver.

    At least one criterion must be non-``None``; if several are set the
    optimisation stops as soon as **any** of them is satisfied.

    Attributes:
        grad_norm_abs: Stop when the L2 norm of the gradient falls below
            this absolute threshold (``||g|| < threshold``).
        grad_norm_rel: Stop when ``||g|| / ||g_0|| < threshold`` where
            ``g_0`` is the gradient at the first iteration.
        value_change_abs: Stop when ``|f_k - f_{k-1}| < threshold``.
        value_change_rel: Stop when
            ``|f_k - f_{k-1}| / (|f_{k-1}| + eps) < threshold``.
        jacobian_violations_threshold: Stop when the count of non-positive
            Jacobian determinants falls at or below this threshold
            (only checked when Jacobian barrier is active, otherwise ignored).
        max_iter: Hard upper bound on the number of iterations.

    Example::

        criteria = StopCriteria(
            grad_norm_abs=1e-6,
            value_change_rel=1e-8,
            max_iter=500,
        )
    """

    grad_norm_abs: float | None = None
    grad_norm_rel: float | None = None
    value_change_abs: float | None = None
    value_change_rel: float | None = None
    jacobian_violations_threshold: int | None = None
    max_iter: int = 100

    def check(
        self,
        *,
        grad_norm: float,
        grad_norm_0: float,
        value: float,
        prev_value: float,
        current_iter: int,
        aux: dict[str, Any] | None = None,
    ) -> str | None:
        """Return the first satisfied stopping criterion, if any."""
        eps = 1e-30

        if self.grad_norm_abs is not None and grad_norm < self.grad_norm_abs:
            return "grad_norm_abs"

        if self.grad_norm_rel is not None:
            relative = grad_norm / (grad_norm_0 + eps)
            if relative < self.grad_norm_rel:
                return "grad_norm_rel"

        if self.value_change_abs is not None:
            if abs(value - prev_value) < self.value_change_abs:
                return "value_change_abs"

        if self.value_change_rel is not None:
            relative = abs(value - prev_value) / (abs(prev_value) + eps)
            if relative < self.value_change_rel:
                return "value_change_rel"

        jacobian_violations = (
            aux.get("jacobian_violations") if isinstance(aux, dict) else None
        )
        if (
            self.jacobian_violations_threshold is not None
            and jacobian_violations is not None
        ):
            violations_count = (
                int(jacobian_violations)
                if hasattr(jacobian_violations, "__array__")
                else jacobian_violations
            )
            if (
                violations_count >= 0
                and violations_count <= self.jacobian_violations_threshold
            ):
                return "jacobian_violations"

        if self.max_iter is not None and current_iter >= self.max_iter:
            return "max_iter"

        return None

    def validate(self) -> None:
        """Raise :class:`ValueError` if the configuration is invalid.

        Checks:
        - At least one stopping condition is set: either a threshold
          (grad_norm_abs, grad_norm_rel, value_change_abs, value_change_rel)
          or max_iter < infinity.  In practice *max_iter* always terminates
          the loop, so it alone is a valid stopping condition.
        - All threshold values are positive.
        - *max_iter* is a positive integer.
        """
        if self.max_iter <= 0:
            raise ValueError(
                f"StopCriteria.max_iter must be a positive integer, got {self.max_iter!r}."
            )
        for name, val in [
            ("grad_norm_abs", self.grad_norm_abs),
            ("grad_norm_rel", self.grad_norm_rel),
            ("value_change_abs", self.value_change_abs),
            ("value_change_rel", self.value_change_rel),
        ]:
            if val is not None and val <= 0:
                raise ValueError(f"StopCriteria.{name} must be positive, got {val!r}.")


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


class Checkpoint(NamedTuple):
    """On-disk checkpoint state for resumable solves.

    Instances are serialised / deserialised with :mod:`pickle`.  They carry
    enough state to resume :func:`solve_with_checkpointing` from any saved
    checkpoint.

    Attributes:
        params: Optimiser parameters at checkpoint time.
        opt_state: Optax optimizer state at checkpoint time.
        history: Partial history collected up to the checkpoint
            (same schema as :attr:`SolveResult.history`).
        barrier_iter: Barrier-method outer iteration index at checkpoint time
            (``0`` for non-barrier solves).
        iter_num: Global iteration count at checkpoint time.

    Example::

        import pickle, pathlib
        ckpt_path = pathlib.Path("checkpoint.pkl")
        # load manually if needed
        with ckpt_path.open("rb") as f:
            ckpt = pickle.load(f)
        print(ckpt.iter_num)
    """

    params: Any
    opt_state: Any
    history: dict[str, list]
    barrier_iter: int
    iter_num: int
