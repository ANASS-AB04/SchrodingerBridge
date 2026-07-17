"""Helpers for resolving solver names to Optax optimizers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import optax

from .natural_gradient import natural_gradient

SupportedSolverName = Literal["lbfgs", "natural_gradient"]


def make_solver_opt_resolver(
    *,
    residual_fn: Callable[[Any], dict],
    natural_gradient_linesearch: Literal["zoom", "log_grid"] = "zoom",
    matrix_regularization: float = 1e-6,
    rtol: float = 1e-10,
    max_iter: int | None = None,
) -> Callable[[str], optax.GradientTransformationExtraArgs | None]:
    """Build a resolver mapping solver names to Optax optimizers.

    Maps ``"lbfgs"`` to ``None`` (the default in :func:`make_solver`) and
    ``"natural_gradient"`` to :func:`natural_gradient`.

    Args:
        residual_fn: Residual function required by natural gradient.
        natural_gradient_linesearch: Line search for natural gradient stages.
        matrix_regularization: Tikhonov regularization for natural gradient.
        rtol: Relative tolerance for the inner PCG solve.
        max_iter: Maximum PCG iterations (defaults to ``min(n_params, 100)``).

    Returns:
        Callable accepting a solver name and returning an Optax optimizer or
        ``None`` for LBFGS.
    """

    def resolve(solver_name: str) -> optax.GradientTransformationExtraArgs | None:
        if solver_name == "lbfgs":
            return None
        if solver_name == "natural_gradient":
            return natural_gradient(
                residual_fn=residual_fn,
                matrix_regularization=matrix_regularization,
                linesearch=natural_gradient_linesearch,
                rtol=rtol,
                max_iter=max_iter,
            )
        raise ValueError(
            f"Unknown solver {solver_name!r}; expected 'lbfgs' or 'natural_gradient'."
        )

    return resolve
