"""Matrix-free Energy Natural Gradient (Gauss-Newton) Optax optimizer.

The natural-gradient direction solves ``G d = grad`` where
``G = J^T J + λ I`` and ``J`` is the Jacobian of stacked residual vectors.
``G v`` is applied matrix-free via ``vjp(jvp(v))`` per residual term.

``conjugate_gradient`` is adapted from SCIMBA's ``scimba_jax/utils/conjugate_gradient.py``
(https://gitlab.com/scimba/scimba, experimental ``scimba_jax`` module).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import chex
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax
from optax._src import base, transform

# Cap inner PCG iterations when ``max_iter`` is omitted.  A full solve to
# parameter dimension is rarely needed for optimisation directions and is
# very expensive for matrix-free J^T J matvecs on FFD residuals.
DEFAULT_CG_MAX_ITER = 100


def _resolve_cg_max_iter(param_dim: int, max_iter: int | None) -> int:
    """Resolve the PCG iteration budget for a parameter vector of size *param_dim*."""
    if max_iter is not None:
        return max_iter
    return min(param_dim, DEFAULT_CG_MAX_ITER)


@dataclass(frozen=True, slots=True)
class ConjugateGradientInfo:
    """Diagnostics from :func:`conjugate_gradient`."""

    num_iter: int
    residual_norm: chex.Numeric
    converged: bool


def conjugate_gradient(
    g: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    *,
    rtol: float = 1e-10,
    max_iter: int | None = None,
    pginv: Callable[[jax.Array], jax.Array] | None = None,
) -> tuple[jax.Array, ConjugateGradientInfo]:
    """Preconditioned conjugate gradient for symmetric positive-definite ``g``.

    Solves ``g(x) = b`` for ``x`` using matrix-free matvec ``g``.

    Args:
        g: Callable mapping ``v -> G @ v``.
        b: Right-hand side vector.
        rtol: Relative tolerance on ``||r|| / ||b||``.
        max_iter: Maximum CG iterations (defaults to ``b.size``).
        pginv: Optional preconditioner ``r -> M^{-1} r``.

    Returns:
        Approximate solution ``x`` and diagnostic info.
    """
    if max_iter is None:
        max_iter = int(b.size)
    if pginv is None:

        def pginv(r: jax.Array) -> jax.Array:
            return r

    x = jnp.zeros_like(b)
    r = b - g(x)
    z = pginv(r)
    p = z
    rsold = jnp.dot(r, z)
    b_norm = jnp.maximum(jnp.linalg.norm(b), jnp.finfo(b.dtype).eps)

    def body(carry: tuple[jax.Array, jax.Array, jax.Array, chex.Numeric, int]):
        x_i, r_i, p_i, rsold_i, i = carry
        ap = g(p_i)
        alpha = rsold_i / jnp.maximum(jnp.dot(p_i, ap), jnp.finfo(b.dtype).eps)
        x_new = x_i + alpha * p_i
        r_new = r_i - alpha * ap
        z_new = pginv(r_new)
        rsnew = jnp.dot(r_new, z_new)
        beta = rsnew / jnp.maximum(rsold_i, jnp.finfo(b.dtype).eps)
        p_new = z_new + beta * p_i
        return x_new, r_new, p_new, rsnew, i + 1

    def cond(carry: tuple[jax.Array, jax.Array, jax.Array, chex.Numeric, int]):
        _, r_i, _, _, i = carry
        not_converged = jnp.linalg.norm(r_i) > rtol * b_norm
        return jnp.logical_and(not_converged, i < max_iter)

    x, r, _, _, num_iter = jax.lax.while_loop(
        cond,
        body,
        (x, r, p, rsold, 0),
    )
    residual_norm = jnp.linalg.norm(r)
    converged = residual_norm <= rtol * b_norm
    info = ConjugateGradientInfo(
        num_iter=num_iter,
        residual_norm=residual_norm,
        converged=converged,
    )
    return x, info


class _NaturalGradientDirectionState(NamedTuple):
    """Minimal array state so the transform is JIT-compatible."""

    step_count: chex.Numeric


class _LogGridLinesearchState(NamedTuple):
    """State for logarithmic-grid line search."""

    learning_rate: chex.Numeric
    grad: Any | None = None


def _build_g_matvec(
    params: Any,
    residual_fn: Callable[[Any], dict[str, jax.Array]],
    weights: dict[str, float] | None,
    matrix_regularization: float,
    unravel: Callable[[jax.Array], Any],
) -> Callable[[jax.Array], jax.Array]:
    """Build matrix-free Gram matvec ``v -> (J^T J + λ I) v``."""

    def weighted_flat_residual(p: Any) -> jax.Array:
        r_dict = residual_fn(p)
        parts = []
        for name, vec in r_dict.items():
            w = 1.0 if weights is None else weights.get(name, 1.0)
            parts.append(jnp.sqrt(w) * jnp.asarray(vec).ravel())
        if not parts:
            leaves = jax.tree_util.tree_leaves(p)
            dtype = leaves[0].dtype if leaves else jnp.float64
            return jnp.zeros((0,), dtype=dtype)
        return jnp.concatenate(parts)

    _, vjp_fn = jax.vjp(weighted_flat_residual, params)

    def g_matvec(v_flat: jax.Array) -> jax.Array:
        v_tree = unravel(v_flat)
        jvp_val = jax.jvp(weighted_flat_residual, (params,), (v_tree,))[1]
        gt_jvp = vjp_fn(jvp_val)[0]
        gt_flat, _ = ravel_pytree(gt_jvp)
        return gt_flat + matrix_regularization * v_flat

    return g_matvec


def _scale_by_natural_gradient_direction(
    residual_fn: Callable[[Any], dict[str, jax.Array]],
    *,
    weights: dict[str, float] | None,
    matrix_regularization: float,
    rtol: float,
    max_iter: int | None,
) -> base.GradientTransformationExtraArgs:
    """Compute ``G^{-1} grad`` via PCG (returns the raw preconditioned direction)."""

    def init_fn(params: Any) -> _NaturalGradientDirectionState:
        del params
        return _NaturalGradientDirectionState(step_count=jnp.array(0, dtype=jnp.int32))

    def update_fn(
        updates: Any,
        state: _NaturalGradientDirectionState,
        params: Any | None = None,
        **kwargs: Any,
    ) -> tuple[Any, _NaturalGradientDirectionState]:
        del updates
        if params is None:
            raise ValueError("natural_gradient requires `params` in update.")
        grad = kwargs.get("grad")
        if grad is None:
            raise ValueError("natural_gradient requires `grad` in update.")

        _, unravel = ravel_pytree(params)
        g_matvec = _build_g_matvec(
            params,
            residual_fn,
            weights,
            matrix_regularization,
            unravel,
        )
        grad_flat, _ = ravel_pytree(grad)

        cg_max_iter = _resolve_cg_max_iter(grad_flat.size, max_iter)
        ng_flat, _cg_info = conjugate_gradient(
            g_matvec,
            grad_flat,
            rtol=rtol,
            max_iter=cg_max_iter,
        )
        ng_tree = unravel(ng_flat)
        return ng_tree, _NaturalGradientDirectionState(
            step_count=state.step_count + 1,
        )

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def _scale_by_logarithmic_grid_linesearch(
    *,
    nb_max_steps: int = 20,
    beta: float = 2.0,
) -> base.GradientTransformationExtraArgs:
    """Line search over ``eta in {beta^{-k}}`` picking the best objective value."""

    def init_fn(params: Any) -> _LogGridLinesearchState:
        del params
        return _LogGridLinesearchState(learning_rate=jnp.array(1.0))

    def update_fn(
        updates: Any,
        state: _LogGridLinesearchState,
        params: Any | None = None,
        **kwargs: Any,
    ) -> tuple[Any, _LogGridLinesearchState]:
        if params is None:
            raise ValueError("log_grid linesearch requires `params`.")
        value_fn = kwargs.get("value_fn")
        if value_fn is None:
            raise ValueError("log_grid linesearch requires `value_fn`.")

        exponents = jnp.arange(-nb_max_steps, 1)
        etas = beta ** exponents.astype(jnp.float64)

        def eval_eta(eta: jax.Array) -> jax.Array:
            scaled = jax.tree.map(lambda u: eta * u, updates)
            trial = optax.apply_updates(params, scaled)
            return value_fn(trial)

        def scan_fn(
            carry: tuple[jax.Array, jax.Array],
            eta: jax.Array,
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            best_val, best_eta = carry
            val = eval_eta(eta)
            pick_new = val < best_val
            new_val = jnp.where(pick_new, val, best_val)
            new_eta = jnp.where(pick_new, eta, best_eta)
            return (new_val, new_eta), None

        init_val = eval_eta(etas[0])
        (best_val, eta), _ = jax.lax.scan(scan_fn, (init_val, etas[0]), etas[1:])
        del best_val
        grad = kwargs.get("grad")
        scaled_updates = jax.tree.map(lambda u: eta * u, updates)
        return scaled_updates, _LogGridLinesearchState(learning_rate=eta, grad=grad)

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def natural_gradient(
    residual_fn: Callable[[Any], dict[str, jax.Array]],
    *,
    weights: dict[str, float] | None = None,
    matrix_regularization: float = 1e-6,
    linesearch: Literal["zoom", "log_grid"] = "zoom",
    rtol: float = 1e-10,
    max_iter: int | None = None,
    ls_kwargs: dict[str, Any] | None = None,
) -> base.GradientTransformationExtraArgs:
    """Energy Natural Gradient as an Optax transform (drop-in for ``optax.lbfgs``).

    Args:
        residual_fn: ``params -> {term_name: residual_vector}`` with unsquared
            1-D residual vectors ``r_k(theta)``.
        weights: Optional per-term multipliers (defaults to ``1.0`` each).
        matrix_regularization: Tikhonov term ``λ`` added to the Gram matrix.
        linesearch: ``"zoom"`` (Optax Wolfe zoom) or ``"log_grid"`` (SCIMBA-style).
        rtol: Relative tolerance for the inner PCG solve.
        max_iter: Maximum PCG iterations (defaults to ``min(n_params, 100)``).
        ls_kwargs: Keyword arguments forwarded to the line-search constructor.

    Returns:
        An :class:`optax.GradientTransformationExtraArgs` chain computing
        ``params + updates`` with ``updates ≈ -η G^{-1} grad``.
    """
    ls_kwargs = dict(ls_kwargs or {})
    direction = _scale_by_natural_gradient_direction(
        residual_fn,
        weights=weights,
        matrix_regularization=matrix_regularization,
        rtol=rtol,
        max_iter=max_iter,
    )

    if linesearch == "zoom":
        zoom_defaults = {"max_linesearch_steps": 20, "initial_guess_strategy": "one"}
        zoom_defaults.update(ls_kwargs)
        linesearch_tx = optax.scale_by_zoom_linesearch(**zoom_defaults)
    elif linesearch == "log_grid":
        grid_defaults = {"nb_max_steps": 20, "beta": 2.0}
        grid_defaults.update(ls_kwargs)
        linesearch_tx = _scale_by_logarithmic_grid_linesearch(**grid_defaults)
    else:
        raise ValueError(
            f"Unknown linesearch {linesearch!r}; expected 'zoom' or 'log_grid'."
        )

    return optax.chain(
        direction,
        transform.scale(-1.0),
        linesearch_tx,
    )
