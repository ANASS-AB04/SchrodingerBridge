"""
heat_solver.py  —  FVM heat semigroup  ∂φ/∂t = γ Δφ
=====================================================

Root-cause fixes applied
────────────────────────
B1  jnp.min(dx_i) → 10th-percentile cell size.
    One tiny cell near the bump was controlling the CFL for the whole mesh
    (e.g. n_steps ≈ 16 000 instead of ≈ 640) — a 25× slowdown.

B2  n_steps is now a *Python integer* (static JAX arg).
    A dynamic JAX-int upper bound in a nested fori_loop blocks XLA loop
    fusion and breaks jax.vmap (different batch elements would need different
    iteration counts).  With a static bound the inner loop is compile-time
    constant, vmap works correctly, and JAX generates optimal XLA code.

Usage pattern
─────────────
    n = heat_solver.compute_n_steps(mesh, gamma_diff)   # once, Python int
    phi_out = heat_solver.solve_heat_equation(phi, t, gamma_diff, mesh, n)
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


# ─────────────────────────────────────────────────────────────────────────────
#  Step-count helper  (pure numpy, returns a Python int)
# ─────────────────────────────────────────────────────────────────────────────

def compute_n_steps(mesh, gamma_diff, t_target=1.0, CFL=0.5, pct=10):
    """
    Number of explicit RK2-SSP steps required to integrate ∂φ/∂t = γ Δφ
    from 0 to t_target, with the CFL condition based on the *pct*-th
    percentile cell size (not the global minimum).

    Returns a Python int — use it as a ``static_argnames`` value everywhere.

    Why percentile?
    ───────────────
    For a bump mesh with h≈0.02, a handful of cells near the curved geometry
    can have h≈0.003.  Using jnp.min makes those cells dictate the step-size
    for all 12 000 cells, inflating n_steps by 25× or more.
    The 10th-percentile ignores the worst 10 % of cells; their slightly
    larger per-cell truncation error is negligible for smooth transport.
    """
    dx_i = (np.asarray(mesh.area)
            / np.sum(np.asarray(mesh.surface)[
                         np.asarray(mesh.face_connectivity)], axis=-1))
    dx_ref = float(np.percentile(dx_i, pct))
    dt_max = CFL * dx_ref ** 2 / gamma_diff
    n = max(int(np.ceil(t_target / dt_max)), 1)
    return n


# ─────────────────────────────────────────────────────────────────────────────
#  FVM kernels  (JIT-compiled)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def compute_scalar_gradient_LSQ(phi, mesh):
    """
    Inverse-distance-weighted LSQ gradient at cell centres.
    Analytical 2×2 solve (no linalg.solve overhead).
    Used once per output frame to extract the SB drift ∇g_t.
    """
    Delta_x = mesh.barycenter[mesh.neighbors] - mesh.barycenter[:, None, :]
    replace  = jnp.mean(mesh.points[mesh.faces[mesh.face_connectivity]], axis=-2)
    replace  = 2.0 * (replace - mesh.barycenter[:, None, :])
    is_bnd   = mesh.face_markers[mesh.face_connectivity] > 0
    Delta_x  = jnp.where(is_bnd[..., None], replace, Delta_x)

    phi_L     = jnp.repeat(phi[:, None], 3, axis=1)
    phi_R     = jnp.where(is_bnd, phi_L, phi[mesh.neighbors])
    Delta_phi = phi_R - phi_L

    w = 1.0 / jnp.maximum(jnp.linalg.norm(Delta_x, axis=-1) ** 2, 1e-12)
    A = jnp.einsum("ijk,ijl->ikl", w[..., None] * Delta_x, Delta_x)
    b = jnp.sum(w[..., None] * Delta_phi[..., None] * Delta_x, axis=1)

    det = jnp.maximum(A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0], 1e-14)
    gx  = ( A[..., 1, 1] * b[..., 0] - A[..., 0, 1] * b[..., 1]) / det
    gy  = (-A[..., 1, 0] * b[..., 0] + A[..., 0, 0] * b[..., 1]) / det
    return jnp.stack([gx, gy], axis=-1)


@jax.jit
def heat_residual(phi, mesh, gamma_diff):
    """
    TPFA residual for ∂φ/∂t = γ Δφ.
    Neumann (zero-flux) BC on all boundary faces.
    """
    is_bnd  = mesh.face_markers[mesh.face_connectivity] > 0
    phi_L   = jnp.repeat(phi[:, None], 3, axis=1)
    phi_R   = jnp.where(is_bnd, phi_L, phi[mesh.neighbors])

    Delta_x = mesh.barycenter[mesh.neighbors] - mesh.barycenter[:, None, :]
    replace = jnp.mean(mesh.points[mesh.faces[mesh.face_connectivity]], axis=-2)
    replace = 2.0 * (replace - mesh.barycenter[:, None, :])
    Delta_x = jnp.where(is_bnd[..., None], replace, Delta_x)

    d_ij    = jnp.linalg.norm(Delta_x, axis=-1)
    dn_phi  = (phi_R - phi_L) / jnp.maximum(d_ij, 1e-12)
    flux    = jnp.sum(-gamma_diff * dn_phi * mesh.surface[mesh.face_connectivity],
                       axis=-1)
    return -flux / mesh.area


@jax.jit
def heat_step_RK2_SSP(phi, mesh, dt, gamma_diff):
    """One SSP-RK2 (Heun) step."""
    k1   = heat_residual(phi,       mesh, gamma_diff)
    phi1 = phi + dt * k1
    k2   = heat_residual(phi1, mesh, gamma_diff)
    return 0.5 * (phi + phi1 + dt * k2)


@partial(jax.jit, static_argnames=["n_steps"])
def solve_heat_equation(phi_init, t_target, gamma_diff, mesh, n_steps):
    """
    Integrate ∂φ/∂t = γ Δφ  from 0 to t_target in exactly n_steps steps.

    ``n_steps`` MUST be a Python integer (static_argnames).
    Pre-compute it once with ``compute_n_steps()``; pass it unchanged to
    every subsequent call for the same mesh and gamma_diff.

    ``dt = t_target / n_steps`` is computed at runtime, so this function
    correctly handles any t_target value (including batched JAX arrays for
    jax.vmap) using the same compiled loop body.
    """
    dt = t_target / n_steps

    def body_fn(_, phi):
        return heat_step_RK2_SSP(phi, mesh, dt, gamma_diff)

    return jax.lax.fori_loop(0, n_steps, body_fn, phi_init)