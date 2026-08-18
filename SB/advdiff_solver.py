"""
advdiff_solver.py  —  FVM advection–diffusion semigroups  Q  and  Q†
====================================================================

These are the drifted generalisation of ``heat_solver.py``.  With a reference
drift β(x) (a per-cell velocity field, shape ``(N_cells, 2)``) the Schrödinger
bridge uses the transition operator of the reference SDE

        dX_τ = β(X_τ) dτ + √(2γ) dW_τ ,

whose generator is  L = β·∇ + γΔ.  Two operators are needed (they coincide only
when β≡0, because the heat kernel is self-adjoint):

    Q_τ  = e^{τL}    non-conservative backward semigroup   ∂_τu = β·∇u + γΔu
    Q†_τ = e^{τL*}   conservative   forward  semigroup     ∂_τu = −div(βu) + γΔu

(see ``referrence_drift.tex`` §"Log-potentials and drifted IPFP", eqs.
``backdrift`` / ``forwarddrift``).  Because

        div(βu) = u·(div β) + β·∇u ,

both operators share ONE conservative advection routine ``_advection_div`` that
discretises div(βu) with a limited (MUSCL) upwind flux, plus the TPFA diffusion
term reused verbatim from ``heat_solver.heat_residual``:

    Q  residual = γΔu + β·∇u      = heat_residual + ( div(βu) − u·div β )
    Q† residual = γΔu − div(βu)   = heat_residual − div(βu)

Design choices
──────────────
* β is a *traced* array argument (not static) so a per-frame β reuses the same
  compiled kernel; ``n_steps`` stays a static Python int, as in ``heat_solver``.
* Advective boundary flux is set to zero on every boundary face (β·n·φ = 0),
  matching the Neumann diffusion BC.  This makes Q† discretely mass-conserving
  (∫ div(βu) dA = 0) and preserves constants under Q (Q[1] = div(β·1) − div β = 0).
* The face reconstruction uses a minmod-limited LSQ slope → TVD / positivity
  preserving, which the exponential/log SB machinery requires.
* When β≡0 every added term vanishes and Q, Q† return exactly the heat result.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from . import heat_solver


# ─────────────────────────────────────────────────────────────────────────────
#  Step-count helper  (advection + diffusion CFL, returns a Python int)
# ─────────────────────────────────────────────────────────────────────────────

def compute_n_steps_advdiff(mesh, gamma_diff, beta_max, t_target=1.0,
                            CFL_diff=0.5, CFL_adv=0.5, pct=10, warn=True):
    """
    Number of explicit RK2-SSP steps for ∂_τu = β·∇u + γΔu, from the tighter of
    the diffusive (dt ≤ CFL·dx²/γ) and advective (dt ≤ CFL·dx/|β|) limits.

    ``beta_max`` is the max cell-wise ‖β‖ over the field being propagated.
    Falls back to the pure-diffusion count of ``heat_solver.compute_n_steps``
    when ``beta_max`` is ~0.
    """
    dx_ref = heat_solver.reference_cell_size(mesh, pct)
    dt_diff = CFL_diff * dx_ref ** 2 / gamma_diff
    beta_max = float(beta_max)
    if beta_max <= 1e-14:
        dt_max = dt_diff
    else:
        dt_adv = CFL_adv * dx_ref / beta_max
        dt_max = min(dt_diff, dt_adv)
    n = max(int(np.ceil(t_target / dt_max)), 1)
    if warn:
        heat_solver.check_resolution(dx_ref, gamma_diff, n, label="advdiff")
    return n


# ─────────────────────────────────────────────────────────────────────────────
#  First-order upwind advection  D†  and its exact area-weighted transpose  D*
#
#  A Schrödinger reference kernel must be a LINEAR Markov operator, so the
#  advection uses first-order upwind (NOT a nonlinear limiter): a limited MUSCL
#  reconstruction would break the conditional-mean identity behind T/S and the
#  coupling interpretation of the IPFP fixed point.  Q and Q† are then built as
#  an exact discrete transpose pair, which buys three things at once:
#    (i)   positivity under the advective CFL at ANY γ (both operators are
#          M-matrices — off-diagonals ≥ 0), curing the small-γ Péclet blow-up
#          that a conservative-upwind Q suffered;
#    (ii)  correct upwinding for β·∇ (information pulled from the +β side);
#    (iii) ⟨Qu,v⟩_A = ⟨u,Q†v⟩_A to machine precision (RK2 on a frozen linear
#          operator transposes step-by-step), so (Q,Q†) is ONE Markov kernel and
#          the IPFP fixed point is a genuine coupling.
#
#  Honest cost of dropping MUSCL: first-order numerical diffusion ~|β·n|·d/2
#  along β inside the band (a modest anisotropic γ inflation the bridge absorbs).
#  For sharper transport while staying linear+monotone+transposable at all
#  Péclet, the principled upgrade is a Scharfetter–Gummel exponential-fit flux.
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def _beta_face_normal(mesh, beta_cells):
    """β·n at each face (averaged from the two adjacent cells).  (N_cells, 3)."""
    is_bnd    = mesh.face_markers[mesh.face_connectivity] > 0
    beta_i    = jnp.repeat(beta_cells[:, None, :], 3, axis=1)          # (N,3,2)
    beta_j    = jnp.where(is_bnd[..., None], beta_i, beta_cells[mesh.neighbors])
    beta_face = 0.5 * (beta_i + beta_j)                               # (N,3,2)
    return jnp.sum(beta_face * mesh.normals, axis=-1)                 # (N,3)


@jax.jit
def compute_div_beta(mesh, beta_cells):
    """Discrete div β per cell (Green–Gauss, zero flux on boundary faces).
    Diagnostic only — no longer used by the residuals (the transpose form of Q
    needs no reaction term)."""
    bn      = _beta_face_normal(mesh, beta_cells)                     # (N,3)
    is_bnd  = mesh.face_markers[mesh.face_connectivity] > 0
    surf    = mesh.surface[mesh.face_connectivity]
    flux    = jnp.where(is_bnd, 0.0, bn * surf)
    return jnp.sum(flux, axis=-1) / mesh.area


@jax.jit
def _advection_div_conservative(phi, mesh, beta_cells):
    """
    D†φ = conservative first-order-upwind discretisation of div(βφ):
        (D†φ)_i = (1/a_i) Σ_f s_f [ max(bn,0)·φ_i + min(bn,0)·φ_j ].
    Boundary faces carry zero advective flux (β·n·φ = 0).
    """
    bn       = _beta_face_normal(mesh, beta_cells)                   # (N,3)
    is_bnd   = mesh.face_markers[mesh.face_connectivity] > 0
    phi_i    = jnp.repeat(phi[:, None], 3, axis=1)                   # (N,3)
    phi_j    = jnp.where(is_bnd, phi_i, phi[mesh.neighbors])
    phi_up   = jnp.where(bn > 0.0, phi_i, phi_j)                     # upwind
    surf     = mesh.surface[mesh.face_connectivity]
    flux     = jnp.where(is_bnd, 0.0, bn * phi_up * surf)
    return jnp.sum(flux, axis=-1) / mesh.area


@jax.jit
def _advection_transpose(phi, mesh, beta_cells):
    """
    D*φ = exact area-weighted transpose of D† (⟨D†φ,ψ⟩_A = ⟨φ,D*ψ⟩_A):
        (D*φ)_i = (1/a_i) Σ_f s_f · max(bn,0) · (φ_i − φ_j).
    Since the L²-adjoint of div(β·) is −β·∇, we have β·∇φ = −D*φ.  Boundary
    faces carry zero flux.  Off-diagonal coefficient of φ_j is −max(bn,0)s/a_i,
    so −D* has non-negative off-diagonals → Q is an M-matrix (positivity).
    """
    bn       = _beta_face_normal(mesh, beta_cells)                   # (N,3)
    is_bnd   = mesh.face_markers[mesh.face_connectivity] > 0
    phi_i    = jnp.repeat(phi[:, None], 3, axis=1)                   # (N,3)
    phi_j    = jnp.where(is_bnd, phi_i, phi[mesh.neighbors])
    w        = jnp.maximum(bn, 0.0)                                  # max(bn,0)
    surf     = mesh.surface[mesh.face_connectivity]
    flux     = jnp.where(is_bnd, 0.0, w * (phi_i - phi_j) * surf)
    return jnp.sum(flux, axis=-1) / mesh.area


# ─────────────────────────────────────────────────────────────────────────────
#  Residuals:  Q (non-conservative)  and  Q† (conservative adjoint)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def advdiff_residual(phi, mesh, beta_cells, gamma_diff):
    """Q residual:  ∂_τφ = β·∇φ + γΔφ  =  heat_residual − D*φ  (β·∇ = −D*)."""
    diff = heat_solver.heat_residual(phi, mesh, gamma_diff)          # γΔφ
    return diff - _advection_transpose(phi, mesh, beta_cells)


@jax.jit
def advdiff_residual_adjoint(phi, mesh, beta_cells, gamma_diff):
    """Q† residual:  ∂_τφ = −div(βφ) + γΔφ  =  heat_residual − D†φ."""
    diff = heat_solver.heat_residual(phi, mesh, gamma_diff)          # γΔφ
    return diff - _advection_div_conservative(phi, mesh, beta_cells)


# ─────────────────────────────────────────────────────────────────────────────
#  SSP-RK2 (Heun) steps
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def _step_Q(phi, mesh, dt, beta_cells, gamma_diff):
    k1   = advdiff_residual(phi,  mesh, beta_cells, gamma_diff)
    phi1 = phi + dt * k1
    k2   = advdiff_residual(phi1, mesh, beta_cells, gamma_diff)
    return 0.5 * (phi + phi1 + dt * k2)


@jax.jit
def _step_Qadj(phi, mesh, dt, beta_cells, gamma_diff):
    k1   = advdiff_residual_adjoint(phi,  mesh, beta_cells, gamma_diff)
    phi1 = phi + dt * k1
    k2   = advdiff_residual_adjoint(phi1, mesh, beta_cells, gamma_diff)
    return 0.5 * (phi + phi1 + dt * k2)


# ─────────────────────────────────────────────────────────────────────────────
#  Semigroup solvers  (mirror heat_solver.solve_heat_equation)
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=["n_steps"])
def solve_advdiff_equation(phi_init, t_target, gamma_diff, mesh, beta_cells, n_steps):
    """
    Integrate the non-conservative Q PDE  ∂_τφ = β·∇φ + γΔφ  over [0, t_target].
    Returns Q_{t_target}[φ_init].  ``n_steps`` is a static Python int.
    """
    dt = t_target / n_steps
    # Promote the carry to the common dtype of the state and the drift so the
    # fori_loop body input/output dtypes match (β may be float64 while a field
    # built from float32 mesh coordinates — e.g. the identity-bias solve — is not).
    dtype    = jnp.result_type(phi_init, beta_cells, gamma_diff)
    phi_init = phi_init.astype(dtype)

    def body_fn(_, phi):
        return _step_Q(phi, mesh, dt, beta_cells, gamma_diff)

    return jax.lax.fori_loop(0, n_steps, body_fn, phi_init)


@partial(jax.jit, static_argnames=["n_steps"])
def solve_advdiff_equation_adjoint(phi_init, t_target, gamma_diff, mesh, beta_cells, n_steps):
    """
    Integrate the conservative Q† PDE  ∂_τφ = −div(βφ) + γΔφ  over [0, t_target].
    Returns Q†_{t_target}[φ_init].  ``n_steps`` is a static Python int.
    """
    dt = t_target / n_steps
    dtype    = jnp.result_type(phi_init, beta_cells, gamma_diff)
    phi_init = phi_init.astype(dtype)

    def body_fn(_, phi):
        return _step_Qadj(phi, mesh, dt, beta_cells, gamma_diff)

    return jax.lax.fori_loop(0, n_steps, body_fn, phi_init)
