"""
resolution.py  —  Schrödinger Bridge solvers (1-D + 2-D)
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from functools import partial
from . import heat_solver
from . import advdiff_solver


# ─────────────────────────────────────────────────────────────────────────────
#  OU reference
# ─────────────────────────────────────────────────────────────────────────────

def ou_transition_density(x_target, x_source, t, theta, sigma, mean):
    theta    = jnp.maximum(theta, 1e-12)
    variance = sigma**2*(1.0-jnp.exp(-2.0*theta*t))/(2.0*theta)
    variance = jnp.maximum(variance, 1e-14)
    mean_t   = mean + (x_source-mean)*jnp.exp(-theta*t)
    log_k    = (-0.5*(x_target-mean_t)**2/variance
                - 0.5*jnp.log(2.0*jnp.pi*variance))
    return jnp.exp(log_k)

def propagate_ou_density(initial_density, x_grid, t, theta, sigma, mean, dx):
    t      = jnp.maximum(t, 1e-7)
    kernel = ou_transition_density(x_grid[:,None], x_grid[None,:], t, theta, sigma, mean)
    prop   = kernel @ (initial_density * dx)
    return prop / jnp.sum(prop * dx)

def compare_with_ou(rho_solution, initial_density, x_grid, t, theta, sigma, mean, dx):
    rho_ou = propagate_ou_density(initial_density, x_grid, t, theta, sigma, mean, dx)
    return rho_ou, jnp.sqrt(jnp.sum((rho_solution-rho_ou)**2*dx))


# ─────────────────────────────────────────────────────────────────────────────
#  1-D heat semigroup
# ─────────────────────────────────────────────────────────────────────────────

def _neumann_log_kernel(x, t, gamma, x_min, interval_length, num_images=5):
    x_col = x[:,None,None]; y_row = x[None,:,None]
    offsets = (2.0*interval_length)*jnp.arange(-num_images, num_images+1, dtype=x.dtype)
    direct    = x_col - (y_row + offsets[None,None,:])
    reflected = x_col + y_row - 2.0*x_min + offsets[None,None,:]
    log_d = -direct**2/(4.0*gamma*t) - 0.5*jnp.log(4.0*jnp.pi*gamma*t)
    log_r = -reflected**2/(4.0*gamma*t) - 0.5*jnp.log(4.0*jnp.pi*gamma*t)
    return logsumexp(jnp.logaddexp(log_d, log_r), axis=-1)

def _enforce_zero_flux(v):
    return v.at[0].set(v[1]).at[-1].set(v[-2])

def apply_logPt_exp(h, x, t, gamma, dx):
    t          = jnp.maximum(t, 1e-7)
    log_kernel = _neumann_log_kernel(x, t, gamma, x[0], x[-1]-x[0])
    weights    = jnp.ones_like(x).at[0].set(0.5).at[-1].set(0.5)
    log_w      = jnp.log(weights)
    row_norm   = logsumexp(log_kernel+log_w[None,:]+jnp.log(dx), axis=1)
    log_kernel = log_kernel - row_norm[:,None]
    return logsumexp(log_kernel+h[None,:]+log_w[None,:]+jnp.log(dx), axis=1)


# ─────────────────────────────────────────────────────────────────────────────
#  IPFP 1-D
# ─────────────────────────────────────────────────────────────────────────────

def apply_IPFP(log_mu0, log_mu1, x, gamma, dx, num_iter=50):
    def body_fn(_, val):
        f_k, g_k = val
        g_n = _enforce_zero_flux(log_mu1 - apply_logPt_exp(f_k, x, 1.0, gamma, dx))
        f_n = _enforce_zero_flux(log_mu0 - apply_logPt_exp(g_n, x, 1.0, gamma, dx))
        return f_n, g_n
    f, g = jax.lax.fori_loop(0, num_iter, body_fn,
                              (jnp.zeros_like(x), jnp.zeros_like(x)))
    return _enforce_zero_flux(f), _enforce_zero_flux(g)


# ─────────────────────────────────────────────────────────────────────────────
#  Bridge density / drift (1-D)
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_rho(f, g, x, t, gamma, dx):
    rho = jnp.exp(apply_logPt_exp(f, x, t, gamma, dx)
                + apply_logPt_exp(g, x, 1.0-t, gamma, dx))
    return rho / jnp.maximum(jnp.sum(rho*dx), 1e-14)

def retrieve_b(g, x, t, gamma, dx):
    g_t = apply_logPt_exp(g, x, 1.0-t, gamma, dx)
    dg  = jnp.gradient(g_t, dx).at[0].set(0.0).at[-1].set(0.0)
    return 2.0*gamma*dg

def compute_drift_field(g, x, t_array, gamma, dx):
    return jnp.array([retrieve_b(g, x, jnp.maximum(t, 1e-4), gamma, dx)
                      for t in t_array])


# ─────────────────────────────────────────────────────────────────────────────
#  2-D FVM heat semigroup
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=["n_steps"])
def apply_logPt_fvm(h, t, gamma, mesh, n_steps):
    t         = jnp.maximum(t, 1e-12)
    M_shift   = jnp.max(h)
    phi_safe  = jnp.exp(h - M_shift)
    # Floor at 1e-300 (a normal float64, FTZ-safe) rather than 1e-30: the floor
    # sets where the log-potential tail is truncated — log(1e-30)≈-69 vs
    # log(1e-300)≈-690.  At small γ the faint mass that must bridge *around* the
    # body has kernel weight exp(-d²/4γ); the 1e-30 floor clamps it to zero once
    # d²/4γ>69 (decoupling the two sides), while 1e-300 carries it to d²/4γ<690,
    # lowering the connectivity floor by ~10× in γ.  Only affects clamped cells.
    phi_final = jnp.maximum(
        heat_solver.solve_heat_equation(phi_safe, t, gamma, mesh, n_steps), 1e-300)
    return jnp.log(phi_final) + M_shift

@partial(jax.jit, static_argnames=["n_steps"])
def retrieve_rho_2d(f, g, t, gamma, mesh, n_steps):
    f_t = apply_logPt_fvm(f, t,       gamma, mesh, n_steps)
    g_t = apply_logPt_fvm(g, 1.0-t,   gamma, mesh, n_steps)
    rho = jnp.exp(f_t + g_t)
    return rho / jnp.maximum(jnp.sum(rho*mesh.area), 1e-14)

@partial(jax.jit, static_argnames=["n_steps"])
def retrieve_b_2d(g, t, gamma, mesh, n_steps):
    """b_t(x) = 2γ ∇g_t(x),  g_t = log P_{1-t} e^g."""
    g_t    = apply_logPt_fvm(g, 1.0-t, gamma, mesh, n_steps)
    grad_g = heat_solver.compute_scalar_gradient_LSQ(g_t, mesh)
    return 2.0*gamma*grad_g


# ─────────────────────────────────────────────────────────────────────────────
#  2-D FVM advection–diffusion semigroups  Q  and  Q†  (log domain)
#
#  Drifted generalisation of apply_logPt_fvm.  Q propagates the backward wave
#  function η* (non-conservative β·∇+γΔ); Q† propagates the forward η
#  (conservative −div(βu)+γΔ).  Both reduce to apply_logPt_fvm when β≡0.
#  See referrence_drift.tex eq. (ipfpdrift):
#      f = log μ − log Q₁[e^g],     g = log ν − log Q†₁[e^f].
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=["n_steps"])
def apply_logQt_fvm(h, t, gamma, mesh, beta_cells, n_steps):
    """log Q_t[e^h] — non-conservative backward semigroup (β·∇ + γΔ)."""
    t         = jnp.maximum(t, 1e-12)
    M_shift   = jnp.max(h)
    phi_safe  = jnp.exp(h - M_shift)
    phi_final = jnp.maximum(
        advdiff_solver.solve_advdiff_equation(
            phi_safe, t, gamma, mesh, beta_cells, n_steps), 1e-300)
    return jnp.log(phi_final) + M_shift


@partial(jax.jit, static_argnames=["n_steps"])
def apply_logQt_adjoint_fvm(h, t, gamma, mesh, beta_cells, n_steps):
    """log Q†_t[e^h] — conservative forward semigroup (−div(βu) + γΔ)."""
    t         = jnp.maximum(t, 1e-12)
    M_shift   = jnp.max(h)
    phi_safe  = jnp.exp(h - M_shift)
    phi_final = jnp.maximum(
        advdiff_solver.solve_advdiff_equation_adjoint(
            phi_safe, t, gamma, mesh, beta_cells, n_steps), 1e-300)
    return jnp.log(phi_final) + M_shift


@partial(jax.jit, static_argnames=["n_steps"])
def retrieve_b_2d_drift(g, t, gamma, mesh, beta_cells, n_steps):
    """Total optimal drift  b_t = β + 2γ ∇g_t,  g_t = log Q_{1-t}[e^g]  (drifted
    case).  β is the reference drift; 2γ∇log η* is the entropic correction."""
    g_t    = apply_logQt_fvm(g, 1.0 - t, gamma, mesh, beta_cells, n_steps)
    grad_g = heat_solver.compute_scalar_gradient_LSQ(g_t, mesh)
    return beta_cells + 2.0 * gamma * grad_g


@partial(jax.jit, static_argnames=["n_steps"])
def retrieve_rho_2d_drift(f, g, t, gamma, mesh, beta_cells, n_steps):
    """
    ρ_t = Q_t[e^f]·Q†_{1-t}... — the drifted marginal.  The forward wave function
    η_t is propagated by Q† from t=0, the backward η*_t by Q from t=1, so
    ρ_t = exp( log Q†_t[e^f] + log Q_{1-t}[e^g] ), normalised to unit mass.
    """
    f_t = apply_logQt_adjoint_fvm(f, t,       gamma, mesh, beta_cells, n_steps)
    g_t = apply_logQt_fvm(        g, 1.0 - t, gamma, mesh, beta_cells, n_steps)
    rho = jnp.exp(f_t + g_t)
    return rho / jnp.maximum(jnp.sum(rho * mesh.area), 1e-14)


# ─────────────────────────────────────────────────────────────────────────────
#  IPFP 2-D — JIT  (use after calibrating num_iter with the debug/Anderson run)
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=["num_iter","n_steps"])
def apply_IPFP_2d(log_mu0, log_mu1, gamma, mesh, num_iter=50, n_steps=100):
    def body_fn(_, val):
        f_k, g_k = val
        g_n = log_mu1 - apply_logPt_fvm(f_k, 1.0, gamma, mesh, n_steps)
        f_n = log_mu0 - apply_logPt_fvm(g_n, 1.0, gamma, mesh, n_steps)
        return f_n, g_n
    N = mesh.tris.shape[0]
    return jax.lax.fori_loop(0, num_iter, body_fn,
                              (jnp.zeros(N), jnp.zeros(N)))


# ─────────────────────────────────────────────────────────────────────────────
#  IPFP 2-D — Anderson(m) acceleration  ← USE THIS FOR PRODUCTION MACH RUNS
# ─────────────────────────────────────────────────────────────────────────────

def apply_IPFP_2d_anderson(log_mu0, log_mu1, gamma, mesh, n_steps,
                            num_iter=4000, tol=1e-5, m=5, print_every=50,
                            f_init=None, g_init=None):
    """
    Anderson(m)-accelerated IPFP.  3-7× fewer iterations than plain Sinkhorn.

    Each step costs the same as one plain Sinkhorn step (one forward + one
    backward heat solve), plus a negligible O(m²N) CPU least-squares solve.

    Convergence note
    ────────────────
    For gamma=0.002 on the bump mesh the plain Sinkhorn rate is
    ρ ≈ 0.9973/iter.  Reaching tol=1e-4 from cold start requires ~2150 plain
    iterations.  Anderson(5) typically converges in 400-700 iterations.

    Parameters
    ──────────
    m      : Anderson memory window (5 is usually optimal, up to 10 for noisy problems)
    tol    : convergence threshold on max‖Δf‖ (1e-4 is sufficient for CDI)
    f_init : warm-start for f (e.g. result from previous annealing stage)
    g_init : warm-start for g
    """
    N = mesh.tris.shape[0]
    f = jnp.zeros(N) if f_init is None else jnp.asarray(f_init)
    g = jnp.zeros(N) if g_init is None else jnp.asarray(g_init)

    X_hist: list = []   # concatenated (f,g) proposals,  each shape (2N,)
    R_hist: list = []   # corresponding residuals,        each shape (2N,)
    residuals    = []

    for k in range(num_iter):
        g_new = log_mu1 - apply_logPt_fvm(f, 1.0, gamma, mesh, n_steps)
        f_new = log_mu0 - apply_logPt_fvm(g_new, 1.0, gamma, mesh, n_steps)

        rf  = np.asarray(f_new - f)
        rg  = np.asarray(g_new - g)
        res = float(np.max(np.abs(rf)))
        residuals.append(res)

        if (k+1) % print_every == 0:
            eta = ""
            if len(residuals) >= 2*print_every and res > tol:
                window = 2*print_every
                rho_est = (res / residuals[-window]) ** (1.0/window)
                if 0.0 < rho_est < 1.0:
                    n_left  = int(np.ceil(np.log(tol/res) / np.log(rho_est)))
                    eta = f"  ρ={rho_est:.4f}  ETA ~{n_left} more iters"
            print(f"  Anderson IPFP  iter {k+1:5d}  res={res:.3e}{eta}")

        if res < tol:
            f, g = f_new, g_new
            print(f"  Converged at iter {k+1}  (res={res:.2e})")
            break

        X_hist.append(np.concatenate([np.asarray(f_new), np.asarray(g_new)]))
        R_hist.append(np.concatenate([rf, rg]))
        if len(X_hist) > m:
            X_hist.pop(0); R_hist.pop(0)

        mk = len(X_hist)
        if mk < 2:
            f, g = f_new, g_new
            continue

        R   = np.stack(R_hist, axis=1)
        X   = np.stack(X_hist, axis=1)
        RtR = R.T @ R
        lam = max(1e-12*float(np.trace(RtR)), 1e-30)
        try:
            c = np.linalg.solve(RtR + lam*np.eye(mk), np.ones(mk))
            c = c / c.sum()
        except np.linalg.LinAlgError:
            f, g = f_new, g_new
            continue

        x_aa = X @ c
        f    = jnp.array(x_aa[:N])
        g    = jnp.array(x_aa[N:])

    return f, g, residuals


# ─────────────────────────────────────────────────────────────────────────────
#  Drifted IPFP 2-D — Anderson(m) acceleration with a reference drift β
#
#  Same structure as apply_IPFP_2d_anderson but with the two half-steps of
#  referrence_drift.tex eq. (ipfpdrift):
#      f = log μ − log Q₁ [e^g]      (Q  = non-conservative backward semigroup)
#      g = log ν − log Q†₁[e^f]      (Q† = conservative   forward  semigroup)
#  When beta_cells ≡ 0 both operators equal the heat semigroup and this reduces
#  to apply_IPFP_2d_anderson exactly.
# ─────────────────────────────────────────────────────────────────────────────

def apply_IPFP_2d_anderson_drift(log_mu0, log_mu1, gamma, mesh, beta_cells, n_steps,
                                 num_iter=4000, tol=1e-5, m=5, print_every=50,
                                 f_init=None, g_init=None):
    """Anderson(m)-accelerated drifted IPFP.  See module header and the heat
    version apply_IPFP_2d_anderson for the acceleration details."""
    N = mesh.tris.shape[0]
    f = jnp.zeros(N) if f_init is None else jnp.asarray(f_init)
    g = jnp.zeros(N) if g_init is None else jnp.asarray(g_init)

    X_hist: list = []
    R_hist: list = []
    residuals    = []

    for k in range(num_iter):
        # eq. (ipfpdrift): f uses Q (non-conservative), g uses Q† (conservative)
        f_new = log_mu0 - apply_logQt_fvm(g, 1.0, gamma, mesh, beta_cells, n_steps)
        g_new = log_mu1 - apply_logQt_adjoint_fvm(f_new, 1.0, gamma, mesh, beta_cells, n_steps)

        rf  = np.asarray(f_new - f)
        rg  = np.asarray(g_new - g)
        res = float(np.max(np.abs(rf)))
        residuals.append(res)

        if (k+1) % print_every == 0:
            eta = ""
            if len(residuals) >= 2*print_every and res > tol:
                window = 2*print_every
                rho_est = (res / residuals[-window]) ** (1.0/window)
                if 0.0 < rho_est < 1.0:
                    n_left  = int(np.ceil(np.log(tol/res) / np.log(rho_est)))
                    eta = f"  ρ={rho_est:.4f}  ETA ~{n_left} more iters"
            print(f"  Drifted IPFP  iter {k+1:5d}  res={res:.3e}{eta}")

        if res < tol:
            f, g = f_new, g_new
            print(f"  Converged at iter {k+1}  (res={res:.2e})")
            break

        X_hist.append(np.concatenate([np.asarray(f_new), np.asarray(g_new)]))
        R_hist.append(np.concatenate([rf, rg]))
        if len(X_hist) > m:
            X_hist.pop(0); R_hist.pop(0)

        mk = len(X_hist)
        if mk < 2:
            f, g = f_new, g_new
            continue

        R   = np.stack(R_hist, axis=1)
        X   = np.stack(X_hist, axis=1)
        RtR = R.T @ R
        lam = max(1e-12*float(np.trace(RtR)), 1e-30)
        try:
            c = np.linalg.solve(RtR + lam*np.eye(mk), np.ones(mk))
            c = c / c.sum()
        except np.linalg.LinAlgError:
            f, g = f_new, g_new
            continue

        x_aa = X @ c
        f    = jnp.array(x_aa[:N])
        g    = jnp.array(x_aa[N:])

    return f, g, residuals


# ─────────────────────────────────────────────────────────────────────────────
#  IPFP 2-D — plain Python loop (convergence monitoring)
# ─────────────────────────────────────────────────────────────────────────────

def apply_IPFP_2d_debug(log_mu0, log_mu1, gamma, mesh, n_steps,
                         num_iter=500, tol=1e-5, print_every=50):
    N = mesh.tris.shape[0]
    f = g = jnp.zeros(N)
    rs = []
    for i in range(num_iter):
        g_new = log_mu1 - apply_logPt_fvm(f, 1.0, gamma, mesh, n_steps)
        f_new = log_mu0 - apply_logPt_fvm(g_new, 1.0, gamma, mesh, n_steps)
        res   = float(jnp.max(jnp.abs(f_new - f)))
        rs.append(res); f, g = f_new, g_new
        if (i+1) % print_every == 0:
            print(f"  IPFP 2D  iter {i+1:5d}  residual = {res:.3e}")
        if res < tol:
            print(f"  Converged at iter {i+1}"); break
    return f, g, rs