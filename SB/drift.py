"""
drift.py  —  analytic reference drift β(t,x) for the diamond airfoil
====================================================================

Implements the physically-informed reference drift of ``referrence_drift.tex``
(§"Analytic reference drift for a diamond airfoil").  The drift is a sum of
windowed rigid rotations, one per shock ray, that carry the leading- and
trailing-edge shocks from their M₀ wave-angle toward their M₁ wave-angle:

    β(t,x) = Σ_{κ∈{LE,TE}} Σ_{σ∈{+1,-1}}
                 σ · ω_κ(t) · w_{κ,σ}(t,x) · R (x − a_κ),        R = [[0,−1],[1,0]]

with the wave angles from the oblique-shock θ–β–M relation (LE) and a
LE-shock → Prandtl–Meyer(2α) → TE-shock chain (TE).

Everything here is plain NumPy/SciPy (evaluated once per frame, not a JAX hot
path); only the returned per-cell array ``(N_cells, 2)`` is later handed to the
JAX advection solver.  ``reference_drift = "null"`` returns zeros → the pipeline
reduces exactly to the pure-heat bridge.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.spatial import cKDTree
from scipy.interpolate import NearestNDInterpolator

R_MAT = np.array([[0.0, -1.0], [1.0, 0.0]])       # 90° rotation


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry (read from mesh metadata; Mach-independent)
# ─────────────────────────────────────────────────────────────────────────────

def read_diamond_geometry(mesh):
    """
    Return (apex_le, apex_te, alpha) for a diamond mesh.

        apex_le = (cx − c/2, cy),   apex_te = (cx + c/2, cy),
        alpha   = arctan(height / chord)          (wedge half-angle)

    ``height`` in the metadata is the *total* diamond height H = c·tan α, so
    arctan(H/c) = α exactly (referrence_drift.tex, Geometry subsection).
    """
    md = mesh.metadata
    cx = float(md["center"]["cx"])
    cy = float(md["center"]["cy"])
    chord  = float(md["chord"])
    height = float(md["height"])
    apex_le = np.array([cx - chord / 2.0, cy])
    apex_te = np.array([cx + chord / 2.0, cy])
    alpha   = float(np.arctan(height / chord))
    return apex_le, apex_te, alpha


# ─────────────────────────────────────────────────────────────────────────────
#  Oblique-shock θ–β–M relation and the Prandtl–Meyer chain
# ─────────────────────────────────────────────────────────────────────────────

def theta_from_beta(beta_s, M, gamma_g=1.4):
    """Deflection θ produced by wave angle β_s at Mach M (the θ–β–M relation)."""
    num = M**2 * np.sin(beta_s)**2 - 1.0
    den = M**2 * (gamma_g + np.cos(2.0 * beta_s)) + 2.0
    return np.arctan(2.0 / np.tan(beta_s) * num / den)


def oblique_shock_beta(theta, M, gamma_g=1.4):
    """
    Weak-shock root β_s(θ, M) of the θ–β–M relation: the smaller root, lying
    between the Mach angle arcsin(1/M) and the θ_max detachment angle.
    Raises ValueError if the deflection exceeds θ_max(M) (shock detaches).
    """
    if M <= 1.0:
        raise ValueError(f"oblique_shock_beta: subsonic/sonic M={M}")
    mach_angle = np.arcsin(1.0 / M)
    # θ(β) rises from 0 at the Mach angle to θ_max, then falls to 0 at π/2.
    # Locate θ_max on a grid, then bracket the weak (pre-peak) root.
    betas   = np.linspace(mach_angle + 1e-6, np.pi / 2 - 1e-6, 400)
    thetas  = theta_from_beta(betas, M, gamma_g)
    i_peak  = int(np.argmax(thetas))
    theta_max = thetas[i_peak]
    if theta > theta_max:
        raise ValueError(
            f"deflection θ={np.degrees(theta):.2f}° exceeds θ_max="
            f"{np.degrees(theta_max):.2f}° at M={M:.3f} → shock detaches")
    def residual(b):
        return theta_from_beta(b, M, gamma_g) - theta
    return brentq(residual, mach_angle + 1e-9, betas[i_peak])


def prandtl_meyer(M, gamma_g=1.4):
    """Prandtl–Meyer function ν(M) (radians).  Raises on M<1: a subsonic Mach in
    the TE chain is a physics failure (detached/embedded-subsonic flow) that must
    surface, not be silently clamped."""
    if M < 1.0:
        raise ValueError(f"prandtl_meyer: subsonic M={M:.4f} (TE chain requires M≥1)")
    gm = gamma_g
    t  = np.sqrt((gm - 1.0) / (gm + 1.0) * (M**2 - 1.0))
    return (np.sqrt((gm + 1.0) / (gm - 1.0)) * np.arctan(t)
            - np.arctan(np.sqrt(M**2 - 1.0)))


def invert_prandtl_meyer(nu_target, gamma_g=1.4):
    """Invert ν(M) = nu_target for M > 1."""
    if nu_target <= 0.0:
        return 1.0
    def residual(M):
        return prandtl_meyer(M, gamma_g) - nu_target
    return brentq(residual, 1.0 + 1e-9, 100.0)


def beta_le(M, alpha, gamma_g=1.4):
    """Leading-edge wave angle β_s^LE(M) = β_s(α, M)."""
    return oblique_shock_beta(alpha, M, gamma_g)


def te_incoming_mach(M, alpha, gamma_g=1.4):
    """
    Mach M_b entering the TE shock: cross the LE shock, then expand by 2α around
    the shoulder (LE-shock → Prandtl–Meyer(2α) chain).
    """
    beta = beta_le(M, alpha, gamma_g)
    Mn1  = M * np.sin(beta)
    Mn2  = np.sqrt((1.0 + 0.5 * (gamma_g - 1.0) * Mn1**2)
                   / (gamma_g * Mn1**2 - 0.5 * (gamma_g - 1.0)))
    Ma   = Mn2 / np.sin(beta - alpha)                 # Mach behind LE shock
    Mb   = invert_prandtl_meyer(prandtl_meyer(Ma, gamma_g) + 2.0 * alpha, gamma_g)
    return Mb


def beta_te(M, alpha, gamma_g=1.4):
    """Trailing-edge wave angle β_s^TE(M) = β_s(α, M_b(M))."""
    Mb = te_incoming_mach(M, alpha, gamma_g)
    return oblique_shock_beta(alpha, Mb, gamma_g)


# ─────────────────────────────────────────────────────────────────────────────
#  Ray-angle magnitudes ψ_κ(M) and their rates dψ_κ/dM
# ─────────────────────────────────────────────────────────────────────────────

def _psi(M, alpha, gamma_g, kappa):
    """ψ_LE = β_LE ;  ψ_TE = β_TE − α  (ray-angle magnitude in the global frame)."""
    if kappa == "LE":
        return beta_le(M, alpha, gamma_g)
    return beta_te(M, alpha, gamma_g) - alpha


def _psi_and_rate(M, alpha, gamma_g, kappa, dM=1e-4):
    """(ψ_κ(M), dψ_κ/dM) via a central finite difference in M."""
    psi = _psi(M, alpha, gamma_g, kappa)
    dpsi_dM = (_psi(M + dM, alpha, gamma_g, kappa)
               - _psi(M - dM, alpha, gamma_g, kappa)) / (2.0 * dM)
    return psi, dpsi_dM


# ─────────────────────────────────────────────────────────────────────────────
#  Drift assembly
# ─────────────────────────────────────────────────────────────────────────────

def assemble_drift(barycenters, t, apex_le, apex_te, alpha, M0, M1,
                   gamma_g=1.4, sigma_w=0.2):
    """
    β(t,x) at every cell centre, shape (N_cells, 2).

    Schedule M(t) = (1−t)M₀ + tM₁ ; ω_κ(t) = (dβ_s^κ/dM)|_{M(t)}·(M₁−M₀).
    Sums the four windowed rigid rotations (LE/TE × upper/lower).
    """
    bary = np.asarray(barycenters, dtype=float)
    beta = np.zeros_like(bary)
    M    = (1.0 - t) * M0 + t * M1
    dMdt = (M1 - M0)

    for kappa, apex in (("LE", apex_le), ("TE", apex_te)):
        psi, dpsi_dM = _psi_and_rate(M, alpha, gamma_g, kappa)
        omega = dpsi_dM * dMdt                       # dψ/dt = dψ/dM · dM/dt
        # Conical band: the shock sweeps through Δψ = |ψ(M₁)−ψ(M₀)| between the
        # endpoints, so at along-ray distance s the endpoint shocks are ≈ s·Δψ
        # apart.  Widen the Gaussian half-width with s to cover the whole swept
        # wedge (a constant σ_w under-covers the outer ray — Finding 4).
        sweep = abs(_psi(M1, alpha, gamma_g, kappa) - _psi(M0, alpha, gamma_g, kappa))
        rel   = bary - apex[None, :]                 # x − a_κ           (N,2)
        Rrel  = rel @ R_MAT.T                         # R (x − a_κ)       (N,2)
        for sigma in (+1.0, -1.0):
            ang    = sigma * psi
            d_hat  = np.array([np.cos(ang), np.sin(ang)])
            s      = rel @ d_hat                      # along-ray coord   (N,)
            proj   = s[:, None] * d_hat[None, :]
            dist2  = np.sum((rel - proj)**2, axis=1)  # perp distance²    (N,)
            sig_r  = sigma_w + 0.5 * sweep * np.maximum(s, 0.0)
            w      = np.exp(-dist2 / sig_r**2) * (s > 0.0)
            beta  += (sigma * omega) * w[:, None] * Rrel
    return beta


def build_drift_sequence(mesh, t_array, M0, M1, gamma_g=1.4, sigma_w=0.2):
    """
    (N_frames, N_cells, 2) analytic drift over the frames in ``t_array``.
    """
    apex_le, apex_te, alpha = read_diamond_geometry(mesh)
    bary = np.asarray(mesh.barycenter, dtype=float)
    return np.stack([
        assemble_drift(bary, float(t), apex_le, apex_te, alpha,
                       M0, M1, gamma_g, sigma_w)
        for t in t_array], axis=0)


def make_inside_body(mesh):
    """
    Exact point-in-body predicate from mesh metadata.

    For the diamond (symmetric rhombus, half-chord h_c, half-height h_t centred
    at (cx,cy)) the body is  |x−cx|/h_c + |y−cy|/h_t ≤ 1  (cf. diamond.py's
    distance_diamond).  The old KD-tree test with a 4·median-NN threshold could
    never fire: the diamond's HALF-HEIGHT (≈0.044) is well below that threshold
    (≈0.1–0.2), so every in-body point passed as "fluid".
    """
    md   = mesh.metadata
    case = str(md.get("case", ""))
    if case == "diamond":
        cx = float(md["center"]["cx"]);  cy = float(md["center"]["cy"])
        hc = float(md["chord"])  / 2.0
        ht = float(md["height"]) / 2.0
        def inside(P):
            P = np.atleast_2d(np.asarray(P, dtype=float))
            return (np.abs(P[:, 0] - cx) / hc + np.abs(P[:, 1] - cy) / ht) <= 1.0
        return inside

    # Generic fallback (bump / unknown): far from every cell centre ⇒ outside the
    # fluid.  Valid for the outer void; unreliable for bodies thinner than `thr`.
    bary = np.asarray(mesh.barycenter, dtype=float)
    kd   = cKDTree(bary)
    thr  = 4.0 * float(np.median(kd.query(bary, k=2)[0][:, 1]))
    return lambda P: kd.query(np.atleast_2d(np.asarray(P, float)))[0] > thr


def flow_map(barycenters, beta_cells, sign=+1.0, n_sub=20, inside_body=None,
             domain_lo=None, domain_hi=None, n_retract=4):
    """
    Deterministic flow Φ of the drift over unit pseudo-time, integrated with the
    SAME frozen per-cell field `beta_cells` the FVM kernel uses, and with the SAME
    no-flux boundaries (β·n masked on every boundary face).  Midpoint (RK2) steps.

    This is the γ→0 limit of the kernel mean:  Q₁[id]/Q₁[1] → Φ_{+β},
    Q†₁[id]/Q†₁[1] → Φ_{−β}.  Integrating the unbounded ANALYTIC β instead leaves
    a phantom displacement in the bias — outward at the outflow edge, into the
    wedge at the shock feet — which is then subtracted from T/S.

    ``sign=+1`` gives the forward flow Φ_β (for T); ``sign=-1`` the backward flow
    Φ_{-β} (for S).  For a frozen field the two are exact inverses.  β≡0 returns
    the identity.

    ``inside_body`` (from `make_inside_body`) blocks the flow from penetrating
    the solid: any step landing inside is bisection-retracted toward the previous
    (fluid) position, mirroring the kernel's zero-flux wall condition.  Domain
    bounds clamp the outflow/farfield edges the same way the kernel's masked
    boundary faces do.
    """
    bary = np.asarray(barycenters, dtype=float)
    beta = float(sign) * np.asarray(beta_cells, dtype=float)
    if not np.any(beta):
        return bary.copy()

    sample = NearestNDInterpolator(bary, beta)      # the frozen kernel field
    lo = bary.min(axis=0) if domain_lo is None else np.asarray(domain_lo, float)
    hi = bary.max(axis=0) if domain_hi is None else np.asarray(domain_hi, float)

    def project(x_new, x_prev):
        """Clamp to the domain box; retract any step that penetrates the body."""
        x_new = np.clip(x_new, lo, hi)
        if inside_body is not None:
            for _ in range(n_retract):                    # bisection retract:
                bad = inside_body(x_new)                  # keeps most of the
                if not bad.any():                         # tangential motion
                    return x_new
                x_new[bad] = 0.5 * (x_new[bad] + x_prev[bad])
            bad = inside_body(x_new)
            x_new[bad] = x_prev[bad]                      # last resort: no motion
        return x_new

    x  = bary.copy()
    dt = 1.0 / int(n_sub)
    for _ in range(int(n_sub)):
        x_mid = project(x + 0.5 * dt * sample(x), x)      # midpoint
        x     = project(x + dt * sample(x_mid), x)
    return x


def null_drift(mesh):
    """Zero drift, shape (N_cells, 2): reduces the bridge to the pure-heat case."""
    return np.zeros((int(mesh.tris.shape[0]), 2), dtype=float)


def build_reference_drift(mesh, mode, t_array, M0, M1, gamma_g=1.4, sigma_w=0.2):
    """
    Dispatch on the Config ``reference_drift`` mode.

        "null"    → zeros for every frame  (pure-heat bridge, unchanged behaviour)
        "oblique" → analytic θ–β–M diamond drift

    Returns (N_frames, N_cells, 2).
    """
    if mode in (None, "null", "none", "off"):
        z = null_drift(mesh)
        return np.repeat(z[None], len(t_array), axis=0)
    if mode == "oblique":
        return build_drift_sequence(mesh, t_array, M0, M1, gamma_g, sigma_w)
    raise ValueError(f"unknown reference_drift mode {mode!r} (expected 'null' or 'oblique')")
