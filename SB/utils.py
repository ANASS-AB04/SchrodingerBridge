"""
utils.py  —  pipeline runners and physics helpers
"""

import os, sys
import time as _time
from functools import partial as _partial

_SB_DIR    = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SB_DIR)
_EULER_DIR = os.path.join(_REPO_ROOT, "Euler")
for _p in (_REPO_ROOT, _EULER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree, Delaunay

from . import heat_solver
from .resolution import (
    apply_IPFP,
    apply_IPFP_2d_anderson,
    apply_IPFP_2d_anderson_drift,
    apply_logPt_fvm,
    apply_logQt_fvm,
    apply_logQt_adjoint_fvm,
    compute_drift_field,
    retrieve_b_2d,
    retrieve_b_2d_drift,
    retrieve_rho,
    retrieve_rho_2d,
    retrieve_rho_2d_drift,
)
from . import drift as drift_mod
from . import advdiff_solver
from .plot import (
    plot_bridge_dashboard,
    plot_density_transport,
    plot_drift_field,
    plot_entropy,
    plot_l2_linf_errors,
    plot_abs_error_grid,
    plot_1d_sde_trajectories,
    plot_3d_evolution,
    plot_3d_density_surface,
)
from .testcases import get_1d_case, get_2d_case, gaussian_ot_interpolant_2d
from Euler.jax_fvm.src.mesh import Mesh
from Euler.jax_fvm.src.plot import plot_solution


def compute_bridge_entropy(rho_ref, rho_sequence, measure):
    """−∫ (ρ/ρ_ref) log(ρ/ρ_ref) dμ — negative entropy of the density ratio
    along the bridge.  Diagnostic only; not a KL divergence."""
    entropies = []
    for rho in rho_sequence:
        mass     = jnp.sum(rho * measure)
        rho_norm = rho / mass
        ratio    = jnp.maximum(rho_norm / rho_ref, 1e-14)
        entropies.append(float(-jnp.sum(ratio * jnp.log(ratio) * measure)))
    return np.array(entropies)


def compute_cell_gradient(mesh, field):
    N      = field.shape[0]
    valid  = mesh.neighbors >= 0
    safe_n = jnp.where(valid, mesh.neighbors, jnp.arange(N)[:, None])
    f_face = 0.5 * (field[:, None] + field[safe_n])
    fl     = mesh.surface[mesh.face_connectivity]
    flux   = f_face[..., None] * mesh.normals * fl[..., None]
    return jnp.sum(flux, axis=1) / mesh.area[:, None]


def compute_ducros_sensor(mesh, primitives, gamma_fluid=1.4, eps=0.01):
    rho, u_vel, v_vel, p = [primitives[:, i] for i in range(4)]
    a2     = gamma_fluid * p / jnp.maximum(rho, 1e-12)
    grad_u = compute_cell_gradient(mesh, u_vel)
    grad_v = compute_cell_gradient(mesh, v_vel)
    grad_p = compute_cell_gradient(mesh, p)

    div_v      = grad_u[:, 0] + grad_v[:, 1]
    curl_v     = grad_v[:, 0] - grad_u[:, 1]
    neg_div    = jnp.maximum(-div_v, 0.0)
    denom_kin  = div_v**2 + curl_v**2 + jnp.maximum(a2**2, 1e-12)
    phi_kin    = neg_div / jnp.sqrt(jnp.maximum(denom_kin, 1e-12))
    v_mag      = jnp.sqrt(u_vel**2 + v_vel**2)
    grad_p_mag = jnp.sqrt(grad_p[:, 0]**2 + grad_p[:, 1]**2)
    return phi_kin * (grad_p_mag*v_mag / jnp.maximum(p+ eps, 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
#  Iso-contour density  (density_mode = "iso")
# ─────────────────────────────────────────────────────────────────────────────
 
def build_isocontour_density(mach_field, mesh, mach_inlet,
                             n_contours=6, floor_pct=50.0,
                             verbose=True):
    """
    Density of n_contours equispaced Mach iso-contour lines in the active zone.

    Active zone = cells where |M − M_inlet| > floor_pct-th percentile of that
    field, which keeps contours across shocks AND expansion fans while dropping
    the flat freestream background.  Heat smoothing is applied by the caller
    (transform_to_shock_density via _maybe_heat) so the smooth_heat toggle in
    Config.toml is respected uniformly across all density modes.
    """
    area   = np.asarray(mesh.area).ravel()
    mach   = np.asarray(mach_field, dtype=float)

    # 1. Active zone: cells where the Mach number differs from the inlet value
    pert_mach   = np.abs(mach - float(mach_inlet))
    thresh      = float(np.percentile(pert_mach, floor_pct))
    active      = pert_mach > thresh
 
    # 2. Equispaced Mach levels over the active zone's Mach range
    mach_active  = mach[active]
    v_min, v_max = float(mach_active.min()), float(mach_active.max())
    levels = np.linspace(v_min, v_max, n_contours + 2)[1:-1]   # skip endpoints
 
    # Gather neighbour Mach values safely
    neighbors = np.asarray(mesh.neighbors)              
    safe_nbr  = np.maximum(neighbors, 0)               
    nbr_mach  = np.where(neighbors >= 0,
                         mach[safe_nbr],
                         mach[:, None])                
 
    # 3. Find cells bracketed by a contour level
    u = np.zeros(len(mach))
    for level in levels:
        brackets = ((mach[:, None] - level) * (nbr_mach - level) < 0).any(axis=1)
        u[brackets & active] = 1.0
 
    n_on = int((u > 0).sum())

    # 4. Spread binary mask into soft ridges via 3 neighbour-averaging steps
    # (reduces log-density dynamic range; cheaper than the FVM heat solve which
    # the caller applies on top when smooth_heat=True)
    for _ in range(3):
        u_nbr = (np.sum(u[safe_nbr] * (neighbors >= 0), axis=1)
                 / np.maximum(np.sum(neighbors >= 0, axis=1).astype(float), 1.0))
        u = np.maximum(u, 0.5 * u_nbr)

    mass = float(np.sum(u * area))
    if verbose:
        print(f"    iso-contour density: {n_contours} levels in "
              f"[{v_min:.3f}, {v_max:.3f}],  {n_on} contour cells,  "
              f"active={float(active.mean())*100:.1f}%  "
              f"(M_inlet={float(mach_inlet):.3f}, floor@{floor_pct:.0f}%={thresh:.3e})")

    return u / max(mass, 1e-30), pert_mach


def build_hessian_density(mach_field, mesh, mach_inlet,
                          lp=2.0, mode="det", floor_pct=50.0,
                          verbose=True):
    """
    Metric-based (Alauzet–Loseille) Hessian density of the Mach field.

    The marginal is the node density of the L^p-optimal anisotropic metric of M:

        sqrt(det M_Lp)  ∝  (det|H_M|)^{p/(2p+n)},   n = 2,

    so the exponent p/(2p+2) is the shock-vs-fan knob:
        lp → ∞  (expo → 1/2)  concentrates mass on the shocks,
        lp = 1  (expo → 1/4)  spreads it onto the weaker expansion fans.

    mode = "det" : rho ∝ (det|H|)^{lp/(2lp+2)} = |M_xx M_yy − M_xy²|^{lp/(2lp+2)}
                   (the faithful metric node density; can under-weight perfectly
                    straight shocks, where one Hessian eigenvalue ≈ 0).
    mode = "eig" : rho ∝ |λ_max|^{lp/(2lp+2)}   (isotropic feature strength; does
                   not under-weight straight shocks).

    Same logic as the iso mode: an active-zone gate keeps mass only where the
    flow is disturbed (|M − M∞| above the floor_pct percentile), discarding the
    freestream where the recovered Hessian is pure numerical noise; the result
    is then smoothed by the FVM heat equation (which also denoises ∂²M).

    Returns (density_unnormalised_/mass, pert_mach), mirroring build_isocontour_density.
    """
    area = np.asarray(mesh.area).ravel()
    mach = jnp.asarray(np.asarray(mach_field, dtype=float))

    # 1. Hessian via two FVM-gradient passes, symmetrised
    gM  = compute_cell_gradient(mesh, mach)          # M_x , M_y
    gMx = compute_cell_gradient(mesh, gM[:, 0])      # M_xx, M_xy
    gMy = compute_cell_gradient(mesh, gM[:, 1])      # M_yx, M_yy
    Hxx = gMx[:, 0]
    Hyy = gMy[:, 1]
    Hxy = 0.5 * (gMx[:, 1] + gMy[:, 0])

    # 2. feature field and the L^p node-density exponent  p/(2p+n), n = 2
    expo = lp / (2.0 * lp + 2.0)
    #expo= 1.0
    if mode == "det":
        feat = jnp.abs(Hxx * Hyy - Hxy**2)           # det|H| = |det H| in 2D
    elif mode == "eig":
        tr   = Hxx + Hyy
        disc = jnp.sqrt(jnp.maximum(0.25 * (Hxx - Hyy)**2 + Hxy**2, 0.0))
        feat = jnp.maximum(jnp.abs(0.5*tr + disc), jnp.abs(0.5*tr - disc))
    else:
        raise ValueError(f"unknown iso_hessian mode {mode!r}")
    feat_np = np.asarray(feat)
    raw     = np.power(np.maximum(feat_np, 0.0), expo)

    # 3. active-zone gate on the Mach perturbation (same logic as build_isocontour_density)
    mach_np   = np.asarray(mach)
    pert_mach = np.abs(mach_np - float(mach_inlet))
    thresh    = float(np.percentile(pert_mach, floor_pct))
    active    = pert_mach > thresh
    raw       = raw * active

    mass = float(np.sum(raw * area))
    if verbose:
        print(f"    Hessian density [{mode}, p={lp:g}, expo={expo:.3f}]: "
              f"active={float(active.mean())*100:.1f}%  "
              f"feat_max={feat_np.max():.3e}  "
              f"(M_inlet={float(mach_inlet):.3f}, floor@{floor_pct:.0f}%={thresh:.3e})")
    return raw / max(mass, 1e-30), pert_mach




# ─────────────────────────────────────────────────────────────────────────────
#  Shock density  — Ducros sensor → IPFP marginal
# ─────────────────────────────────────────────────────────────────────────────

def build_field_density(primitives, mesh, mach_inlet,
                        source="pert_mach", floor_pct=50.0,
                        gamma_fluid=1.4, combo_w=(0.5, 0.5),
                        eps_bg=1e-5, verbose=True):
    """
    Whole-field marginal capturing shocks AND expansion fans (unlike Ducros).
      source: "pert_mach" |M-M_inf|, "pert_p" |p-p_inf|, "grad_mach" ||grad M||,
              "grad_p" ||grad p||, "combo" (||grad M|| + |M-M_inf|).
    M_inf = mach_inlet if given else median.  Freestream removed by the
    floor_pct percentile.  Returns (raw_rectified_feature, feat) — the caller
    smooths, floors with eps_bg and normalises.
    """
    prims = jnp.asarray(np.asarray(primitives, dtype=float))
    rho, uu, vv, p = (prims[:, i] for i in range(4))
    a    = jnp.sqrt(jnp.maximum(gamma_fluid * p / jnp.maximum(rho, 1e-12), 1e-12))
    mach = jnp.sqrt(uu**2 + vv**2) / a

    def gmag(field):
        g = compute_cell_gradient(mesh, field)
        return jnp.sqrt(g[:, 0]**2 + g[:, 1]**2)

    M_inf = float(mach_inlet) if mach_inlet is not None else float(jnp.median(mach))
    if source == "pert_mach":
        feat = jnp.abs(mach - M_inf)
    elif source == "pert_p":
        feat = jnp.abs(p - jnp.median(p))
    elif source == "grad_mach":
        feat = gmag(mach)
    elif source == "grad_p":
        feat = gmag(p)
    elif source == "combo":
        gm = gmag(mach);  pm = jnp.abs(mach - M_inf)
        gm = gm / jnp.maximum(jnp.max(gm), 1e-12)
        pm = pm / jnp.maximum(jnp.max(pm), 1e-12)
        feat = combo_w[0]*gm + combo_w[1]*pm
    else:
        raise ValueError(f"unknown field source {source!r}")

    feat_np = np.asarray(feat)
    base    = float(np.percentile(feat_np, floor_pct))
    raw     = np.maximum(feat_np - base, 0.0)
    if verbose:
        print(f"    field density [{source}]: floor@{floor_pct:.0f}%={base:.3e}  "
              f"feat_max={feat_np.max():.3e}  M_inf={M_inf:.3f}")
    return raw, feat_np


def transform_to_shock_density(mesh, primitives, n_steps_smooth,
                                p=2, gamma_fluid=1.4,
                                eps_sensor=0.01, eps_bg=1e-5,
                                smooth_gamma=0.1, smooth_t=0.005,
                                apply_heat=True,
                                density_mode="mask", density_pct=92.0,
                                density_decay=0.05, tv_weight=1.0,
                                density_target="unit",
                                isocurve_order=1, isocurve_width=0.0,
                                iso_n_contours=6, iso_ducros_pct=50.0,
                                hessian_lp=2.0, hessian_mode="det",
                                field_source="pert_mach", field_floor_pct=50.0,
                                mach_inlet=None):
    """
    Shock-concentrated probability density.

    density_mode
    ────────────
    "mask"      : |φ − median(φ)|^p; heat-smoothed when apply_heat=True.
    "screened"  : screened-Poisson inpainting anchored at shock cells.
    "tv"        : total-variation inpainting (edge-preserving).
    "iso"       : Gaussian ridges along Mach iso-contour lines, gated by the
                  Ducros sensor (iso_n_contours lines, Ducros > iso_ducros_pct).
    "isocurve"  : thin PCA-fitted ridge along the Ducros iso-curve.
    "iso_hessian": Alauzet–Loseille metric node density (det|H_M|)^{p/(2p+2)}.
    "field"     : whole-field marginal |M-M_inf| / ||grad M|| (field_source).

    apply_heat  : when True, apply the FVM heat equation (γ=smooth_gamma,
                  t=smooth_t) to the raw density for IPFP conditioning.
                  For "mask" this smooths the weights; for all other modes it
                  is a post-processing step on the constructed density.

    Returns (density, sensor_np, smoothed_np).
    """
    def _maybe_heat(u_raw):
        """Apply heat smoothing if requested; always returns numpy array."""
        if not apply_heat:
            return np.asarray(u_raw)
        return np.asarray(heat_solver.solve_heat_equation(
            jnp.asarray(u_raw), t_target=smooth_t, gamma_diff=smooth_gamma,
            mesh=mesh, n_steps=n_steps_smooth))

    area     = jnp.asarray(mesh.area)
    prims_jx = jnp.array(np.asarray(primitives, dtype=float))
    phi      = compute_ducros_sensor(mesh, prims_jx,
                                     gamma_fluid=gamma_fluid, eps=eps_sensor)
    sensor_np  = np.asarray(phi)
    median_val = float(np.median(sensor_np))
    print(f"    sensor median={median_val:.3e}  max={sensor_np.max():.3e}")

    if density_mode == "mask":
        raw_weights = np.abs(sensor_np - median_val) ** p
        print(f"    weight max={raw_weights.max():.3e}")
        smoothed_np = _maybe_heat(raw_weights)
        if apply_heat:
            print(f"    Smoothing σ={np.sqrt(2*smooth_gamma*smooth_t):.4f}")
        density = (jnp.asarray(smoothed_np) + eps_bg) / jnp.sum((jnp.asarray(smoothed_np) + eps_bg) * area)
        return density, sensor_np, smoothed_np

    if density_mode in ("iso", "iso_hessian"):
        rho_j, u_j, v_j, p_j = (prims_jx[:, i] for i in range(4))
        a_j = jnp.sqrt(jnp.maximum(gamma_fluid * p_j / jnp.maximum(rho_j, 1e-12), 1e-12))
        mach_np = np.asarray(jnp.sqrt(u_j**2 + v_j**2) / a_j)
        ref = float(mach_inlet) if mach_inlet is not None else float(np.median(mach_np))
        if density_mode == "iso":
            u, pert_mach = build_isocontour_density(
                mach_np, mesh, ref,
                n_contours=iso_n_contours, floor_pct=iso_ducros_pct)
        else:
            u, pert_mach = build_hessian_density(
                mach_np, mesh, ref,
                lp=hessian_lp, mode=hessian_mode, floor_pct=iso_ducros_pct)
        u_raw   = np.asarray(u)      # raw density before heat (sharp weights for CDI blending)
        u       = _maybe_heat(u)
        u_jx    = jnp.asarray(u) + eps_bg
        density = u_jx / jnp.sum(u_jx * area)
        return density, pert_mach, u_raw

    if density_mode == "field":
        u, feat = build_field_density(
            prims_jx, mesh, mach_inlet,
            source=field_source, floor_pct=field_floor_pct)
        u_raw   = np.asarray(u)      # raw density before heat (sharp weights for CDI blending)
        u       = _maybe_heat(u)
        u_jx    = jnp.asarray(u) + eps_bg
        density = u_jx / jnp.sum(u_jx * area)
        return density, feat, u_raw

    # --- REST OF THE MODES (Mask, Screened, TV) ---
    phi      = compute_ducros_sensor(mesh, prims_jx,
                                     gamma_fluid=gamma_fluid, eps=eps_sensor)
    sensor_np  = np.asarray(phi)
    median_val = float(np.median(sensor_np))
    print(f"    sensor median={median_val:.3e}  max={sensor_np.max():.3e}")
    if density_mode == "isocurve":
        u = build_isocurve_density(
            sensor_np, mesh, pct=density_pct,
            curve_order=isocurve_order, width=isocurve_width)
        u_raw   = np.asarray(u)      # raw density before heat (sharp weights for CDI blending)
        u       = _maybe_heat(u)
        u_jx    = jnp.asarray(u) + eps_bg
        density = u_jx / jnp.sum(u_jx * area)
        return density, sensor_np, u_raw

    # screened / tv inpainting
    u = solve_anchored_density(
        sensor_np, mesh, pct=density_pct, decay_len=density_decay,
        mode=density_mode, target=density_target, tv_weight=tv_weight)
    u = _maybe_heat(u)
    u_jx    = jnp.asarray(u) + eps_bg
    density = u_jx / jnp.sum(u_jx * area)
    return density, sensor_np, u


# ─────────────────────────────────────────────────────────────────────────────
#  Error vs. high-resolution reference solutions
# ─────────────────────────────────────────────────────────────────────────────

def _sinkhorn_ot_cost(log_a, log_b, area, gamma_w, mesh, n_steps, num_iter):
    """
    Entropy-regularised OT cost ≈ W₂² between two densities on the mesh, via the
    convolutional-Wasserstein method (Solomon et al., ACM ToG 2015): the heat
    semigroup e^{γΔ} (to t=1) IS the Gibbs kernel exp(-d²/4γ) with ε = 4γ and
    cost C = d².  Log-domain Sinkhorn (reuses apply_logPt_fvm); area weights are
    carried inside the heat operator, so the scalings need none.

    Dual value at convergence:  OT_ε = ε ( <F,a> + <G,b> ),  ε = 4γ_w,
    with a = e^{log_a}, b = e^{log_b} and <.,.> the area-weighted inner product.
    """
    def body(_, FG):
        F, G = FG
        G = log_b - apply_logPt_fvm(F, 1.0, gamma_w, mesh, n_steps)
        F = log_a - apply_logPt_fvm(G, 1.0, gamma_w, mesh, n_steps)
        return (F, G)
    F0 = jnp.zeros_like(log_a)
    F, G = jax.lax.fori_loop(0, num_iter, body, (F0, F0))
    a = jnp.exp(log_a);  b = jnp.exp(log_b)
    return 4.0 * gamma_w * (jnp.sum(F * a * area) + jnp.sum(G * b * area))


_sinkhorn_ot_cost = _partial(jax.jit, static_argnames=["n_steps", "num_iter"])(_sinkhorn_ot_cost)


def wasserstein2_distance(mu0, mu1, area, gamma_w, mesh, n_steps, num_iter=200):
    """
    Debiased entropic W₂ (Sinkhorn divergence):
        S(a,b) = OT_ε(a,b) − ½OT_ε(a,a) − ½OT_ε(b,b),   W₂ ≈ sqrt(max(S,0)).
    The debiasing removes the entropic bias so S→0 when the fields coincide,
    making it a proper divergence comparable across methods.  Inputs are
    densities (normalised to unit area-weighted mass inside).
    """
    a = jnp.maximum(jnp.asarray(mu0), 1e-30)
    b = jnp.maximum(jnp.asarray(mu1), 1e-30)
    a = a / jnp.sum(a * area)
    b = b / jnp.sum(b * area)
    la, lb = jnp.log(a), jnp.log(b)
    c_ab = _sinkhorn_ot_cost(la, lb, area, gamma_w, mesh, n_steps, num_iter)
    c_aa = _sinkhorn_ot_cost(la, la, area, gamma_w, mesh, n_steps, num_iter)
    c_bb = _sinkhorn_ot_cost(lb, lb, area, gamma_w, mesh, n_steps, num_iter)
    return float(jnp.sqrt(jnp.maximum(c_ab - 0.5 * (c_aa + c_bb), 0.0)))


def _feature_density(mach_field):
    """|M − M∞| (M∞ = field median) — the disturbed-flow mass whose LOCATION the
    Wasserstein distance scores.  Returned un-normalised (caller normalises)."""
    m = np.asarray(mach_field, dtype=float)
    return np.maximum(np.abs(m - float(np.median(m))), 0.0)


def compute_interpolation_error(t_array, mach_bcdi, mach_lin,
                                ref_bundle_paths, mesh, output_dir,
                                wass_gamma=0.005, wass_iter=50,
                                compute_w2=True):
    """
    Compare interpolated Mach fields against high-resolution solver solutions
    at t = 0.1, 0.2, ..., 0.9 (ref_bundle_paths must be 9 paths in that order).

    Saves one |error| grid image per method plus a combined L2/L∞ vs t plot.
    Prints an L∞ summary table.
    Returns dict {method: {"l2": array, "linf": array, "w2": array}}.
    """
    area   = np.asarray(mesh.area)
    area_j = jnp.asarray(area)
    t_ref = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    methods = [("BaryCDI", mach_bcdi), ("Linear", mach_lin)]
    errors   = {name: {"l2": [], "linf": [], "w2": []} for name, _ in methods}
    abs_errs = {name: [] for name, _ in methods}
    n_steps_w = heat_solver.compute_n_steps(mesh, wass_gamma, CFL=0.5)
    if compute_w2:
        print(f"    Wasserstein W₂ on |M−M∞| density  (entropic, γ_w={wass_gamma}, "
              f"σ_w={np.sqrt(2*wass_gamma):.3f}, {n_steps_w} heat steps, "
              f"{wass_iter} Sinkhorn iters, debiased)")

    if compute_w2:
        hdr = (f"  │  {'t':>4s}  {'BaryCDI L∞':>12s}  {'Linear L∞':>12s}"
               f"  {'BaryCDI W₂':>12s}  {'Linear W₂':>12s}")
    else:
        hdr = f"  │  {'t':>4s}  {'BaryCDI L∞':>12s}  {'Linear L∞':>12s}"
    print("  ┌─ Interpolation error vs. high-resolution reference ─────────────────────────────────────")
    print(hdr)

    tol_frame = 0.5 / max(len(t_array) - 1, 1)
    for t_val, ref_path in zip(t_ref, ref_bundle_paths):
        idx = int(np.argmin(np.abs(t_array - t_val)))
        assert abs(t_array[idx] - t_val) <= tol_frame, (
            f"No frame at t={t_val}: nearest is t={t_array[idx]:.4f}. "
            f"Set n_frames such that t_ref values fall on the grid.")
        data = np.load(ref_path)
        mach_ref = data["mach"].astype(float)
        data.close()

        norm_ref = float(np.sqrt(np.sum(mach_ref**2 * area)))
        feat_ref_np = _feature_density(mach_ref)
        feat_ref    = jnp.asarray(feat_ref_np)

        # c_bb = OT(ref, ref) — same for all methods at this t; compute once
        if compute_w2:
            b_norm = jnp.maximum(jnp.sum(feat_ref * area_j), 1e-30)
            log_b  = jnp.log(jnp.maximum(feat_ref / b_norm, 1e-30))
            print(f"      W₂ t={t_val:.1f} [ref self-cost] ...", end=" ", flush=True)
            c_bb = _sinkhorn_ot_cost(log_b, log_b, area_j, wass_gamma, mesh, n_steps_w, wass_iter)
            print(f"{c_bb:.3e}", flush=True)

        row, w2_row = f"  │  {t_val:.1f}", ""
        for name, mach_seq in methods:
            diff    = mach_seq[idx] - mach_ref
            abs_err = np.abs(diff)
            l2_err  = float(np.sqrt(np.sum(diff**2 * area)) / max(norm_ref, 1e-30))
            li_err  = float(abs_err.max())
            if compute_w2:
                feat_pred = jnp.asarray(_feature_density(mach_seq[idx]))
                a_norm = jnp.maximum(jnp.sum(feat_pred * area_j), 1e-30)
                log_a  = jnp.log(jnp.maximum(feat_pred / a_norm, 1e-30))
                print(f"      W₂ t={t_val:.1f} [{name}] ...", end=" ", flush=True)
                c_ab = _sinkhorn_ot_cost(log_a, log_b, area_j, wass_gamma, mesh, n_steps_w, wass_iter)
                c_aa = _sinkhorn_ot_cost(log_a, log_a, area_j, wass_gamma, mesh, n_steps_w, wass_iter)
                w2_err = float(jnp.sqrt(jnp.maximum(c_ab - 0.5*(c_aa + c_bb), 0.0)))
                print(f"{w2_err:.3e}", flush=True)
            else:
                w2_err = float("nan")
            errors[name]["l2"].append(l2_err)
            errors[name]["linf"].append(li_err)
            errors[name]["w2"].append(w2_err)
            abs_errs[name].append(abs_err)
            row    += f"  {li_err:>12.4e}"
            if compute_w2:
                w2_row += f"  {w2_err:>12.4e}"
        print(row + w2_row)

    print("  └────────────────────────────────────────────────────────────────────────────────────────")

    for name in errors:
        errors[name]["l2"]   = np.array(errors[name]["l2"])
        errors[name]["linf"] = np.array(errors[name]["linf"])
        errors[name]["w2"]   = np.array(errors[name]["w2"])

    # One grid image per method
    for name, _ in methods:
        err_matrix = np.stack(abs_errs[name])   # (9, N_cells)
        prefix = os.path.join(output_dir, f"AbsErr_{name}")
        plot_abs_error_grid(mesh, err_matrix, t_ref, prefix, method_name=name)

    plot_l2_linf_errors(
        os.path.join(output_dir, "Mach_L2_Linf"), t_ref, errors,
        title="Mach interpolation error vs. high-resolution reference")

    import matplotlib.pyplot as _plt
    _fig, _ax = _plt.subplots(figsize=(8, 5))
    for _name in errors:
        _ax.plot(t_ref, errors[_name]["w2"], "o-", label=_name)
    _ax.set_xlabel("t");  _ax.set_ylabel(r"$W_2$ on $|M-M_\infty|$ density")
    _ax.set_title("Wasserstein-2 interpolation error vs. reference "
                  f"(entropic, γ_w={wass_gamma})")
    _ax.legend();  _ax.grid(True, alpha=0.3)
    _plt.tight_layout()
    _plt.savefig(os.path.join(output_dir, "Mach_Wasserstein.png"),
                 dpi=200, bbox_inches="tight");  _plt.close()

    return errors


 
# ─────────────────────────────────────────────────────────────────────────────
#  Barycentric SB-CDI  (gradient-free, smooth, exact at endpoints)
# ─────────────────────────────────────────────────────────────────────────────
#
#  T(x₀) = E[X₁ | X₀ = x₀] = P₁[y·e^g](x₀) / P₁[e^g](x₀)
#
#  Three heat-equation solves (one per coordinate plus a denominator).  No
#  gradient of g is taken, so the cell-to-cell LSQ noise that makes the
#  drift-CDI shear along shocks is absent.  The CDI uses the linearised
#  Iollo–Taddei eq. 10b barycentric formula (non-iterative).
 
from functools import partial as _partial


@_partial(jax.jit, static_argnames=["n_steps"])
def compute_sb_barycentric_maps(f, g, gamma, mesh, n_steps):
    """
    Boundary-corrected SB barycentric maps.

        T_raw(x₀) = P₁[y·e^g] / P₁[e^g],   S_raw(x₁) = P₁[x·e^f] / P₁[e^f]
        bias(x)   = P₁[y]   / P₁[1]    − x        ← identity-leak from Neumann BC
        T = T_raw − bias,    S = S_raw − bias

    The bias is what T would be if e^g were uniform.  Removing it makes T → x
    exactly in cells far from any shock (including boundary cells), which the
    raw formula does not because the heat semigroup with no-flux BCs warps the
    diffusion of a linear function near ∂Ω.
    """
    bary = mesh.barycenter

    # Forward T (raw)
    g_max = jnp.max(g);  e_g = jnp.exp(g - g_max)
    den_T  = heat_solver.solve_heat_equation(e_g,             1.0, gamma, mesh, n_steps)
    num_Tx = heat_solver.solve_heat_equation(bary[:,0] * e_g, 1.0, gamma, mesh, n_steps)
    num_Ty = heat_solver.solve_heat_equation(bary[:,1] * e_g, 1.0, gamma, mesh, n_steps)
    den_T  = jnp.maximum(den_T, 1e-30)
    T_raw  = jnp.stack([num_Tx / den_T, num_Ty / den_T], axis=-1)

    # Backward S (raw)
    f_max = jnp.max(f);  e_f = jnp.exp(f - f_max)
    den_S  = heat_solver.solve_heat_equation(e_f,             1.0, gamma, mesh, n_steps)
    num_Sx = heat_solver.solve_heat_equation(bary[:,0] * e_f, 1.0, gamma, mesh, n_steps)
    num_Sy = heat_solver.solve_heat_equation(bary[:,1] * e_f, 1.0, gamma, mesh, n_steps)
    den_S  = jnp.maximum(den_S, 1e-30)
    S_raw  = jnp.stack([num_Sx / den_S, num_Sy / den_S], axis=-1)

    # Identity bias  (one extra denominator + 2 numerators)
    ones = jnp.ones_like(e_g)
    den_id = heat_solver.solve_heat_equation(ones,        1.0, gamma, mesh, n_steps)
    num_ix = heat_solver.solve_heat_equation(bary[:, 0],  1.0, gamma, mesh, n_steps)
    num_iy = heat_solver.solve_heat_equation(bary[:, 1],  1.0, gamma, mesh, n_steps)
    den_id = jnp.maximum(den_id, 1e-30)
    bias   = jnp.stack([num_ix / den_id - bary[:, 0],
                        num_iy / den_id - bary[:, 1]], axis=-1)

    return T_raw - bias, S_raw - bias, den_T, den_S, bias

# ─────────────────────────────────────────────────────────────────────────────
#  Iso-curve helpers  (shared by the "isocurve" density mode, Option B)
#  fit_shock_curve / sample_shock_curve reduce a shock branch to a centreline.
#  NOTE: cross-field branch *matching* was removed — it injected ~0.5-unit
#  mismatches that SB then faithfully transported, tearing the field.  Iso-
#  curves now only build a clean per-field marginal; SB does all the transport.
# ─────────────────────────────────────────────────────────────────────────────

def fit_shock_curve(mask, barycenters, order: int = 1):
    """PCA + polynomial fit to the cells of one connected shock branch.

    Order 1 → straight line (good for oblique shocks on the diamond).
    Order 2 → quadratic (curved shocks like the bump).
    Returns dict with centroid, tangent/normal axes, polynomial coeffs in the
    normal direction, and the arc-length range [s_min, s_max] along the tangent.
    Returns None if too few cells.
    """
    pts = barycenters[mask]
    if len(pts) < max(3, order + 2):
        return None
    centroid = pts.mean(0)
    centered = pts - centroid
    cov      = centered.T @ centered / max(len(pts), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    tangent  = eigvecs[:, -1]                            # principal axis
    normal   = eigvecs[:, 0]
    s_vals   = centered @ tangent
    n_vals   = centered @ normal
    coeffs   = (np.polyfit(s_vals, n_vals, order) if order > 1
                else np.array([0.0]))
    return {"centroid": centroid, "tangent": tangent, "normal": normal,
            "coeffs": coeffs, "s_min": float(s_vals.min()),
            "s_max": float(s_vals.max())}


def sample_shock_curve(curve, n: int = 200):
    """Evaluate the curve at n equispaced arc-length samples (original coords)."""
    s   = np.linspace(curve["s_min"], curve["s_max"], n)
    off = (np.polyval(curve["coeffs"], s)
           if curve["coeffs"].size > 1 else np.zeros_like(s))
    return (curve["centroid"][None, :]
            + s[:, None] * curve["tangent"][None, :]
            + off[:, None] * curve["normal"][None, :])


def build_isocurve_density(phi, mesh, pct=92.0, curve_order=1, width=0.0,
                           min_size=10, n_samples=200, verbose=True):
    """Thin PCA-fitted ridge along the sensor iso-curve (density_mode='isocurve').

    Masks cells above the ``pct`` percentile of the sensor ``phi``, splits them
    into connected shock branches, fits each to a curve, and lays a Gaussian
    ridge of half-width ``width`` (0 → auto, 1.5× median cell size) along the
    sampled centrelines.  Returns the mass-normalised density (numpy)."""
    barycenters = np.asarray(mesh.barycenter)
    area        = np.asarray(mesh.area).ravel()
    sensor      = np.asarray(phi, dtype=float)

    if width <= 0.0:
        width = 1.5 * float(np.sqrt(np.median(area)))

    mask           = sensor > np.percentile(sensor, pct)
    adj            = build_cell_adjacency(mesh)
    labels, n_comp = find_connected_components(mask, adj, min_size=min_size)
    if n_comp == 0:
        raise RuntimeError("iso-curve density: no shock branches found "
                           "(lower --density-pct)")

    samples, kept = [], 0
    for k in range(n_comp):
        curve = fit_shock_curve(labels == k, barycenters, order=curve_order)
        if curve is None:
            continue
        samples.append(sample_shock_curve(curve, n=n_samples));  kept += 1
    if not samples:
        raise RuntimeError("iso-curve density: all branches too small to fit")
    samples = np.vstack(samples)

    d, _ = cKDTree(samples).query(barycenters)
    u    = np.exp(-(d / width) ** 2)
    mass = float(np.sum(u * area))
    if verbose:
        print(f"    iso-curve density: {kept}/{n_comp} branches fitted, "
              f"ridge width={width:.4f}, mass={mass:.3e}")
    return u / max(mass, 1e-30)


def compute_sb_barycentric_maps(f, g, gamma, mesh, n_steps):
    """
    Boundary-corrected SB barycentric maps (Stable Log-Domain Version).

        T_raw(x₀) = P₁[y·e^g] / P₁[e^g],   S_raw(x₁) = P₁[x·e^f] / P₁[e^f]
        bias(x)   = P₁[y]   / P₁[1]    − x

    Uses `apply_logPt_fvm` to compute the conditional expectations entirely 
    in the log-domain. This prevents the catastrophic `1e-30` denominator 
    underflow when evaluating small-gamma or long-distance kernels.
    """
    bary = mesh.barycenter
    X = bary[:, 0]
    Y = bary[:, 1]
    
    # Coordinate shift to ensure strict positivity for the log-domain operator
    # X_pos = X - min(X) + 1.0  =>  X = X_pos + min(X) - 1.0
    x_min = jnp.min(X)
    y_min = jnp.min(Y)
    X_pos = X - x_min + 1.0
    Y_pos = Y - y_min + 1.0
    
    log_X_pos = jnp.log(X_pos)
    log_Y_pos = jnp.log(Y_pos)

    # ── Forward Maps (T) ──
    log_den_T  = apply_logPt_fvm(g, 1.0, gamma, mesh, n_steps)
    log_num_Tx = apply_logPt_fvm(g + log_X_pos, 1.0, gamma, mesh, n_steps)
    log_num_Ty = apply_logPt_fvm(g + log_Y_pos, 1.0, gamma, mesh, n_steps)
    
    # Subtraction in log-domain = Division in linear-domain (Underflow-proof!)
    Tx = jnp.exp(log_num_Tx - log_den_T) + x_min - 1.0
    Ty = jnp.exp(log_num_Ty - log_den_T) + y_min - 1.0
    T_raw = jnp.stack([Tx, Ty], axis=-1)

    # ── Backward Maps (S) ──
    log_den_S  = apply_logPt_fvm(f, 1.0, gamma, mesh, n_steps)
    log_num_Sx = apply_logPt_fvm(f + log_X_pos, 1.0, gamma, mesh, n_steps)
    log_num_Sy = apply_logPt_fvm(f + log_Y_pos, 1.0, gamma, mesh, n_steps)
    
    Sx = jnp.exp(log_num_Sx - log_den_S) + x_min - 1.0
    Sy = jnp.exp(log_num_Sy - log_den_S) + y_min - 1.0
    S_raw = jnp.stack([Sx, Sy], axis=-1)

    # ── Identity bias ──
    h_zero = jnp.zeros_like(f)
    log_den_id = apply_logPt_fvm(h_zero, 1.0, gamma, mesh, n_steps)
    log_num_ix = apply_logPt_fvm(log_X_pos, 1.0, gamma, mesh, n_steps)
    log_num_iy = apply_logPt_fvm(log_Y_pos, 1.0, gamma, mesh, n_steps)
    
    ix = jnp.exp(log_num_ix - log_den_id) + x_min - 1.0
    iy = jnp.exp(log_num_iy - log_den_id) + y_min - 1.0
    bias = jnp.stack([ix - X, iy - Y], axis=-1)

    # NOTE: no trust gate here — the driftless heat bridge is self-adjoint, so the
    # numerator/denominator blow-ups in near-empty cells cancel in the T/S ratio
    # and the maps are already well-conditioned (this is the original behaviour).
    # A 4σ cap would instead revert the sharpest-shock cells — the ones that carry
    # the L∞ error and where transport helps most — collapsing BaryCDI to linear.
    # The gate lives only in the drifted branch, where Q≠Q† breaks that cancellation.
    #
    # Return jnp.exp(log_den) for the diagnostic prints.  bias_T = bias_S = bias
    # here: the self-adjoint heat kernel uses one identity bias for both directions
    # (the drifted branch needs separate Φ_β/Φ_{-β} references → distinct biases).
    return T_raw - bias, S_raw - bias, jnp.exp(log_den_T), jnp.exp(log_den_S), bias, bias


def _trust_gate(M, bary, cap):
    """
    A conditional mean cannot legitimately move further than the reference flow
    plus a few diffusion lengths.  Cells whose kernel support has collapsed (the
    "worst T/S" outliers) produce garbage displacements; revert those to the
    identity so CDI degrades locally to linear interpolation instead of sampling
    the field at a meaningless location.  ``cap`` is max|Φ−x| + 4σ (drifted) or
    4σ alone (β≡0, Φ=id) — the gate itself always measures displacement from the
    identity ``bary``, not from Φ.
    """
    r  = jnp.linalg.norm(M - bary, axis=-1, keepdims=True)
    ok = jnp.isfinite(r) & (r <= cap)
    return jnp.where(ok, M, bary)


@_partial(jax.jit, static_argnames=["n_steps"])
def compute_sb_barycentric_maps_drift(f, g, gamma, mesh, beta_cells,
                                      phi_fwd, phi_bwd, n_steps):
    """
    Drifted SB barycentric maps (stable log-domain), the β-generalisation of
    compute_sb_barycentric_maps.  The forward map T is built from g through Q₁
    (non-conservative backward semigroup), and the backward map S from f through
    Q†₁ (conservative forward semigroup).

    CRITICAL — the leak bias is measured against the drift's own deterministic
    REFLECTED flow Φ_β, NOT the identity and NOT the unbounded analytic flow.
    For the symmetric heat kernel P[id]=id in the interior, so P[id]/P[1]−id ≈ 0
    except in the Neumann-reflection strip — an identity bias removes only the
    boundary leak.  With a drift that fails: Q[id]−id ≈ the reference's own mean
    displacement ≈ β's transport, so an identity (or heat-P) bias deletes exactly
    the transport β provides — and the better β conditions the bridge, the more
    it deletes.  Measuring the kernel mean Q[id]/Q[1] against Φ_β (the REFLECTED
    flow of β over unit time, `drift.flow_map` with the same no-flux boundary and
    body-retraction the FVM kernel itself uses) isolates ONLY the diffusion/
    boundary leak and keeps β's transport in T/S:

        bias_T = Q₁[id]/Q₁[1]  − Φ_β        (forward flow,  from drift.flow_map(+β))
        bias_S = Q†₁[id]/Q†₁[1] − Φ_{-β}    (backward flow, from drift.flow_map(-β))

    Using the unreflected analytic β for Φ instead leaves a phantom displacement
    exactly where the kernel's boundary masking diverges from the free flow — the
    outflow edge (Φ exits, the kernel can't) and the shock feet on the wedge (Φ
    penetrates the body, the kernel can't) — which lands in bias_T/bias_S and is
    then subtracted from T/S on precisely the wall/outer-band cells that carry
    the "eig" marginal's mass.

    At β≡0, Φ_β=id and Q=Q†=P, so this reduces to compute_sb_barycentric_maps.
    ``phi_fwd``/``phi_bwd`` are the precomputed (N,2) REFLECTED flow-map
    positions (`drift.flow_map(..., inside_body=...)`), built from the SAME
    frozen ``beta_cells`` the kernel advects with.
    """
    bary = mesh.barycenter
    X = bary[:, 0]
    Y = bary[:, 1]

    x_min = jnp.min(X); y_min = jnp.min(Y)
    log_X_pos = jnp.log(X - x_min + 1.0)
    log_Y_pos = jnp.log(Y - y_min + 1.0)
    h_zero    = jnp.zeros_like(f)

    # ── Forward map T (from g, via Q) ──
    log_den_T  = apply_logQt_fvm(g,             1.0, gamma, mesh, beta_cells, n_steps)
    log_num_Tx = apply_logQt_fvm(g + log_X_pos, 1.0, gamma, mesh, beta_cells, n_steps)
    log_num_Ty = apply_logQt_fvm(g + log_Y_pos, 1.0, gamma, mesh, beta_cells, n_steps)
    Tx = jnp.exp(log_num_Tx - log_den_T) + x_min - 1.0
    Ty = jnp.exp(log_num_Ty - log_den_T) + y_min - 1.0
    T_raw = jnp.stack([Tx, Ty], axis=-1)

    # ── Backward map S (from f, via Q†) ──
    log_den_S  = apply_logQt_adjoint_fvm(f,             1.0, gamma, mesh, beta_cells, n_steps)
    log_num_Sx = apply_logQt_adjoint_fvm(f + log_X_pos, 1.0, gamma, mesh, beta_cells, n_steps)
    log_num_Sy = apply_logQt_adjoint_fvm(f + log_Y_pos, 1.0, gamma, mesh, beta_cells, n_steps)
    Sx = jnp.exp(log_num_Sx - log_den_S) + x_min - 1.0
    Sy = jnp.exp(log_num_Sy - log_den_S) + y_min - 1.0
    S_raw = jnp.stack([Sx, Sy], axis=-1)

    # ── Leak bias vs. the deterministic reflected flow Φ (see docstring) ──
    # Forward kernel mean  Q₁[id]/Q₁[1]  measured against Φ_β:
    log_den_iT = apply_logQt_fvm(h_zero,    1.0, gamma, mesh, beta_cells, n_steps)
    log_ixT    = apply_logQt_fvm(log_X_pos, 1.0, gamma, mesh, beta_cells, n_steps)
    log_iyT    = apply_logQt_fvm(log_Y_pos, 1.0, gamma, mesh, beta_cells, n_steps)
    meanT_x = jnp.exp(log_ixT - log_den_iT) + x_min - 1.0
    meanT_y = jnp.exp(log_iyT - log_den_iT) + y_min - 1.0
    bias_T  = jnp.stack([meanT_x - phi_fwd[:, 0], meanT_y - phi_fwd[:, 1]], axis=-1)

    # Backward kernel mean  Q†₁[id]/Q†₁[1]  measured against Φ_{-β}:
    log_den_iS = apply_logQt_adjoint_fvm(h_zero,    1.0, gamma, mesh, beta_cells, n_steps)
    log_ixS    = apply_logQt_adjoint_fvm(log_X_pos, 1.0, gamma, mesh, beta_cells, n_steps)
    log_iyS    = apply_logQt_adjoint_fvm(log_Y_pos, 1.0, gamma, mesh, beta_cells, n_steps)
    meanS_x = jnp.exp(log_ixS - log_den_iS) + x_min - 1.0
    meanS_y = jnp.exp(log_iyS - log_den_iS) + y_min - 1.0
    bias_S  = jnp.stack([meanS_x - phi_bwd[:, 0], meanS_y - phi_bwd[:, 1]], axis=-1)

    T = T_raw - bias_T
    S = S_raw - bias_S

    # ── Trust gate: revert kernel-collapse outliers to the identity ──
    sigma = jnp.sqrt(2.0 * gamma)
    cap_T = jnp.max(jnp.linalg.norm(phi_fwd - bary, axis=-1)) + 4.0 * sigma
    cap_S = jnp.max(jnp.linalg.norm(phi_bwd - bary, axis=-1)) + 4.0 * sigma
    T = _trust_gate(T, bary, cap_T)
    S = _trust_gate(S, bary, cap_S)

    return T, S, jnp.exp(log_den_T), jnp.exp(log_den_S), bias_T, bias_S


# ─────────────────────────────────────────────────────────────────────────────
#  Barycentric SB-CDI reconstruction  (Iollo–Taddei eq. 10b, no iteration)
# ─────────────────────────────────────────────────────────────────────────────

def build_cdi_interpolators(barycenters, mach0, mach1, barycenter_delaunay, mesh):
    """
    Build the CDI interpolators once (outside the frame loop).

    Returns (interp0, interp1, inside_body, domain_lo, domain_hi).  The nearest-
    neighbour interpolators are gone: a query point that lands inside the body or
    outside the mesh is no longer snapped to the nearest fluid cell (which can sit
    on the WRONG SIDE of the airfoil) — the reconstruction falls back to the
    undisplaced endpoint value instead.
    """
    interp0 = LinearNDInterpolator(barycenter_delaunay, mach0, fill_value=np.nan)
    interp1 = LinearNDInterpolator(barycenter_delaunay, mach1, fill_value=np.nan)
    inside_body = drift_mod.make_inside_body(mesh)
    bary = np.asarray(barycenters, dtype=float)
    return interp0, interp1, inside_body, bary.min(axis=0), bary.max(axis=0)


def reconstruct_mach_barycentric_cdi(t, T_map, S_map, barycenters,
                                     mach0, mach1,
                                     interp0, interp1,
                                     inside_body, domain_lo, domain_hi):
    """
    û(t,x) = (1−t) M₀( (1−t)x + t S(x) )  +  t M₁( t x + (1−t) T(x) )

    T = forward map μ₀→μ₁ (from g), S = backward map μ₁→μ₀ (from f).
    Gradient-free, exact at the endpoints: û(0)=M₀, û(1)=M₁.

    Query points are clamped to the mesh bounding box, then any point landing
    inside the solid body (exact predicate) or outside the Delaunay hull falls
    back to the UNDISPLACED endpoint value at the cell itself — i.e. the identity
    map locally — rather than to the nearest fluid cell to the garbage point (that
    nearest cell can sit on the wrong side of the airfoil).
    """
    mach0 = np.asarray(mach0, dtype=float)
    mach1 = np.asarray(mach1, dtype=float)
    if t <= 0.0:
        return mach0.copy()
    if t >= 1.0:
        return mach1.copy()

    W  = (1.0 - t) * barycenters + t * S_map        # pre-image sampled in M₀
    Tg = t * barycenters + (1.0 - t) * T_map        # pre-image sampled in M₁

    W  = np.clip(W,  domain_lo, domain_hi)
    Tg = np.clip(Tg, domain_lo, domain_hi)

    M0 = np.asarray(interp0(W),  dtype=float)
    M1 = np.asarray(interp1(Tg), dtype=float)

    bad0 = inside_body(W)  | ~np.isfinite(M0)
    bad1 = inside_body(Tg) | ~np.isfinite(M1)
    M0[bad0] = mach0[bad0]                          # identity-map fallback
    M1[bad1] = mach1[bad1]

    return (1.0 - t) * M0 + t * M1


# ─────────────────────────────────────────────────────────────────────────────
#  Cell connectivity helpers  (used by iso-curve transport and CC masking)
# ─────────────────────────────────────────────────────────────────────────────

def build_cell_adjacency(mesh):
    """Cell-to-cell adjacency list straight from mesh.neighbors (−1 = boundary)."""
    neighbors = np.asarray(mesh.neighbors)
    return [[int(j) for j in row if j >= 0] for row in neighbors]


def find_connected_components(mask, adj, min_size: int = 20):
    """BFS connected components on the cell graph, restricted to `mask`.

    Returns (labels, n_components):  labels[i] ∈ [0, n) or −1 if cell i is not
    in a kept component (components smaller than min_size are discarded)."""
    labels = -np.ones(len(mask), dtype=np.int32)
    next_label = 0
    sizes = []
    for start in np.where(mask)[0]:
        if labels[start] >= 0:
            continue
        stack = [int(start)];  labels[start] = next_label;  sz = 1
        while stack:
            i = stack.pop()
            for j in adj[i]:
                if mask[j] and labels[j] < 0:
                    labels[j] = next_label;  stack.append(int(j));  sz += 1
        sizes.append(sz);  next_label += 1
    keep    = [k for k in range(next_label) if sizes[k] >= min_size]
    relabel = -np.ones(next_label, dtype=np.int32)
    for new_k, old_k in enumerate(keep):
        relabel[old_k] = new_k
    valid = labels >= 0
    labels[valid] = relabel[labels[valid]]
    return labels, len(keep)


# ─────────────────────────────────────────────────────────────────────────────
#  Shock-anchored density  —  Rudin–Osher–Fatemi / screened-Poisson inpainting
#
#      −κ Δu + (ρ(x) + κ/ℓ²) u = ρ(x) f ,     ∂ₙu|∂Ω = 0
#
#  ρ(x) = λ_fid · trust(f):  large at shocks  ⇒  u ≈ f there (sensor magnitude
#  preserved exactly);  ≈ 0 in expansion / decompression zones  ⇒  those noisy
#  sensor values are never read — u is harmonically (screened) or TV inpainted.
#  ℓ = decay length of the off-shock skirt (replaces the Gaussian smoothing σ).
#  Homogeneous Neumann at every boundary ⇒ mass cannot leak through the outer
#  box or into the airfoil body.
# ─────────────────────────────────────────────────────────────────────────────

def _internal_face_arrays(mesh):
    """Undirected internal faces (i<j) with TPFA transmissibility T = ℓ_face / d.

    Boundary faces (neighbour < 0) are dropped → zero-flux (Neumann)."""
    neighbors = np.asarray(mesh.neighbors)                 # (N,3), −1 at boundary
    face_conn = np.asarray(mesh.face_connectivity)         # (N,3) → global face id
    surface   = np.asarray(mesh.surface).ravel()           # face (edge) lengths
    bary      = np.asarray(mesh.barycenter)                # (N,2)
    N         = neighbors.shape[0]

    cell = np.repeat(np.arange(N), 3)
    nbr  = neighbors.reshape(-1)
    fid  = face_conn.reshape(-1)
    keep = (nbr >= 0) & (cell < nbr)                       # internal + dedup
    i, j, fid = cell[keep], nbr[keep], fid[keep]
    d  = np.linalg.norm(bary[i] - bary[j], axis=1)
    T  = surface[fid] / np.maximum(d, 1e-14)
    return i.astype(np.int64), j.astype(np.int64), T.astype(float)


def build_trust_map(phi, pct=92.0, ramp=0.10):
    """Smooth sigmoid trust ramp ∈(0,1): ≈1 at shocks, ≈0 in smooth/expansion
    regions.  Replaces the hard percentile mask, so no threshold step remains."""
    tau = np.percentile(phi, pct)
    s   = max(ramp * abs(tau), 1e-12)
    return 1.0 / (1.0 + np.exp(-(phi - tau) / s))


def solve_anchored_density(phi, mesh, pct=92.0, lambda_fid=None,
                           decay_len=0.05, mode="screened", target="unit",
                           tv_outer=12, tv_beta=1e-3, tv_weight=1.0,
                           kappa=1.0, verbose=True):
    """Inpaint the trusted shock band into a shock-anchored probability density.

    mode="screened":  one sparse SPD solve (linear).
    mode="tv":        Rudin–Osher–Fatemi total variation via lagged diffusivity
                      (edge-preserving plateaus that do not bleed across slip
                      lines behind the body);  tv_weight scales the TV term.

    target
    ──────
    "unit"   : pin u→1 on the trusted band (a smooth shock *indicator*).  The
               sensor MAGNITUDE is discarded, exactly as the binary mask does,
               so IPFP transports shock geometry only — no parasitic matching of
               the (M-dependent) strength profile along the shock.  DEFAULT.
    "sensor" : pin u≈f (keeps the sensor magnitude).  Reintroduces strength-
               profile transport; kept only for comparison.

    Returns the normalised density μ = u / ∫u dA (numpy, length N_cells).
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    area = np.asarray(mesh.area).ravel()
    f    = np.asarray(phi, dtype=float)
    f    = f / max(float(f.max()), 1e-30)                  # scale-free target
    N    = len(f)

    if lambda_fid is None:
        h_typ      = float(np.sqrt(np.median(area)))
        lambda_fid = 100.0 / h_typ**2                      # ≈1 % pinning error
    rho = lambda_fid * build_trust_map(f, pct=pct)
    eps = kappa / float(decay_len)**2
    tgt = np.ones_like(f) if target == "unit" else f      # discard magnitude (unit)
    rhs = area * rho * tgt

    I, J, T0 = _internal_face_arrays(mesh)

    def assemble(weights):
        w    = kappa * T0 * weights                        # edge conductances
        rows = np.concatenate([I, J, I, J])
        cols = np.concatenate([J, I, I, J])
        vals = np.concatenate([-w, -w,  w,  w])            # symmetric graph Lap.
        L    = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
        return (L + sp.diags(area * (rho + eps))).tocsc()  # SPD

    if mode == "screened":
        u = spla.spsolve(assemble(np.ones_like(T0)), rhs)
    elif mode == "tv":
        u  = f.copy()
        di = np.linalg.norm(np.asarray(mesh.barycenter)[I]
                            - np.asarray(mesh.barycenter)[J], axis=1)
        di = np.maximum(di, 1e-14)
        for it in range(tv_outer):
            grad = np.abs(u[I] - u[J]) / di
            wgt  = tv_weight / np.sqrt(grad**2 + tv_beta**2)
            u    = spla.spsolve(assemble(wgt), rhs)
            if verbose and (it + 1) % 4 == 0:
                print(f"      TV lagged-diffusivity iter {it+1}/{tv_outer}")
    else:
        raise ValueError(f"unknown density mode {mode!r}")

    u    = np.maximum(u, 0.0)
    mass = float(np.sum(u * area))
    if verbose:
        frac = float((rho > 0.5 * lambda_fid).mean()) * 100.0
        print(f"    anchored density [{mode}/{target}]:  λ_fid={lambda_fid:.3e}  "
              f"ℓ={decay_len}  trusted≈{frac:.1f}%  mass={mass:.3e}")
    return u / max(mass, 1e-30)


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline runners
# ─────────────────────────────────────────────────────────────────────────────

def run_1d_case(test_case, output_dir="outputs", gamma=0.5, num_iter=500):
    print(f"\n{'─'*55}\n  1D case: {test_case}\n{'─'*55}")
    os.makedirs(output_dir, exist_ok=True)

    N, L    = 400, 6.0
    x       = jnp.linspace(-L, L, N)
    dx      = x[1] - x[0]
    t_array = np.linspace(0.0, 1.0, 101)
    prefix  = os.path.join(output_dir, f"SB_{test_case}_gamma{gamma}")

    mu0, mu1, rho_ref, ref_label = get_1d_case(test_case, x, t_array, dx)
    eps = 1e-7
    log_mu0 = jnp.log(jnp.maximum(mu0, eps))
    log_mu1 = jnp.log(jnp.maximum(mu1, eps))

    print("  Running 1D IPFP …")
    f, g = jax.jit(apply_IPFP, static_argnames=["num_iter"])(
        log_mu0, log_mu1, x, gamma, dx, num_iter=num_iter)

    rho_seq   = np.array([retrieve_rho(f, g, x, t, gamma, dx) for t in t_array])
    b_field   = compute_drift_field(g, x, t_array, gamma, dx)
    entropies = compute_bridge_entropy(mu1, rho_seq, dx)

    # L2 / L∞ error vs reference at each t
    dx_np = float(dx)
    abs_diff  = np.abs(rho_seq - np.asarray(rho_ref))
    l2_errors = np.array([
        float(np.sqrt(np.sum(abs_diff[i]**2) * dx_np))
        for i in range(len(t_array))
    ])
    linf_errors = np.array([float(abs_diff[i].max()) for i in range(len(t_array))])
    print(f"  Validation — L2(SB, {ref_label}):")
    print(f"    max={l2_errors.max():.4e}  mean={l2_errors.mean():.4e}  "
          f"at t=0: {l2_errors[0]:.4e}  at t=0.5: {l2_errors[len(t_array)//2]:.4e}  "
          f"at t=1: {l2_errors[-1]:.4e}")

    plot_bridge_dashboard(prefix, np.array(x), np.array(mu0), np.array(mu1),
                          t_array, rho_seq, rho_ref, np.array(b_field),
                          ref_label=ref_label)
    plot_3d_evolution(prefix, np.array(x), t_array, rho_seq,
                      title=f"SB Evolution: {test_case}")
    plot_entropy(prefix, t_array, entropies, title=f"Entropy ({test_case})")
    plot_1d_sde_trajectories(prefix, np.array(x), t_array, np.array(b_field),
                              np.array(mu0), np.array(mu1),
                              gamma=gamma, num_particles=20)
    plot_l2_linf_errors(
        prefix, t_array,
        {f"SB vs {ref_label}": {"l2": l2_errors, "linf": linf_errors}},
        title=f"1D Validation: error(SB, {ref_label})  γ={gamma}",
        l2_formula=(r"\|\rho-\rho_{\mathrm{ref}}\|_{L^2}"
                    r"=\sqrt{\sum_i (\rho_i-\rho_i^{\mathrm{ref}})^2\,\Delta x}"),
        linf_formula=(r"\|\rho-\rho_{\mathrm{ref}}\|_{L^\infty}"
                      r"=\max_i |\rho_i-\rho_i^{\mathrm{ref}}|"))
    print(f"  Done → {output_dir}/")


def run_2d_case(test_case, output_dir="outputs", gamma=0.05, num_iter=1000):
    print(f"\n{'─'*55}\n  2D case: {test_case}\n{'─'*55}")
    os.makedirs(output_dir, exist_ok=True)

    mesh = Mesh()
    mesh.mesh_generator(maxV=1e-3, marker_boundary=1,
                         x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0)
    mesh.save_mesh(os.path.join(output_dir, f"{test_case}_mesh.vtk"))
    n_steps = heat_solver.compute_n_steps(mesh, gamma)
    print(f"    FVM steps: {n_steps}")

    t_array = np.linspace(0.0, 1.0, 10)
    prefix  = os.path.join(output_dir, f"SB_{test_case}_gamma{gamma}")
    mu0, mu1 = get_2d_case(test_case, mesh)

    f, g, _ = apply_IPFP_2d_anderson(jnp.log(mu0), jnp.log(mu1),
                          gamma, mesh, num_iter=num_iter, n_steps=n_steps, m=5)

    rho_seq   = np.stack([np.asarray(retrieve_rho_2d(f, g, float(t), gamma, mesh, n_steps))
                          for t in t_array])
    rho_seq  /= np.sum(rho_seq * np.asarray(mesh.area), axis=1, keepdims=True)
    drift_seq = np.stack([np.asarray(retrieve_b_2d(g, float(t), gamma, mesh, n_steps))
                          for t in t_array])
    entropies = compute_bridge_entropy(mu1, rho_seq, jnp.asarray(mesh.area))

    plot_3d_density_surface(mesh, mu0, prefix+"_source", title="Source (t=0)")
    plot_3d_density_surface(mesh, mu1, prefix+"_target", title="Target (t=1)")
    plot_density_transport(mesh, rho_seq, t_array, prefix)
    plot_drift_field(mesh, drift_seq, t_array, prefix, time_index=5)
    plot_entropy(prefix, t_array, entropies, title=f"Entropy ({test_case})")

    # ── Exact OT validation for Gaussian → Gaussian ───────────────────────────
    if test_case == "2d_gauss_to_gauss":
        print("  Computing exact W₂ OT reference (isotropic Gaussian geodesic) …")
        # Parameters must match get_2d_case("2d_gauss_to_gauss")
        rho_ot  = gaussian_ot_interpolant_2d(
            mesh, [0.4, -0.4], 0.1, [-0.4, 0.4], 0.03, t_array)
        area_np   = np.asarray(mesh.area)
        diff_ot   = rho_seq - rho_ot
        diff_lin  = np.stack([(1-t)*np.asarray(mu0) + t*np.asarray(mu1)
                              for t in t_array]) - rho_ot
        l2_ot     = np.array([float(np.sqrt(np.sum(diff_ot[i]**2 * area_np)))
                              for i in range(len(t_array))])
        l2_lin    = np.array([float(np.sqrt(np.sum(diff_lin[i]**2 * area_np)))
                              for i in range(len(t_array))])
        linf_ot   = np.array([float(np.abs(diff_ot[i]).max()) for i in range(len(t_array))])
        linf_lin  = np.array([float(np.abs(diff_lin[i]).max()) for i in range(len(t_array))])
        print(f"  Validation — L2(SB, exact OT):  "
              f"max={l2_ot.max():.4e}  mean={l2_ot.mean():.4e}")
        print(f"  Baseline  — L2(linear, exact OT): "
              f"max={l2_lin.max():.4e}  mean={l2_lin.mean():.4e}")

        plot_density_transport(mesh, rho_ot, t_array, prefix + "_exact_OT",
                               label=r"$\rho^{OT}$")
        plot_l2_linf_errors(
            prefix, t_array,
            {"SB vs exact OT": {"l2": l2_ot, "linf": linf_ot},
             "Linear vs exact OT": {"l2": l2_lin, "linf": linf_lin}},
            title=f"2D Validation: error vs exact W₂ OT  (γ={gamma})")

    print(f"  Done → {output_dir}/")

# ─────────────────────────────────────────────────────────────────────────────
#  Custom Contour Plotter for Raw Mach Fields
# ─────────────────────────────────────────────────────────────────────────────
def plot_mach_contours(mesh, mach_field, title, filename, levels=8):
    """
    Plots the Mach field contour lines (without the background heatmap).
    FVM data is cell-centered, but Matplotlib contours require node-centered
    data. This function performs a safe volume-weighted average to project
    the cells to the nodes prior to contouring.
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    pts = np.asarray(mesh.points)
    tris = np.asarray(mesh.tris)
    mach = np.asarray(mach_field)
    areas = np.asarray(mesh.area).ravel()

    # Project cell-centered Mach data onto nodes
    node_mach = np.zeros(len(pts))
    node_weights = np.zeros(len(pts))

    for i, tri in enumerate(tris):
        for node in tri:
            node_mach[node] += mach[i] * areas[i]
            node_weights[node] += areas[i]
    
    node_mach /= np.maximum(node_weights, 1e-14)

    fig, ax = plt.subplots(figsize=(8, 4))
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)

    # Plot standalone colored contour lines
    contour = ax.tricontour(triang, node_mach, levels=levels, cmap='viridis', 
                            linewidths=1.5, alpha=1.0)
    fig.colorbar(contour, ax=ax, label="Mach")

    ax.set_aspect('equal')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()


 
def run_mach_interpolation_case(
        bundle_path0, bundle_path1, mesh_path,
        output_dir="outputs",
        gamma_sb=0.002,
        gamma_sb_schedule=None,
        num_iter=2000,
        sensor_p=2,
        n_frames=11,
        anderson_m=5, ref_bundle_paths=None,
        density_mode="mask", density_pct=92.0,
        density_decay=0.05, tv_weight=1.0, density_target="unit",
        isocurve_order=1, isocurve_sigma=0.0,
        iso_n_contours=6, iso_ducros_pct=50.0,
        hessian_lp=2.0, hessian_mode="det",
        field_source="pert_mach", field_floor_pct=50.0,
        smooth_heat=True, smooth_gamma=0.1, smooth_t=0.005,
        ipfp_cfl=0.8,
        mach0_inlet=None, mach1_inlet=None,
        compute_w2=True, wass_gamma=0.005, wass_iter=50,
        reference_drift="null", drift_sigma_w=0.2,
        drift_gamma_gas=1.4, drift_cfl_adv=0.5):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'─'*55}\n  Mach SB-CDI Interpolation\n{'─'*55}")
 
    # ── [1] mesh + data ───────────────────────────────────────────────────────
    print("  [1/6] Loading mesh …")
    mesh    = Mesh()
    mesh.load_mesh(mesh_path)
    N_cells = int(mesh.tris.shape[0])
    print(f"    {N_cells} cells")
    # Boundary faces must be consistently marked in both conventions
    nbr_boundary = np.asarray(mesh.neighbors).min(axis=1) < 0   # has at least one -1 neighbour
    marker_boundary = np.any(np.asarray(mesh.face_markers)[np.asarray(mesh.face_connectivity)] > 0, axis=1)
    assert np.array_equal(nbr_boundary, marker_boundary), (
        "Mesh inconsistency: face_markers and neighbors disagree on which cells are boundary cells.")
 
    print("  [2/6] Loading Mach bundles …")
    data0   = np.load(bundle_path0);  data1 = np.load(bundle_path1)
    mach0   = data0["mach"].astype(float)
    mach1   = data1["mach"].astype(float)
    prims0  = data0["primitives"].astype(float)
    prims1  = data1["primitives"].astype(float)
    data0.close();  data1.close()
 
    plot_solution(mesh, mach0, labels=r"$M$", title="Source Mach  M₀",
                  filename=os.path.join(output_dir, "Mach_M0_raw.png"), cmap="viridis")
    plot_solution(mesh, mach1, labels=r"$M$", title="Target Mach  M₁",
                  filename=os.path.join(output_dir, "Mach_M1_raw.png"), cmap="viridis")
    # NEW: Plot exactly 6 overlaid contour lines for the raw fields
    plot_mach_contours(mesh, mach0, title="Source Mach Contours (M₀)", 
                       filename=os.path.join(output_dir, "Mach_M0_contours.png"), levels=6)
    plot_mach_contours(mesh, mach1, title="Target Mach Contours (M₁)", 
                       filename=os.path.join(output_dir, "Mach_M1_contours.png"), levels=6)
 
    # ── [3] n_steps ───────────────────────────────────────────────────────────
    n_steps        = heat_solver.compute_n_steps(mesh, gamma_sb, CFL=ipfp_cfl)
    n_steps_smooth = heat_solver.compute_n_steps(mesh, smooth_gamma, t_target=smooth_t, CFL=ipfp_cfl)
    sigma_smooth   = np.sqrt(2 * smooth_gamma * smooth_t)
    print(f"    FVM steps / IPFP solve:    {n_steps}")
    print(f"    FVM steps / density smooth: {n_steps_smooth}  "
          f"(γ={smooth_gamma}, t={smooth_t}, σ={sigma_smooth:.4f}, "
          f"apply={smooth_heat})")
 
    import math
    # ρ = exp(-π²γ/L²) is valid only for a plain rectangle.  For domains with
    # obstacles (diamond body) the effective spectral gap is determined by the
    # heat-diffusion path *around* the obstacle, which can be much longer than L.
    # Rule of thumb: γ_sb must satisfy sqrt(2γ) > obstacle_size to keep the
    # kernel connected across the two halves of the domain.
    L = float(np.asarray(mesh.points[:,0]).ptp())
    sigma_sb = math.sqrt(2.0 * gamma_sb)
    rho_theory = math.exp(-math.pi**2 * gamma_sb / L**2)
    iters_plain = int(math.ceil(math.log(1e-4/0.35)/math.log(rho_theory)))
    print(f"    γ_sb={gamma_sb}  σ={sigma_sb:.3f}  "
          f"rect. ρ≈{rho_theory:.4f}/iter → plain ~{iters_plain} iters to 1e-4 "
          f"→ Anderson({anderson_m}) ~{iters_plain//4}")
 
    # ── [4] Ducros → shock densities ─────────────────────────────────────────
    print("  [3/6] Computing Ducros sensors …")
    print("    M₀ …")
    mu0, sensor0, smoothed0 = transform_to_shock_density(
        mesh, prims0, n_steps_smooth, p=sensor_p,
        smooth_gamma=smooth_gamma, smooth_t=smooth_t, apply_heat=smooth_heat,
        density_mode=density_mode, density_pct=density_pct,
        density_decay=density_decay, tv_weight=tv_weight,
        density_target=density_target,
        isocurve_order=isocurve_order, isocurve_width=isocurve_sigma,
        iso_n_contours=iso_n_contours, iso_ducros_pct=iso_ducros_pct,
        hessian_lp=hessian_lp, hessian_mode=hessian_mode,
        field_source=field_source, field_floor_pct=field_floor_pct,
        mach_inlet=mach0_inlet)
    print("    M₁ …")
    mu1, sensor1, smoothed1 = transform_to_shock_density(
        mesh, prims1, n_steps_smooth, p=sensor_p,
        smooth_gamma=smooth_gamma, smooth_t=smooth_t, apply_heat=smooth_heat,
        density_mode=density_mode, density_pct=density_pct,
        density_decay=density_decay, tv_weight=tv_weight,
        density_target=density_target,
        isocurve_order=isocurve_order, isocurve_width=isocurve_sigma,
        iso_n_contours=iso_n_contours, iso_ducros_pct=iso_ducros_pct,
        hessian_lp=hessian_lp, hessian_mode=hessian_mode,
        field_source=field_source, field_floor_pct=field_floor_pct,
        mach_inlet=mach1_inlet)

    if density_mode in ("iso", "iso_hessian", "field"):
        feat_name, feat_title = "Perturbation", "Mach perturbation"
    else:
        feat_name, feat_title = "Ducros", "Ducros sensor"

    for field, fname, title in [
        (sensor0, f"{feat_name}_M0.png",  f"{feat_title}  M₀"),
        (sensor1, f"{feat_name}_M1.png",  f"{feat_title}  M₁"),
        (np.asarray(mu0), "Density_mu0.png", "Shock density  μ₀"),
        (np.asarray(mu1), "Density_mu1.png", "Shock density  μ₁"),
    ]:
        plot_solution(mesh, field, labels=r"$\varphi$", title=title,
                      filename=os.path.join(output_dir, fname), cmap="hot")
 
    # ── [4b] Reference drift β (conditions the bridge; referrence_drift.tex) ───
    use_drift = reference_drift not in (None, "null", "none", "off")
    beta_ipfp = None
    if use_drift:
        if mach0_inlet is None or mach1_inlet is None:
            raise ValueError("reference_drift='oblique' requires mach0_inlet/mach1_inlet")
        # The IPFP kernel Q₁/Q†₁ is a single unit-time solve → one representative
        # β field.  Use the midpoint drift β(t=0.5) (ω is ~constant for linear M(t)).
        beta_np = drift_mod.build_reference_drift(
            mesh, reference_drift, [0.5],
            float(mach0_inlet), float(mach1_inlet),
            gamma_g=drift_gamma_gas, sigma_w=drift_sigma_w)[0]
        beta_ipfp = jnp.asarray(beta_np)
        beta_max  = float(np.max(np.linalg.norm(beta_np, axis=1)))
        # #1 risk check: the operative number is the face Péclet
        # Pe_f = |β·n|_f · d_ij / γ  (not the area/perimeter dx metric, which
        # understates it ~4×).  First-order upwind is positive/monotone at any
        # Pe, but Pe>1 quantifies the anisotropic numerical diffusion (~|β·n|d/2)
        # the linear kernel adds.
        _nb    = np.asarray(mesh.neighbors)
        _fc    = np.asarray(mesh.face_connectivity)
        _bary  = np.asarray(mesh.barycenter)
        _nrm   = np.asarray(mesh.normals)                     # (N,3,2)
        _isb   = np.asarray(mesh.face_markers)[_fc] > 0
        _bi    = np.repeat(beta_np[:, None, :], 3, axis=1)
        _bj    = np.where(_isb[..., None], _bi, beta_np[np.where(_nb >= 0, _nb, 0)])
        _bn    = np.sum(0.5 * (_bi + _bj) * _nrm, axis=-1)    # β·n per face
        _dij   = np.linalg.norm(_bary[np.where(_nb >= 0, _nb, 0)] - _bary[:, None, :], axis=-1)
        _pe    = np.where(_isb, 0.0, np.abs(_bn) * _dij / max(gamma_sb, 1e-30))
        pe_max = float(_pe.max())
        print(f"    reference_drift='{reference_drift}'  |β|_max={beta_max:.4f}  "
              f"σ_w={drift_sigma_w}")
        print(f"    face Péclet max = {pe_max:.2f}  (γ_sb={gamma_sb:.2e}; upwind "
              f"is monotone at any Pe, added num.diff ~|β·n|d/2)")

    # ── [5] Anderson IPFP with γ-annealing (ε-scaling) + GPU timing ──────────
    EPS_LOG = 1e-12
    log_mu0 = jnp.log(jnp.maximum(mu0, EPS_LOG))
    log_mu1 = jnp.log(jnp.maximum(mu1, EPS_LOG))
    schedule = list(gamma_sb_schedule) if gamma_sb_schedule else [gamma_sb]
    dev = jax.devices()[0]
    tag = f"drifted ({reference_drift})" if use_drift else "heat"
    print(f"  [4/6] Anderson IPFP [{tag}] — γ-annealing {schedule}  (device {dev.platform.upper()})")

    def _n_steps_ipfp(gk):
        if use_drift:
            return advdiff_solver.compute_n_steps_advdiff(
                mesh, gk, beta_max, CFL_diff=ipfp_cfl, CFL_adv=drift_cfl_adv)
        return heat_solver.compute_n_steps(mesh, gk, CFL=ipfp_cfl)

    f = g = None
    ipfp_residuals, stage_bounds = [], []
    t_total0 = _time.perf_counter()
    for k, gk in enumerate(schedule):
        n_steps_k = _n_steps_ipfp(gk)
        t0 = _time.perf_counter()
        if use_drift:
            f, g, res_k = apply_IPFP_2d_anderson_drift(
                log_mu0, log_mu1, gk, mesh, beta_ipfp, n_steps=n_steps_k,
                num_iter=num_iter, tol=1e-7, m=anderson_m, print_every=50,
                f_init=f, g_init=g)
        else:
            f, g, res_k = apply_IPFP_2d_anderson(
                log_mu0, log_mu1, gk, mesh, n_steps=n_steps_k,
                num_iter=num_iter, tol=1e-7, m=anderson_m, print_every=50,
                f_init=f, g_init=g)
        f.block_until_ready();  g.block_until_ready()      # GPU is async → sync here
        dt = _time.perf_counter() - t0
        ipfp_residuals.extend(res_k);  stage_bounds.append(len(ipfp_residuals))
        print(f"    stage {k+1}/{len(schedule)}  γ={gk:.4g}  n_steps={n_steps_k}  "
              f"{len(res_k)} iters  {dt:.3f} s  ({1e3*dt/max(len(res_k),1):.1f} ms/iter)")
    t_total = _time.perf_counter() - t_total0
    gamma_sb = schedule[-1]
    n_steps  = _n_steps_ipfp(gamma_sb)
    print(f"    IPFP total time: {t_total:.3f} s  ({len(ipfp_residuals)} iters, "
          f"final γ={gamma_sb:.4g})")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(ipfp_residuals)
    ax.axhline(1e-7, color="red", ls="--", label="tol=1e-7")
    for b, gk in zip(stage_bounds[:-1], schedule[:-1]):
        ax.axvline(b, color="grey", ls=":", alpha=0.7)
    ax.set_xlabel("IPFP iteration (concatenated over γ stages)")
    ax.set_ylabel("residual ‖Δf‖∞")
    ax.set_title(f"Anderson(m={anderson_m}) γ-annealing {schedule} — "
                 f"{t_total:.2f}s on {dev.platform.upper()}")
    ax.legend();  ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "IPFP_convergence.png"),
                dpi=200, bbox_inches="tight");  plt.close()
 
    # ── [6] Transport maps + diagnostic drift sequence ───────────────────────
    print("  [5/6] SB barycentric transport maps (global, boundary-corrected) …")
    barycenters         = np.asarray(mesh.barycenter)
    barycenter_delaunay = Delaunay(barycenters)   # computed ONCE

    # Shock masks from the continuous density (top 0.4 % by smoothed weight)
    mask0_np = smoothed0 >= np.quantile(smoothed0, 0.996)
    mask1_np = smoothed1 >= np.quantile(smoothed1, 0.996)

    # Density sequence (diagnostic)
    t_array  = np.linspace(0.0, 1.0, n_frames)
    area_jx  = jnp.asarray(mesh.area)
    if use_drift:
        rho_vmap = jax.jit(jax.vmap(
            lambda tv: retrieve_rho_2d_drift(f, g, tv, gamma_sb, mesh, beta_ipfp, n_steps)))
    else:
        rho_vmap = jax.jit(jax.vmap(
            lambda tv: retrieve_rho_2d(f, g, tv, gamma_sb, mesh, n_steps)))
    rho_seq_jx = rho_vmap(jnp.array(t_array))
    masses     = jnp.sum(rho_seq_jx * area_jx[None, :], axis=1, keepdims=True)
    rho_seq    = np.asarray(rho_seq_jx / jnp.maximum(masses, 1e-14))

    if use_drift:
        drift_seq = np.stack([
            np.asarray(retrieve_b_2d_drift(g, jnp.array(float(t)), gamma_sb, mesh,
                                           beta_ipfp, n_steps))
            for t in t_array], axis=0)
    else:
        drift_seq = np.stack([
            np.asarray(retrieve_b_2d(g, jnp.array(float(t)), gamma_sb, mesh, n_steps))
            for t in t_array], axis=0)

    # Exact point-in-body predicate, shared by the reflected flow map (leak-bias
    # reference) and the CDI reconstruction's fallback gate.
    inside_body = drift_mod.make_inside_body(mesh)

    if use_drift:
        # Deterministic REFLECTED flows of ±β over unit time (same no-flux
        # boundary + body-retraction the FVM kernel itself uses): the leak bias
        # is measured against these (not the identity, not the unbounded
        # analytic flow) so β's transport stays in T/S without picking up a
        # phantom displacement at the outflow edge or the wedge's shock feet.
        # Uses the SAME frozen beta_np the kernel advects with (beta_ipfp),
        # not a re-evaluated assemble_drift(t).
        phi_fwd = jnp.asarray(drift_mod.flow_map(barycenters, beta_np, sign=+1.0,
                                                  inside_body=inside_body))
        phi_bwd = jnp.asarray(drift_mod.flow_map(barycenters, beta_np, sign=-1.0,
                                                  inside_body=inside_body))
        T_jx, S_jx, den_T_jx, den_S_jx, bias_T_jx, bias_S_jx = \
            compute_sb_barycentric_maps_drift(f, g, gamma_sb, mesh, beta_ipfp,
                                              phi_fwd, phi_bwd, n_steps)
    else:
        T_jx, S_jx, den_T_jx, den_S_jx, bias_T_jx, bias_S_jx = \
            compute_sb_barycentric_maps(f, g, gamma_sb, mesh, n_steps)
    T_map = np.asarray(T_jx);  S_map = np.asarray(S_jx)

    # ── MAP DIAGNOSTICS (read these to locate the T_map/S_map blow-up) ────────
    den_T = np.asarray(den_T_jx);  den_S = np.asarray(den_S_jx)
    bias_T = np.asarray(bias_T_jx);  bias_S = np.asarray(bias_S_jx)
    T_raw = T_map + bias_T                     # undo the bias subtraction
    S_raw = S_map + bias_S
    def _frac_below(d, rel):  return float((d < rel*float(d.max())).mean())
    print("    ── barycentric-map diagnostics ──")
    print(f"      den_T  min/max = {den_T.min():.2e}/{den_T.max():.2e}   "
          f"frac<1e-6·max = {_frac_below(den_T,1e-6):.3f}  "
          f"frac<1e-12·max = {_frac_below(den_T,1e-12):.3f}")
    print(f"      den_S  min/max = {den_S.min():.2e}/{den_S.max():.2e}   "
          f"frac<1e-6·max = {_frac_below(den_S,1e-6):.3f}  "
          f"frac<1e-12·max = {_frac_below(den_S,1e-12):.3f}")
    _bmT = np.linalg.norm(bias_T, axis=1)
    _bmS = np.linalg.norm(bias_S, axis=1)
    print(f"      |bias_T| max = {_bmT.max():.3f}   |bias_T|[mask0] mean = {_bmT[mask0_np].mean():.3f}")
    print(f"      |bias_S| max = {_bmS.max():.3f}   |bias_S|[mask1] mean = {_bmS[mask1_np].mean():.3f}")
    _draw = np.linalg.norm((T_raw - barycenters)[mask0_np], axis=1).mean()
    _dfin = np.linalg.norm((T_map - barycenters)[mask0_np], axis=1).mean()
    print(f"      |T_raw−x|[mask0] mean = {_draw:.3f}   |T−x|[mask0] mean = {_dfin:.3f}   "
          f"(if T_raw small but T big → bias is the culprit)")
    _sraw = np.linalg.norm((S_raw - barycenters)[mask1_np], axis=1).mean()
    _sfin = np.linalg.norm((S_map - barycenters)[mask1_np], axis=1).mean()
    print(f"      |S_raw−x|[mask1] mean = {_sraw:.3f}   |S−x|[mask1] mean = {_sfin:.3f}   "
          f"(if S_raw small but S big → bias is the culprit)")
    print(f"      den_T[mask0] min/max = {den_T[mask0_np].min():.2e}/{den_T[mask0_np].max():.2e}   "
          f"(if min≈1e-30 → forward denominator underflow on shock cells)")
    print(f"      den_S[mask1] min/max = {den_S[mask1_np].min():.2e}/{den_S[mask1_np].max():.2e}   "
          f"(if min≈1e-30 → backward denominator underflow on shock cells)")
    # where do the worst forward/backward cells map TO?
    _worstT = np.argsort(np.linalg.norm(T_map - barycenters, axis=1))[-5:]
    for _i in _worstT:
        print(f"      worst T: x={barycenters[_i].round(3)} -> T={T_map[_i].round(3)}  "
              f"den_T={den_T[_i]:.2e}")
    _worstS = np.argsort(np.linalg.norm(S_map - barycenters, axis=1))[-5:]
    for _i in _worstS:
        print(f"      worst S: x={barycenters[_i].round(3)} -> S={S_map[_i].round(3)}  "
              f"den_S={den_S[_i]:.2e}")
    print("    ─────────────────────────────────")

    disp_fwd = T_map[mask0_np] - barycenters[mask0_np]
    disp_bwd = S_map[mask1_np] - barycenters[mask1_np]
    md_f = disp_fwd.mean(0);  md_b = disp_bwd.mean(0)
    print(f"    Mean μ₀→μ₁ displacement  (T at M₀ shock): ({md_f[0]:+.4f}, {md_f[1]:+.4f})")
    print(f"    Mean μ₁→μ₀ displacement  (S at M₁ shock): ({md_b[0]:+.4f}, {md_b[1]:+.4f})")
    print(f"    Sanity (should approx cancel):           ({md_f[0]+md_b[0]:+.4f}, {md_f[1]+md_b[1]:+.4f})")

    # ── CDI reconstruction ────────────────────────────────────────────────────
    print("  Reconstructing CDI Mach frames …")
    t_interp0 = _time.perf_counter()
    interp0, interp1, inside_body_cdi, domain_lo_cdi, domain_hi_cdi = \
        build_cdi_interpolators(barycenters, mach0, mach1, barycenter_delaunay, mesh)
    mach_bcdi = []   # PRIMARY: barycentric SB-CDI (gradient-free, exact endpoints)
    mach_lin  = []   # baseline: linear
    for t_val in t_array:
        print(f"    t = {t_val:.2f} …", end="\r")
        mach_bcdi.append(reconstruct_mach_barycentric_cdi(
            float(t_val), T_map, S_map, barycenters, mach0, mach1,
            interp0, interp1, inside_body_cdi, domain_lo_cdi, domain_hi_cdi))
        mach_lin.append((1.0-t_val)*mach0 + t_val*mach1)
    mach_bcdi = np.stack(mach_bcdi)
    mach_lin  = np.stack(mach_lin)
    t_interp = _time.perf_counter() - t_interp0
    print()
    print(f"    Interpolation total time: {t_interp:.3f} s  "
          f"({len(t_array)} frames, {1e3*t_interp/max(len(t_array),1):.1f} ms/frame)")

    # ── Error vs. high-resolution reference ───────────────────────────────────
    if ref_bundle_paths and len(ref_bundle_paths) == 9:
        print("  Computing interpolation error vs. reference solutions …")
        compute_interpolation_error(
            t_array, mach_bcdi, mach_lin,
            ref_bundle_paths, mesh, output_dir,
            wass_gamma=wass_gamma, wass_iter=wass_iter,
            compute_w2=compute_w2)
    else:
        print("  (no ref_bundle_paths provided — skipping ground-truth error)")

    # ── [7] Plots ─────────────────────────────────────────────────────────────
    print("  [6/6] Generating plots …")
    entropies = compute_bridge_entropy(mu1, rho_seq, area_jx)

    plot_density_transport(mesh, rho_seq, t_array,
                           os.path.join(output_dir, "SB_Mach_density"))
    plot_drift_field(mesh, drift_seq, t_array,
                     os.path.join(output_dir, "SB_Mach_drift"), time_index=5)
    plot_entropy(os.path.join(output_dir, "SB_Mach_entropy"), t_array, entropies,
                 title="Entropy along SB")

    for i, t_val in enumerate(t_array):
        plot_solution(mesh, mach_lin[i], labels=r"$M$",
                      title=f"Linear  t={t_val:.2f}",
                      filename=os.path.join(output_dir, f"Mach_Linear_t{t_val:.2f}.png"),
                      cmap="viridis")
        plot_solution(mesh, mach_bcdi[i], labels=r"$M$",
                      title=f"SB-CDI (barycentric)  t={t_val:.2f}",
                      filename=os.path.join(output_dir, f"Mach_BaryCDI_t{t_val:.2f}.png"),
                      cmap="viridis")

    print(f"  Done → {output_dir}/")