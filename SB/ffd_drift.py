"""
ffd_drift.py — data-driven reference drift from FFD registration.

Builds beta_cells (N,2) for the drifted Schrödinger bridge by registering the
two smoothed shock-feature marginals with the phdtruel piecewise-FFD framework
on a FICTITIOUS structured grid, then evaluating the resulting (analytic,
Bernstein) forward mapping directly at the unstructured cell centres.

Design constraints honoured:
  * zero changes to the SB solver: output is a frozen (N,2) float64 array,
    consumed by the existing Q/Q†, flow_map, Φ-referenced bias, trust gate,
    CDI and validate checks exactly like the oblique drift;
  * single FFD region, NO boundary-condition machinery (all control points
    free: masks = all-NaN, boundary_conditions=None), per "try without
    focusing on the BCs".  The SB pipeline's own safeguards (kernel boundary
    masking, flow_map body-retraction, trust gate) still apply downstream;
  * the structured grid is ONLY the cost-sampling domain of the registration.
    beta is evaluated analytically at the true cell centres — there is no
    grid→mesh interpolation step and hence no second interpolation error.

Mapping semantics (from cost_fn._compute_alignment):
    cost drives  u0(W(x)) ≈ u1(x)   and   u1(T(x)) ≈ u0(x)
  ⇒ at a field-0 feature p0:  u1(T(p0)) = u0(p0)  ⇒  T(p0) = p1.
  T is the FORWARD point motion (field-0 features → field-1 positions),
  W ≈ T⁻¹ via the alpha_B_bij synchronisation term.  Therefore

      beta(x) = T(x, s=1) − x            (frozen, unit pseudo-time)

  and flow_map(+beta) must carry the μ0 ridge onto the μ1 ridge — the
  pipeline's shock-motion check ([7]) is the acceptance test.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter

# ── phdtruel FFD framework ───────────────────────────────────────────────────
import jax
import jax.numpy as jnp

from phdtruel.mappings.cost_functional.ffd.factory import (
    make_piecewise_ffd_cost_function,
)
from phdtruel.mappings.cost_functional.types import MeshJaxed
from phdtruel.mappings.mappings import identity_mapping

try:  # preferred: the framework's own solver
    from phdtruel.mappings.functional_solver import solve as _phdtruel_solve
    _HAVE_SOLVER = True
except Exception:  # pragma: no cover — fallback optimiser below
    _HAVE_SOLVER = False


# ─────────────────────────────────────────────────────────────────────────────
#  Defaults (overridable per-key from Config [drift.ffd])
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CFG = dict(
    grid_n=144,                # fictitious grid: 144² ≈ 20.7k ≈ n_cells
    n_cp=(10, 10),             # Bernstein control lattice (degree = n_cp − 1)
    alpha_algn=1.0,            # field alignment
    alpha_bij=0.1,             # W/T synchronisation  (W ≈ T⁻¹)
    alpha_jac=1.0,             # Jacobian barrier     (det ∇Φ > 0)
    alpha_cp_norm=1.0e-2,      # control-point regularisation
    alpha_w2=0.1,              # weighted-moment W2 term: global capture range
    barrier_epsilon=0.1,
    barrier_fn="logarithmic",
    smooth_extra_cells=2.0,    # extra Gaussian smoothing of the rasterised
                               # marginals, in GRID cells (0 = none)
    n_iters=800,               # fallback-optimiser iterations
    lr=1.0e-2,                 # fallback-optimiser learning rate
    seed=0,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Structured proxy grid + rasterisation
# ─────────────────────────────────────────────────────────────────────────────
def _structured_mesh(lo, hi, n):
    """MeshJaxed on [lo,hi]² with n×n nodes (meshgrid 'ij', raveled C-order —
    the layout _interpolate_field_piecewise reshapes back with mesh.axes)."""
    x = jnp.linspace(float(lo[0]), float(hi[0]), int(n))
    y = jnp.linspace(float(lo[1]), float(hi[1]), int(n))
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    pts = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    return MeshJaxed(points=pts, axes=(x, y))


def _rasterize(barycenters, values, mesh_jx, fill=0.0):
    """Unstructured cell field → structured grid values (nx*ny,), C-order.

    Linear interpolation from cell centres; anything outside the convex hull
    or inside the body hole (no fluid cells there) gets `fill` (=0: the shock
    sensor is naturally zero in the freestream and undefined-in-body)."""
    x = np.asarray(mesh_jx.axes[0]);  y = np.asarray(mesh_jx.axes[1])
    X, Y = np.meshgrid(x, y, indexing="ij")
    interp = LinearNDInterpolator(np.asarray(barycenters, float),
                                  np.asarray(values, float), fill_value=np.nan)
    F = interp(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    F = np.nan_to_num(F, nan=float(fill))
    return F  # (nx, ny)


def _prep_marginal(barycenters, mu, mesh_jx, smooth_extra_cells):
    """Rasterise, optionally re-smooth (grid-cell units), normalise to max=1.

    Max-normalisation (not mass) keeps the SSD-type alignment focused on ridge
    POSITION rather than amplitude mismatch between the two marginals."""
    F = _rasterize(barycenters, mu, mesh_jx, fill=0.0)
    if smooth_extra_cells and smooth_extra_cells > 0.0:
        F = gaussian_filter(F, sigma=float(smooth_extra_cells))
    m = float(F.max())
    return F / max(m, 1e-30)


# ─────────────────────────────────────────────────────────────────────────────
#  Registration driver (single region, all control points free)
# ─────────────────────────────────────────────────────────────────────────────
def _adam_solve(cost_fn, init_params, n_iters, lr, seed=0):
    """Minimal fallback optimiser if phdtruel.mappings.functional_solver is
    absent: Adam with cosine decay on the traced parameters (a pytree)."""
    import optax
    sched = optax.cosine_decay_schedule(float(lr), int(n_iters))
    opt = optax.adam(sched)
    params = init_params
    state = opt.init(params)
    val_grad = jax.jit(jax.value_and_grad(cost_fn))

    @jax.jit
    def step(params, state):
        v, g = val_grad(params)
        updates, state = opt.update(g, state, params)
        return optax.apply_updates(params, updates), state, v

    v0 = None
    for k in range(int(n_iters)):
        params, state, v = step(params, state)
        if v0 is None:
            v0 = float(v)
    print(f"    [ffd] fallback Adam: cost {v0:.4e} -> {float(v):.4e} "
          f"in {n_iters} iters")
    return params


def register_pair(mu0_grid, mu1_grid, mesh_jx, cfg):
    """Run the two-map FFD registration; return (mapping_T, mapping_W, info).

    T: forward point motion (use for beta);  W ≈ T⁻¹ (diagnostic)."""
    ncx, ncy = (int(cfg["n_cp"][0]), int(cfg["n_cp"][1]))
    free_mask = jnp.full((ncx, ncy, 2), jnp.nan)      # NaN = free DOF (no BCs)

    result = make_piecewise_ffd_cost_function(
        region_meshes=[mesh_jx],
        all_n_cp=[(ncx, ncy)],
        all_imposed_displacements_masks=[free_mask],
        initial_mapping_function_w=identity_mapping,
        initial_mapping_function_t=identity_mapping,
        alpha_A_algn=float(cfg["alpha_algn"]),
        alpha_B_bij=float(cfg["alpha_bij"]),
        alpha_C_jac=float(cfg["alpha_jac"]),
        alpha_D_cp_norm=float(cfg["alpha_cp_norm"]),
        barrier_epsilon=float(cfg["barrier_epsilon"]),
        u0_values=jnp.asarray(np.asarray(mu0_grid, float).ravel()),
        u1_values=jnp.asarray(np.asarray(mu1_grid, float).ravel()),
        alpha_G_got_w2=float(cfg["alpha_w2"]),
        initialisation_noise_factor=0.0,
        boundary_conditions=None,                     # ← no BC machinery
        barrier_fn_name=str(cfg["barrier_fn"]),
    )

    if _HAVE_SOLVER:
        solved = _phdtruel_solve(result.cost_fn, result.init_params)
        params = getattr(solved, "params", solved)
    else:
        params = _adam_solve(result.cost_fn, result.init_params,
                             cfg["n_iters"], cfg["lr"], cfg["seed"])

    mapping_w, mapping_t = result.make_mappings(params)

    # diagnostics on the grid: inverse consistency + final cost
    pts = np.asarray(mesh_jx.points)
    wt = np.asarray(mapping_w(np.asarray(mapping_t(pts, s=1.0)), s=1.0))
    inv_err = float(np.linalg.norm(wt - pts, axis=1).max())
    # cost_fn follows the solver's has_aux pattern → returns (scalar_cost, aux_dict)
    final_cost = result.cost_fn(params)
    if isinstance(final_cost, tuple):
        final_cost = final_cost[0]
    info = dict(final_cost=float(final_cost),
                inverse_consistency_max=inv_err)
    print(f"    [ffd] registration done: cost={info['final_cost']:.4e}, "
          f"max|W(T(x))−x|={inv_err:.3e}")
    return mapping_t, mapping_w, info


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point (the only function the SB pipeline calls)
# ─────────────────────────────────────────────────────────────────────────────
def build_ffd_beta(mesh, mu0_smooth, mu1_smooth, cfg=None, cache_dir=None,
                   tag=""):
    """FFD registration of the two marginals → (beta, alpha), both (N,2) float64.

    beta  = T(x,s=1) − x : forward displacement  (μ₀→μ₁) — the SB reference drift.
    alpha = W(x,s=1) − x : backward displacement (μ₁→μ₀), W ≈ T⁻¹.  Used to run
            the FFD registration as a stand-alone interpolator, so the sweep can
            separate "what the registration gives" from "what the SB adds".

    Parameters
    ----------
    mesh        : SB unstructured mesh (barycenter used).
    mu0_smooth, mu1_smooth : the SMOOTHED marginals feeding IPFP (cell arrays).
    cfg         : dict of overrides for DEFAULT_CFG (Config [drift.ffd]).
    cache_dir   : if given, cache beta to  <cache_dir>/ffd_beta_<hash>.npz
                  (the registration is an offline optimisation; run once per
                  Mach pair, deterministic thereafter).
    """
    c = dict(DEFAULT_CFG)
    if cfg:
        c.update({k: v for k, v in cfg.items() if v is not None})

    bary = np.asarray(mesh.barycenter, dtype=float)
    lo, hi = bary.min(axis=0), bary.max(axis=0)

    # cache key: config + coarse fingerprints of the two marginals
    key = json.dumps({**{k: (list(v) if isinstance(v, (tuple, list)) else v)
                         for k, v in c.items()},
                      "tag": tag,
                      "m0": float(np.asarray(mu0_smooth).sum()),
                      "m1": float(np.asarray(mu1_smooth).sum()),
                      "n": int(len(bary))}, sort_keys=True)
    h = hashlib.sha1(key.encode()).hexdigest()[:12]
    cache = (os.path.join(cache_dir, f"ffd_beta_{h}.npz")
             if cache_dir else None)
    if cache and os.path.exists(cache):
        _d = np.load(cache)
        if "alpha" in _d.files:                       # new-format cache
            print(f"    [ffd] loaded cached beta+alpha  ({cache})")
            return _d["beta"], _d["alpha"]
        print(f"    [ffd] cache lacks the backward map — recomputing ({cache})")

    mesh_jx = _structured_mesh(lo, hi, c["grid_n"])
    F0 = _prep_marginal(bary, mu0_smooth, mesh_jx, c["smooth_extra_cells"])
    F1 = _prep_marginal(bary, mu1_smooth, mesh_jx, c["smooth_extra_cells"])

    mapping_t, mapping_w, info = register_pair(F0, F1, mesh_jx, c)

    # beta at the TRUE cell centres, analytically:  T(x, s=1) − x
    T_cells = np.asarray(mapping_t(bary, s=1.0), dtype=float)
    beta = (T_cells - bary).astype(np.float64)

    # Backward displacement alpha = W(x, s=1) − x  (W ≈ T⁻¹ via the bijectivity
    # term).  Not needed for the SB drift, but it lets the caller use the FFD
    # registration DIRECTLY as an interpolator — the control that isolates what
    # the Schrödinger bridge adds on top of the raw registration map.
    S_cells = np.asarray(mapping_w(bary, s=1.0), dtype=float)
    alpha = (S_cells - bary).astype(np.float64)

    bmag = np.linalg.norm(beta, axis=1)
    print(f"    [ffd] |beta| max={bmag.max():.3f}  mean={bmag.mean():.4f}  "
          f"(grid {c['grid_n']}², lattice {c['n_cp']})")

    if cache:
        np.savez_compressed(cache, beta=beta, alpha=alpha, **info)
        print(f"    [ffd] cached -> {cache}")
    return beta, alpha