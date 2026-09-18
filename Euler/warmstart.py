"""Seed the Euler solver from an interpolated snapshot instead of freestream.

`initialize()` in main.py builds a uniform freestream state.  This module builds
the same object -- a conserved `(N_cells, 4)` array `[rho, rho u, rho v, E]` --
from an SB/FFD/Linear interpolated field, so the solver starts closer to the
steady state and (the thing being measured) reaches it in fewer iterations.

Two input shapes are accepted:

  * ``warmstart.npz`` written by SB/utils.py -- carries `conservatives` of shape
    `(n_methods, n_t, N, 4)`, so `method` and `t` are both required.
  * any Euler bundle ``.npz`` -- uses its `conservatives` directly.  This is what
    makes the endpoint unit test possible: seeding with the M2.00 bundle at
    M=2.00 must converge immediately, which exercises the loader, the dtype
    cast and the stop path with zero SB involvement.

Every guard here is a hard failure.  A seed that silently disagrees with the
mesh, the target condition or the solver dtype does not produce a wrong answer --
it produces a plausible-looking timing number, which is worse.
"""
from __future__ import annotations

import numpy as np

# Cell ordering in a bundle is the mesh file's ordering, so index-for-index reuse
# is valid only if the barycenters agree.  Bundles store float32, hence a
# tolerance well above f32 eps but far below any real mesh difference.
_BARY_TOL = 1e-5


def _pick_frame(data, method, t):
    """Select one (N,4) state out of a warmstart.npz `(n_m, n_t, N, 4)` block."""
    names = [str(s) for s in np.asarray(data["method_names"]).ravel()]
    if method is None:
        raise ValueError(
            f"--init-method is required for a warmstart.npz (available: {names})")
    if method not in names:
        raise ValueError(f"init method {method!r} not in {names}")
    if t is None:
        raise ValueError("--init-t is required for a warmstart.npz")

    t_arr = np.asarray(data["t"], dtype=float).ravel()
    j = int(np.argmin(np.abs(t_arr - float(t))))
    if abs(t_arr[j] - float(t)) > 1e-9:
        raise ValueError(
            f"--init-t {t} does not match any stored frame: {t_arr.tolist()}")
    i = names.index(method)

    meta = {"init_method": method, "init_t": float(t_arr[j])}
    for key, out in (("mach_target", "init_mach_target"),
                     ("aoa_target", "init_aoa_target")):
        if key in data.files:
            meta[out] = float(np.asarray(data[key], dtype=float).ravel()[j])
    return np.asarray(data["conservatives"])[i, j], meta


def load_init_state(path, mesh, cfg, *, method=None, t=None,
                    dtype=None, parent=None, allow_repair=False):
    """Return `(W, meta)` -- a conserved state on `mesh`, ready to hand to run().

    `dtype` must be the dtype `initialize()` produces (float32 unless
    JAX_ENABLE_X64).  A float64 seed would double every buffer and trigger a
    separate XLA compilation, so the warm-started run would be slower for a
    reason that has nothing to do with the seed's quality -- which would silently
    invert the very comparison this exists to make.
    """
    data = np.load(path, allow_pickle=False)
    try:
        if "method_names" in data.files:
            W, meta = _pick_frame(data, method, t)
            meta["init_kind"] = "warmstart"
        else:
            if "conservatives" not in data.files:
                raise ValueError(f"{path}: no `conservatives` array")
            W, meta = np.asarray(data["conservatives"]), {"init_kind": "bundle"}
            if "mach_in" in data.files:
                meta["init_mach_target"] = float(
                    np.asarray(data["mach_in"], dtype=float).ravel()[0])
            if "aoa_in" in data.files:
                meta["init_aoa_target"] = float(
                    np.asarray(data["aoa_in"], dtype=float).ravel()[0])
        bary_file = np.asarray(data["barycenter"], dtype=float) \
            if "barycenter" in data.files else (
                np.asarray(data["node_pos"], dtype=float)
                if "node_pos" in data.files else None)
    finally:
        data.close()

    W = np.asarray(W, dtype=float)
    meta["init_source"] = str(path)

    # ── the seed must describe THIS mesh, index for index ────────────────────
    bary = np.asarray(mesh.barycenter, dtype=float)
    n_target = int(bary.shape[0])
    if bary_file is not None:
        if bary_file.shape != bary.shape:
            raise ValueError(
                f"{path}: barycenter shape {bary_file.shape} != mesh {bary.shape}")
        dmax = float(np.abs(bary_file - bary).max())
        if dmax > _BARY_TOL:
            raise ValueError(
                f"{path}: barycenters differ from the mesh by {dmax:.3e} "
                f"(> {_BARY_TOL:.0e}) -- wrong mesh, or a different cell ordering")
    if W.shape[0] != n_target:
        raise ValueError(f"{path}: {W.shape[0]} cells != mesh {n_target}")
    if W.shape[-1] != 4:
        raise ValueError(f"{path}: expected a (N,4) conserved state, got {W.shape}")

    # ── the seed must describe THIS flow condition ───────────────────────────
    for key, cfg_key, label in (("init_mach_target", "Mach", "Mach"),
                                ("init_aoa_target", "aoa", "AoA")):
        if key in meta and cfg.get(cfg_key) is not None:
            if abs(meta[key] - float(cfg[cfg_key])) > 1e-6:
                raise ValueError(
                    f"{path}: seed {label}={meta[key]:.6f} != run {label}="
                    f"{float(cfg[cfg_key]):.6f}. The seed and the boundary "
                    f"condition would disagree.")

    # ── admissibility, before the solver can turn it into NaNs ───────────────
    gamma = float(cfg.get("gamma", 1.4))
    rho = W[:, 0]
    kin = 0.5 * (W[:, 1] ** 2 + W[:, 2] ** 2) / np.where(rho > 0, rho, np.nan)
    p = (W[:, 3] - kin) * (gamma - 1.0)
    n_bad_rho = int(np.count_nonzero(~(rho > 0)))
    n_bad_p = int(np.count_nonzero(~(p > 0)))
    meta["init_repair_count"] = n_bad_rho + n_bad_p
    meta["init_rho_min"] = float(np.nanmin(rho))
    meta["init_p_min"] = float(np.nanmin(p))
    if not np.all(np.isfinite(W)):
        raise ValueError(f"{path}: seed contains non-finite values")
    if (n_bad_rho or n_bad_p) and not allow_repair:
        raise ValueError(
            f"{path}: inadmissible seed -- {n_bad_rho} cells with rho<=0, "
            f"{n_bad_p} with p<=0 (rho_min={meta['init_rho_min']:.3e}, "
            f"p_min={meta['init_p_min']:.3e}). Pass --init-allow-repair to "
            f"clamp and continue, but the timing it produces is not clean.")

    # ── nested solver mesh: the seed lives on the PARENT, scatter to children ─
    if parent is not None:
        parent = np.asarray(parent)
        if int(parent.max()) != W.shape[0] - 1:
            raise ValueError(
                f"{path}: parent map expects {int(parent.max()) + 1} coarse cells, "
                f"seed has {W.shape[0]}")
        W = W[parent]                       # adjoint of utils.average_to_parent
        meta["init_scattered_to_solver_mesh"] = True

    if dtype is not None:
        W = W.astype(dtype)
    return W, meta
