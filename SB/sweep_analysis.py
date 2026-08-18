"""
sweep_analysis.py — aggregate the γ_min × drift × hessian_mode sweep.

Reads every  <root>/<hmode>/drift_<mode>/gmin<γ>/metrics.json  written by
run_mach_interpolation_case (launched by run_sb_sweep.sbatch) and produces,
in <root>/_sweep/:

  err_vs_gamma_<h>.png    mean-t L2 / worst-t L∞ / mean-t W₂ of BaryCDI vs γ_min,
                          per drift, with the γ-independent Linear baseline
  err_vs_t_<h>.png        L2(t) per drift, curves colour-graded by γ_min
  convergence_<h>.png     final IPFP residual + iterations vs γ_min, and the
                          residual histories at the smallest γ (stall = null collapse)
  gain_heatmap_<h>.png    (drift × γ) heatmap of L2(BaryCDI)/L2(Linear); <1 beats linear
  diagnostics_<h>.png     IPFP wall time, trust-gate reverts, shock displacement vs γ_min
  sweep_summary.csv       every number in one flat table (+ per-hmode CSVs)

Pure numpy/matplotlib/json — no JAX, no GPU: run it locally after rsync-ing outputs/.

    uv run python SB/sweep_analysis.py
    uv run python SB/sweep_analysis.py --root outputs/Mach_interpolation/diamond/iso/hessian
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DRIFT_ORDER  = ["null", "oblique", "ffd"]
DRIFT_COLOR  = {"null": "tab:blue", "oblique": "tab:orange", "ffd": "tab:green"}
LINEAR_STYLE = dict(color="black", ls="--", lw=1.5, label="Linear (γ-independent)")
FFD_STYLE    = dict(color="tab:red", ls=":", lw=1.8, label="FFD registration (γ-independent)")

# BaryCDI is the SB result (γ-dependent);  Linear and FFD are γ-INDEPENDENT
# baselines — Linear needs no transport at all, FFD uses the raw registration map.
METHODS      = ["BaryCDI", "Linear", "FFD"]
METHOD_COLOR = {"BaryCDI": "tab:green", "Linear": "black", "FFD": "tab:red"}


# ─────────────────────────────────────────────────────────────────────────────
#  Loading
# ─────────────────────────────────────────────────────────────────────────────

def _mesh_h(cfg: dict) -> float:
    """Cell size h of the run's mesh, from the config echo in metrics.json.

    Read from ``cfg["mesh"]`` rather than from the directory path, so runs made
    before the output tree gained its h level still land at the right resolution
    instead of being lumped in with the refined ones.  Falls back to inferring h
    from the cell count (h ∝ 1/√N calibrated on diamond_h0.025) when the filename
    carries no _h tag.
    """
    m = re.search(r"_h([0-9.]+)\.npy$", str(cfg.get("mesh", "")))
    if m:
        return float(m.group(1))
    n = cfg.get("n_cells")
    return 0.025 * float(np.sqrt(20190.0 / n)) if n else float("nan")


def load_runs(root: str) -> list[dict]:
    """One flat row per run, from every metrics.json below root."""
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "**", "metrics.json"),
                                 recursive=True)):
        try:
            with open(path) as fh:
                m = json.load(fh)
        except Exception as exc:                              # noqa: BLE001
            print(f"  [warn] unreadable {path}: {exc}")
            continue
        cfg, ipfp, maps = m["config"], m["ipfp"], m["maps"]
        err = m.get("errors")
        tim = m.get("timings", {})
        recon = tim.get("recon_s", {}) or {}
        offline = tim.get("offline_s", {}) or {}
        aero = m.get("aero")
        row = dict(
            path=os.path.dirname(path),
            hmode=cfg.get("hessian_mode", "?"),
            drift=cfg.get("reference_drift", "null"),
            gamma=float(cfg["gamma_final"]),
            h=_mesh_h(cfg),
            n_cells=int(cfg.get("n_cells", 0)),
            n_stages=len(ipfp["stages"]),
            ipfp_iters=int(ipfp["total_iters"]),
            ipfp_res=float(ipfp["final_residual"]) if ipfp["final_residual"] is not None else np.nan,
            # Mesh-independent companion to ipfp_res; absent on pre-0C runs.
            ipfp_res_l2=float(ipfp.get("final_residual_l2") or np.nan),
            ipfp_converged=bool(ipfp["converged"]),
            ipfp_time_s=float(ipfp["total_time_s"]),
            ipfp_tol=float(cfg.get("ipfp_tol", 1e-7)),
            residual_history=ipfp.get("residual_history", []),
            stage_bounds=ipfp.get("stage_bounds", []),
            gate_T=int(maps.get("gate_reverts_T", 0)),
            gate_S=int(maps.get("gate_reverts_S", 0)),
            disp_T=float(np.linalg.norm(maps.get("mean_disp_T_shock", [0, 0]))),
            den_T_max=float(maps.get("den_T_max", np.nan)),
            t_ref=np.asarray(m.get("t_ref") or [], dtype=float),
        )
        for method in METHODS:
            for norm in ("l2", "linf", "w2"):
                vals = np.asarray((err or {}).get(method, {}).get(norm, []), dtype=float)
                row[f"{method}_{norm}"] = vals
                with np.errstate(all="ignore"):
                    row[f"{method}_{norm}_mean"] = (float(np.nanmean(vals))
                                                    if vals.size and np.any(np.isfinite(vals)) else np.nan)
                    row[f"{method}_{norm}_max"] = (float(np.nanmax(vals))
                                                   if vals.size and np.any(np.isfinite(vals)) else np.nan)
            # C_D/C_L (fix 0F).  Runs predating the aero block leave these NaN and
            # simply drop out of the aero panels.
            am = ((aero or {}).get("methods", {}) or {}).get(method, {})
            dcd = np.asarray(am.get("dC_D", []), dtype=float)
            cl  = np.asarray(am.get("C_L", []), dtype=float)
            row[f"{method}_cd"]           = np.asarray(am.get("C_D", []), dtype=float)
            row[f"{method}_dcd"]          = dcd
            row[f"{method}_cd_err_mean"]  = float(np.mean(np.abs(dcd))) if dcd.size else np.nan
            row[f"{method}_cd_err_max"]   = float(np.max(np.abs(dcd))) if dcd.size else np.nan
            row[f"{method}_cl_abs_max"]   = float(np.max(np.abs(cl))) if cl.size else np.nan
            # cost split: offline (IPFP solve / FFD registration) + online (query)
            row[f"{method}_recon_s"]   = float(recon.get(method, np.nan))
            row[f"{method}_offline_s"] = float(offline.get(method, np.nan))
            row[f"{method}_total_s"]   = (row[f"{method}_offline_s"]
                                          + (row[f"{method}_recon_s"] if np.isfinite(row[f"{method}_recon_s"]) else 0.0))
        row["interp_setup_s"] = float(tim.get("interp_setup_s", np.nan))
        row["interp_s"]       = float(tim.get("interp_s", np.nan))
        row["cd_ref"]  = np.asarray((aero or {}).get("C_D_ref", []), dtype=float)
        row["has_aero"] = bool(aero)
        rows.append(row)
    return rows


def _by_drift(rows, drift):
    sel = sorted((r for r in rows if r["drift"] == drift), key=lambda r: r["gamma"])
    return sel


def _linear_baseline(rows, key):
    vals = [r[key] for r in rows if np.isfinite(r[key])]
    return float(np.mean(vals)) if vals else np.nan


def _nearest_gamma(rows, gamma):
    """The row whose γ is closest to ``gamma`` — γ ladders differ between meshes."""
    cand = [r for r in rows if np.isfinite(r["gamma"])]
    return min(cand, key=lambda r: abs(np.log10(r["gamma"]) - np.log10(gamma))) if cand else None


# ─────────────────────────────────────────────────────────────────────────────
#  Mesh-resolution plots  (the deliverable of the h-ladder study)
# ─────────────────────────────────────────────────────────────────────────────

def plot_err_vs_h(rows, hmode, outdir, gammas=None):
    """Error vs cell size h, one panel per γ, three method series.

    The question this answers: BaryCDI should sharpen as the marginals sharpen,
    but the FFD drift lives on a fixed 144² Bernstein grid that does NOT depend on
    h — so if the FFD series flattens while BaryCDI falls, refinement is buying
    something the registration alone cannot.  If both flatten, the bottleneck is
    the drift resolution, not the mesh.
    """
    hs = sorted({r["h"] for r in rows if np.isfinite(r["h"])})
    if len(hs) < 2:
        print(f"  [skip] err_vs_h: only {len(hs)} mesh resolution(s) present")
        return
    if gammas is None:
        # γ values actually shared (to within a decade) across the meshes
        gammas = sorted({r["gamma"] for r in rows}, reverse=True)
        gammas = gammas[:4] if len(gammas) > 4 else gammas

    fig, axes = plt.subplots(1, len(gammas), figsize=(4.6*len(gammas), 4.6),
                             squeeze=False, sharey=True)
    for ax, gam in zip(axes[0], gammas):
        for method in METHODS:
            xs, ys = [], []
            for h in hs:
                sub = [r for r in rows if r["h"] == h]
                r = _nearest_gamma(sub, gam)
                if r is None or not np.isfinite(r[f"{method}_l2_mean"]):
                    continue
                xs.append(h);  ys.append(r[f"{method}_l2_mean"])
            if xs:
                ax.loglog(xs, ys, "o-", color=METHOD_COLOR[method], label=method)
        if hs:                                   # 1st-order reference slope
            y0 = ax.get_ylim()[1] * 0.7
            ax.loglog(hs, [y0 * (h/hs[0]) for h in hs], "k:", alpha=0.5,
                      lw=1, label=r"$\mathcal{O}(h)$")
        ax.set_xlabel("cell size $h$")
        ax.set_title(rf"$\gamma_{{\min}}\approx${gam:g}")
        ax.grid(True, which="both", alpha=0.3)
    axes[0][0].set_ylabel(r"mean$_t$  L2 error")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle(f"Interpolation error vs. mesh resolution — hessian_mode={hmode}")
    plt.tight_layout()
    f = os.path.join(outdir, f"err_vs_h_{hmode}.png")
    plt.savefig(f, dpi=160, bbox_inches="tight");  plt.close()
    print(f"  wrote {f}")


def plot_aero(rows, hmode, outdir):
    """C_D(t) against the reference, and mean|ΔC_D| vs h — the scalar companion.

    C_D is the wave drag: it is decided by the pressure ON THE WALL, so it catches
    a method that misplaces the shock feet even when the field-wide L2 looks fine.
    """
    rows = [r for r in rows if r["has_aero"]]
    if not rows:
        print("  [skip] aero: no run carries an aero block")
        return
    hs = sorted({r["h"] for r in rows if np.isfinite(r["h"])})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # Left: C_D(t) at the finest mesh and smallest γ available.
    best = min(rows, key=lambda r: (r["h"], r["gamma"]))
    t = best["t_ref"]
    if best["cd_ref"].size:
        axes[0].plot(t, best["cd_ref"], "ko-", lw=2, label="reference", zorder=5)
    for method in METHODS:
        cd = best[f"{method}_cd"]
        if cd.size:
            axes[0].plot(t, cd, "s--", color=METHOD_COLOR[method], alpha=0.85, label=method)
    axes[0].set_xlabel("t");  axes[0].set_ylabel(r"$C_D$")
    axes[0].set_title(rf"Drag coefficient  (h={best['h']:g}, $\gamma$={best['gamma']:g})")

    # Right: mean|ΔC_D| vs h, one series per method, at the smallest γ per mesh.
    for method in METHODS:
        xs, ys = [], []
        for h in hs:
            sub = [r for r in rows if r["h"] == h
                   and np.isfinite(r[f"{method}_cd_err_mean"])]
            if not sub:
                continue
            r = min(sub, key=lambda rr: rr["gamma"])
            xs.append(h);  ys.append(r[f"{method}_cd_err_mean"])
        if xs:
            axes[1].loglog(xs, ys, "o-", color=METHOD_COLOR[method], label=method)
    axes[1].set_xlabel("cell size $h$")
    axes[1].set_ylabel(r"mean$_t$  $|C_D - C_D^{\mathrm{ref}}|$")
    axes[1].set_title(r"Drag error vs. mesh resolution (smallest $\gamma$ per mesh)")

    for ax in axes:
        ax.legend(fontsize=8);  ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"Aerodynamic coefficients — hessian_mode={hmode}")
    plt.tight_layout()
    f = os.path.join(outdir, f"aero_{hmode}.png")
    plt.savefig(f, dpi=160, bbox_inches="tight");  plt.close()
    print(f"  wrote {f}")


# ─────────────────────────────────────────────────────────────────────────────
#  Plots (one call per hessian mode)
# ─────────────────────────────────────────────────────────────────────────────

def plot_err_vs_gamma(rows, hmode, outdir):
    panels = [("BaryCDI_l2_mean",  "mean$_t$  L2 error",       "Linear_l2_mean",  "FFD_l2_mean"),
              ("BaryCDI_linf_max", "max$_t$  L$\\infty$ error", "Linear_linf_max", "FFD_linf_max"),
              ("BaryCDI_w2_mean",  "mean$_t$  W$_2$ error",    "Linear_w2_mean",  "FFD_w2_mean")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (key, title, linkey, ffdkey) in zip(axes, panels):
        for drift in DRIFT_ORDER:
            sel = _by_drift(rows, drift)
            g   = [r["gamma"] for r in sel]
            v   = [r[key] for r in sel]
            if not g or not np.any(np.isfinite(v)):
                continue
            ax.plot(g, v, "o-", color=DRIFT_COLOR[drift], label=f"BaryCDI ({drift})")
        base = _linear_baseline(rows, linkey)
        if np.isfinite(base):
            ax.axhline(base, **LINEAR_STYLE)
        # FFD registration used directly as an interpolator — also γ-independent,
        # so it is the bar the SB bridge has to clear to justify its cost.
        fbase = _linear_baseline(rows, ffdkey)
        if np.isfinite(fbase):
            ax.axhline(fbase, **FFD_STYLE)
        ax.set_xscale("log");  ax.set_yscale("log")
        ax.invert_xaxis()                       # γ→0 (OT limit) to the right
        ax.set_xlabel(r"$\gamma_{\min}$  (annealed, log, OT limit $\rightarrow$)")
        ax.set_title(title);  ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Interpolation error vs. annealed γ_min — hessian_mode = {hmode}",
                 fontsize=13)
    fig.tight_layout()
    f = os.path.join(outdir, f"err_vs_gamma_{hmode}.png")
    fig.savefig(f, dpi=200, bbox_inches="tight");  plt.close(fig)
    print(f"  wrote {f}")


def plot_err_vs_t(rows, hmode, outdir):
    fig, axes = plt.subplots(1, len(DRIFT_ORDER), figsize=(16, 4.6),
                             sharey=True)
    gammas = sorted({r["gamma"] for r in rows})
    lognorm = matplotlib.colors.LogNorm(vmin=min(gammas), vmax=max(gammas))
    cmap = plt.cm.viridis
    for ax, drift in zip(np.atleast_1d(axes), DRIFT_ORDER):
        sel = _by_drift(rows, drift)
        for r in sel:
            if r["BaryCDI_l2"].size == 0:
                continue
            ax.plot(r["t_ref"], r["BaryCDI_l2"], "-", lw=1.4,
                    color=cmap(lognorm(r["gamma"])))
        lin = next((r for r in sel if r["Linear_l2"].size), None)
        if lin is not None:
            ax.plot(lin["t_ref"], lin["Linear_l2"], **LINEAR_STYLE)
        ax.set_title(f"drift = {drift}");  ax.set_xlabel("t")
        ax.grid(True, alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("L2 error")
    sm = plt.cm.ScalarMappable(norm=lognorm, cmap=cmap)
    fig.colorbar(sm, ax=list(np.atleast_1d(axes)), fraction=0.02, pad=0.02,
                 label=r"$\gamma_{\min}$")
    fig.suptitle(f"BaryCDI L2(t) colour-graded by γ_min — hessian_mode = {hmode}",
                 fontsize=13)
    f = os.path.join(outdir, f"err_vs_t_{hmode}.png")
    fig.savefig(f, dpi=200, bbox_inches="tight");  plt.close(fig)
    print(f"  wrote {f}")


def plot_convergence(rows, hmode, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    tol = rows[0]["ipfp_tol"] if rows else 1e-7

    for drift in DRIFT_ORDER:                                 # final residual vs γ
        sel = _by_drift(rows, drift)
        if not sel:
            continue
        axes[0].plot([r["gamma"] for r in sel], [r["ipfp_res"] for r in sel],
                     "o-", color=DRIFT_COLOR[drift], label=drift)
        axes[1].plot([r["gamma"] for r in sel], [r["ipfp_iters"] for r in sel],
                     "o-", color=DRIFT_COLOR[drift], label=drift)
        gmin_run = sel[0]                                     # smallest γ run
        hist = np.asarray(gmin_run["residual_history"], dtype=float)
        if hist.size:
            axes[2].semilogy(hist, color=DRIFT_COLOR[drift], lw=1.2,
                             label=f"{drift} (γ={gmin_run['gamma']:g})")
    axes[0].axhline(tol, color="red", ls="--", lw=1, label=f"tol={tol:g}")
    axes[0].set_xscale("log");  axes[0].set_yscale("log");  axes[0].invert_xaxis()
    axes[0].set_xlabel(r"$\gamma_{\min}$");  axes[0].set_title("final IPFP residual")
    axes[1].set_xscale("log");  axes[1].invert_xaxis()
    axes[1].set_xlabel(r"$\gamma_{\min}$")
    axes[1].set_title("total IPFP iterations (all stages)")
    axes[2].axhline(tol, color="red", ls="--", lw=1)
    axes[2].set_xlabel("iteration (stages concatenated)")
    axes[2].set_title("residual history at smallest γ")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3);  ax.legend(fontsize=8)
    fig.suptitle(f"IPFP convergence — hessian_mode = {hmode}", fontsize=13)
    fig.tight_layout()
    f = os.path.join(outdir, f"convergence_{hmode}.png")
    fig.savefig(f, dpi=200, bbox_inches="tight");  plt.close(fig)
    print(f"  wrote {f}")


def plot_gain_heatmap(rows, hmode, outdir):
    gammas = sorted({r["gamma"] for r in rows})
    M = np.full((len(DRIFT_ORDER), len(gammas)), np.nan)
    for i, drift in enumerate(DRIFT_ORDER):
        for r in _by_drift(rows, drift):
            j = gammas.index(r["gamma"])
            if np.isfinite(r["BaryCDI_l2_mean"]) and np.isfinite(r["Linear_l2_mean"]):
                M[i, j] = r["BaryCDI_l2_mean"] / r["Linear_l2_mean"]
    fig, ax = plt.subplots(figsize=(1.1 * len(gammas) + 2, 3.2))
    im = ax.imshow(M, cmap="RdBu_r", vmin=0.0, vmax=2.0, aspect="auto")
    ax.set_xticks(range(len(gammas)), [f"{g:g}" for g in gammas], rotation=45)
    ax.set_yticks(range(len(DRIFT_ORDER)), DRIFT_ORDER)
    ax.set_xlabel(r"$\gamma_{\min}$")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if abs(M[i, j] - 1) > 0.6 else "black")
    fig.colorbar(im, ax=ax, label="mean-t L2  BaryCDI / Linear   (<1 beats linear)")
    ax.set_title(f"SB gain over linear — hessian_mode = {hmode}")
    fig.tight_layout()
    f = os.path.join(outdir, f"gain_heatmap_{hmode}.png")
    fig.savefig(f, dpi=200, bbox_inches="tight");  plt.close(fig)
    print(f"  wrote {f}")


def plot_diagnostics(rows, hmode, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for drift in DRIFT_ORDER:
        sel = _by_drift(rows, drift)
        if not sel:
            continue
        g = [r["gamma"] for r in sel]
        axes[0].plot(g, [r["ipfp_time_s"] for r in sel], "o-",
                     color=DRIFT_COLOR[drift], label=drift)
        axes[1].plot(g, [r["gate_T"] + r["gate_S"] for r in sel], "o-",
                     color=DRIFT_COLOR[drift], label=drift)
        axes[2].plot(g, [r["disp_T"] for r in sel], "o-",
                     color=DRIFT_COLOR[drift], label=drift)
    titles = ["IPFP wall time [s]",
              "trust-gate reverted cells (T+S)",
              r"$|$mean shock displacement of $T|$"]
    for ax, title in zip(axes, titles):
        ax.set_xscale("log");  ax.invert_xaxis()
        ax.set_xlabel(r"$\gamma_{\min}$");  ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3);  ax.legend(fontsize=8)
    axes[0].set_yscale("log")
    fig.suptitle(f"Sweep diagnostics — hessian_mode = {hmode}", fontsize=13)
    fig.tight_layout()
    f = os.path.join(outdir, f"diagnostics_{hmode}.png")
    fig.savefig(f, dpi=200, bbox_inches="tight");  plt.close(fig)
    print(f"  wrote {f}")


def plot_timing(rows, hmode, outdir):
    """Cost breakdown vs γ: offline solve, online query, and their split."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) offline cost: the IPFP solve per drift (BaryCDI's setup) + FFD registration
    for drift in DRIFT_ORDER:
        sel = _by_drift(rows, drift)
        if not sel:
            continue
        axes[0].plot([r["gamma"] for r in sel], [r["ipfp_time_s"] for r in sel],
                     "o-", color=DRIFT_COLOR[drift], label=f"IPFP ({drift})")
    ffd_off = [r["FFD_offline_s"] for r in rows if np.isfinite(r.get("FFD_offline_s", np.nan))]
    if ffd_off:
        axes[0].axhline(float(np.mean(ffd_off)), **FFD_STYLE)
    axes[0].set_title("offline cost  [s]\n(SB: IPFP anneal · FFD: registration)")

    # (b) online cost: per-frame reconstruction, per method (should be γ-flat)
    for meth in METHODS:
        vals = [(r["gamma"], r[f"{meth}_recon_s"]) for r in rows
                if np.isfinite(r.get(f"{meth}_recon_s", np.nan))]
        if not vals:
            continue
        vals.sort()
        axes[1].plot([v[0] for v in vals], [v[1] for v in vals], "o-",
                     color=METHOD_COLOR[meth], label=meth, ms=4)
    axes[1].set_title("online cost  [s]\n(reconstruct all frames)")

    # (c) total wall time per run, split into IPFP vs everything else
    for drift in DRIFT_ORDER:
        sel = _by_drift(rows, drift)
        if not sel:
            continue
        axes[2].plot([r["gamma"] for r in sel],
                     [r["ipfp_time_s"] + (r["interp_s"] if np.isfinite(r["interp_s"]) else 0.0)
                      for r in sel],
                     "o-", color=DRIFT_COLOR[drift], label=drift)
    axes[2].set_title("IPFP + interpolation  [s]")

    for ax in axes:
        ax.set_xscale("log");  ax.set_yscale("log");  ax.invert_xaxis()
        ax.set_xlabel(r"$\gamma_{\min}$")
        ax.grid(True, which="both", alpha=0.3);  ax.legend(fontsize=8)
    fig.suptitle(f"Cost vs γ — hessian_mode = {hmode}", fontsize=13)
    fig.tight_layout()
    f = os.path.join(outdir, f"timing_{hmode}.png")
    fig.savefig(f, dpi=200, bbox_inches="tight");  plt.close(fig)
    print(f"  wrote {f}")


def plot_cost_accuracy(rows, hmode, outdir):
    """
    Pareto view: what does the accuracy actually cost?

    x = total time to produce the interpolation (offline solve + online query),
    y = mean-t L2 error.  Lower-left is better.  SB (BaryCDI) appears once per
    (drift, γ); Linear and FFD appear as single γ-independent reference points.
    If no SB point sits below-left of the FFD point, the bridge is not paying
    for itself on this case.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    gammas = sorted({r["gamma"] for r in rows})
    if not gammas:
        return
    lognorm = matplotlib.colors.LogNorm(vmin=min(gammas), vmax=max(gammas))
    marker = {"null": "o", "oblique": "s", "ffd": "^"}

    for drift in DRIFT_ORDER:
        for r in _by_drift(rows, drift):
            cost = r["ipfp_time_s"] + (r["BaryCDI_recon_s"]
                                       if np.isfinite(r["BaryCDI_recon_s"]) else 0.0)
            err = r["BaryCDI_l2_mean"]
            if not (np.isfinite(cost) and np.isfinite(err)):
                continue
            ax.scatter(cost, err, marker=marker[drift], s=55,
                       c=[plt.cm.viridis(lognorm(r["gamma"]))],
                       edgecolors="k", linewidths=0.4, zorder=3)

    for meth, style in (("Linear", LINEAR_STYLE), ("FFD", FFD_STYLE)):
        c = [r[f"{meth}_total_s"] for r in rows if np.isfinite(r.get(f"{meth}_total_s", np.nan))]
        e = [r[f"{meth}_l2_mean"] for r in rows if np.isfinite(r.get(f"{meth}_l2_mean", np.nan))]
        if not c or not e:
            continue
        ax.scatter(np.mean(c), np.mean(e), marker="*", s=320,
                   color=style["color"], edgecolors="k", linewidths=0.6,
                   zorder=4, label=meth)
        ax.axhline(float(np.mean(e)), color=style["color"], ls=style["ls"], lw=1.0, alpha=0.6)

    handles = [plt.Line2D([], [], ls="", marker=marker[d], color="grey",
                          markeredgecolor="k", label=f"BaryCDI ({d})") for d in DRIFT_ORDER]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0], fontsize=8)
    sm = plt.cm.ScalarMappable(norm=lognorm, cmap=plt.cm.viridis)
    fig.colorbar(sm, ax=ax, label=r"$\gamma_{\min}$")
    ax.set_xscale("log");  ax.set_yscale("log")
    ax.set_xlabel("time to produce the interpolation  [s]   (offline + online)")
    ax.set_ylabel(r"mean$_t$  L2 error")
    ax.set_title(f"Cost vs accuracy — hessian_mode = {hmode}\n(lower-left is better)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    f = os.path.join(outdir, f"cost_accuracy_{hmode}.png")
    fig.savefig(f, dpi=200, bbox_inches="tight");  plt.close(fig)
    print(f"  wrote {f}")


def plot_sb_vs_ffd(rows, hmode, outdir):
    """
    Head-to-head: SB (BaryCDI) / FFD-registration error ratio vs γ.
    <1 means the Schrödinger bridge genuinely improves on the raw registration
    map it was built from;  >1 means the registration alone was better.
    """
    have = [r for r in rows if np.isfinite(r.get("FFD_l2_mean", np.nan))]
    if not have:
        print(f"  (no FFD runs for hmode={hmode} — skipping sb_vs_ffd)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, norm, lbl in zip(axes, ("l2_mean", "linf_max", "w2_mean"),
                             ("mean$_t$ L2", "max$_t$ L$\\infty$", "mean$_t$ W$_2$")):
        for drift in DRIFT_ORDER:
            sel = _by_drift(rows, drift)
            g, ratio = [], []
            for r in sel:
                sb = r.get(f"BaryCDI_{norm}", np.nan)
                # FFD baseline is γ-independent: use the mean over the runs that have it
                fv = [x[f"FFD_{norm}"] for x in have if np.isfinite(x[f"FFD_{norm}"])]
                if not (np.isfinite(sb) and fv):
                    continue
                g.append(r["gamma"]);  ratio.append(sb / float(np.mean(fv)))
            if g:
                ax.plot(g, ratio, "o-", color=DRIFT_COLOR[drift], label=drift)
        ax.axhline(1.0, color="k", ls="--", lw=1.2)
        ax.set_xscale("log");  ax.set_yscale("log");  ax.invert_xaxis()
        ax.set_xlabel(r"$\gamma_{\min}$")
        ax.set_ylabel("SB / FFD")
        ax.set_title(lbl);  ax.grid(True, which="both", alpha=0.3);  ax.legend(fontsize=8)
    fig.suptitle(f"Schrödinger bridge vs. raw FFD registration  (<1 = SB better) "
                 f"— hessian_mode = {hmode}", fontsize=13)
    fig.tight_layout()
    f = os.path.join(outdir, f"sb_vs_ffd_{hmode}.png")
    fig.savefig(f, dpi=200, bbox_inches="tight");  plt.close(fig)
    print(f"  wrote {f}")


# ─────────────────────────────────────────────────────────────────────────────
#  CSV
# ─────────────────────────────────────────────────────────────────────────────

CSV_COLS = ["hmode", "h", "n_cells", "drift", "gamma", "n_stages",
            # accuracy — SB vs the two γ-independent baselines
            "BaryCDI_l2_mean", "Linear_l2_mean", "FFD_l2_mean",
            "BaryCDI_linf_max", "Linear_linf_max", "FFD_linf_max",
            "BaryCDI_w2_mean", "Linear_w2_mean", "FFD_w2_mean",
            # scalar physics metric: wave drag error, and the C_L symmetry residual
            "BaryCDI_cd_err_mean", "Linear_cd_err_mean", "FFD_cd_err_mean",
            "BaryCDI_cl_abs_max", "Linear_cl_abs_max", "FFD_cl_abs_max",
            # cost — offline solve + online query, per method
            "BaryCDI_offline_s", "BaryCDI_recon_s", "BaryCDI_total_s",
            "FFD_offline_s", "FFD_recon_s", "FFD_total_s",
            "Linear_recon_s", "interp_setup_s", "interp_s",
            # convergence + map diagnostics
            "ipfp_res", "ipfp_res_l2", "ipfp_converged", "ipfp_iters", "ipfp_time_s",
            "gate_T", "gate_S", "disp_T", "den_T_max", "path"]


def write_csv(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["hmode"], r["drift"], r["gamma"])):
            w.writerow({k: r.get(k) for k in CSV_COLS})
    print(f"  wrote {path}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="outputs/Mach_interpolation/diamond/iso/hessian",
                    help="directory containing <hmode>/drift_*/gmin*/metrics.json")
    ap.add_argument("--out", default=None,
                    help="summary output dir (default <root>/_sweep)")
    args = ap.parse_args()

    outdir = args.out or os.path.join(args.root, "_sweep")
    os.makedirs(outdir, exist_ok=True)

    rows = load_runs(args.root)
    if not rows:
        raise SystemExit(f"No metrics.json found under {args.root}")
    hmodes = sorted({r["hmode"] for r in rows})
    all_hs = sorted({r["h"] for r in rows if np.isfinite(r["h"])})
    print(f"Loaded {len(rows)} runs  (hessian modes: {hmodes};  "
          f"mesh h: {[f'{h:g}' for h in all_hs]})")

    for hmode in hmodes:
        sub_h = [r for r in rows if r["hmode"] == hmode]
        # The γ-sweep plots compare runs at one resolution; overlaying meshes on
        # them would silently mix ladders.  Cross-h comparison is the job of
        # plot_err_vs_h / plot_aero, which are called once per hessian mode.
        for h in sorted({r["h"] for r in sub_h if np.isfinite(r["h"])}):
            sub = [r for r in sub_h if r["h"] == h]
            tag = f"{hmode}_h{h:g}" if len(all_hs) > 1 else hmode
            print(f"── hessian_mode = {hmode}, h = {h:g}  ({len(sub)} runs) ──")
            plot_err_vs_gamma(sub, tag, outdir)
            plot_err_vs_t(sub, tag, outdir)
            plot_convergence(sub, tag, outdir)
            plot_gain_heatmap(sub, tag, outdir)
            plot_diagnostics(sub, tag, outdir)
            plot_timing(sub, tag, outdir)
            plot_cost_accuracy(sub, tag, outdir)
            plot_sb_vs_ffd(sub, tag, outdir)

        print(f"── hessian_mode = {hmode}: cross-resolution ──")
        plot_err_vs_h(sub_h, hmode, outdir)
        plot_aero(sub_h, hmode, outdir)
        write_csv(sub_h, os.path.join(outdir, f"sweep_summary_{hmode}.csv"))

    write_csv(rows, os.path.join(outdir, "sweep_summary.csv"))
    print(f"Done → {outdir}/")


if __name__ == "__main__":
    main()
