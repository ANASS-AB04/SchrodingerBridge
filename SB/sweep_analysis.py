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

# SBsquared* = the bootstrapped self-conditioned drift (β_{k+1} = the previous γ
# stage's own SB drift).  Listed here or _by_drift filters them out and the runs
# load but vanish silently from every plot.
DRIFT_ORDER  = ["null", "oblique", "ffd", "SBsquared", "SBsquared_exact",
                "SBsquared+ffd", "SBsquared_exact+ffd"]
DRIFT_COLOR  = {"null": "tab:blue", "oblique": "tab:orange", "ffd": "tab:green",
                "SBsquared": "tab:purple", "SBsquared_exact": "tab:brown",
                "SBsquared+ffd": "tab:pink", "SBsquared_exact+ffd": "tab:cyan"}
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
    m = re.search(r"_h([0-9.]+)(?:le[0-9.]+)?\.npy$", str(cfg.get("mesh", "")))
    if m:
        return float(m.group(1))
    n = cfg.get("n_cells")
    return 0.025 * float(np.sqrt(20190.0 / n)) if n else float("nan")


def _mesh_variant(cfg: dict) -> str:
    """"le7" for a nose-refined corpus (mesh <case>_h0.025le7.npy), "" otherwise.

    Those references were solved on a nested nose-refined mesh (see
    Euler/meshing/mesh_utils.py) and are a different corpus from the plain-h one
    at the SAME h, so the variant is folded into the hmode label ("eig_le7"):
    every grouping, plot and CSV then keeps the two apart.
    """
    m = re.search(r"_h[0-9.]+(le[0-9.]+)\.npy$", str(cfg.get("mesh", "")))
    return m.group(1) if m else ""


def run_label(rows):
    """Readable descriptor of WHICH experiment a figure shows.

    Figures were titled with the output-path tag (`eig_h0.025_M2-2.1`): not
    readable, and incomplete -- it never carried the AoA, so a Mach sweep at
    0 deg and the same sweep at 2 deg produced titles differing only in a
    filename fragment.  Everything here comes from the run's own metadata.
    """
    if not rows:
        return ""
    r = rows[0]
    bits = []
    if r.get("hmode"):
        bits.append(str(r["hmode"]))
    h = r.get("h", float("nan"))
    if h == h:
        n = r.get("n_cells")
        bits.append(f"h={h:g}" + (f" ({int(n):,} cells)" if n else ""))
    if r.get("axis") == "aoa":
        m = r.get("mach0", float("nan"))
        if m == m:
            bits.append(f"M={m:g}")
        bits.append(f"AoA {float(r.get('aoa0', 0)):g}$^\\circ$"
                    f"$\\to${float(r.get('aoa1', 0)):g}$^\\circ$")
    else:
        bits.append(f"AoA {float(r.get('aoa', 0)):g}$^\\circ$")
        m0, m1 = r.get("mach0", float("nan")), r.get("mach1", float("nan"))
        if m0 == m0 and m1 == m1:
            bits.append(f"M {m0:.2f}$\\to${m1:.2f}")
    return "   ·   ".join(bits)


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
            hmode=cfg.get("hessian_mode", "?") + (f"_{v}" if (v := _mesh_variant(cfg)) else ""),
            # Seeded bootstraps are distinct schemes: compose the name so that
            # SBsquared_exact seeded from ffd never pools with the heat-seeded one.
            drift=(cfg.get("reference_drift", "null")
                   + (f"+{m['drift']['bootstrap_seed']}"
                      if (m.get("drift", {}).get("bootstrap_seed") or "null") != "null"
                      else "")),
            gamma=float(cfg["gamma_final"]),
            h=_mesh_h(cfg),
            # AoA separates genuinely different physics: at 0 the diamond is
            # symmetric so C_L is mesh-asymmetry noise, at 2 deg it is a real
            # signal ~1000x larger.  Never pool them in one plot.
            aoa=float(cfg.get("aoa_deg", 0.0) or 0.0),
            # Which parameter the bridge interpolates.  Runs predating axis
            # detection carry interp_axis=None, so fall back to the output root:
            # an AoA-axis run always lives under AoA_interpolation/.
            axis=(cfg.get("interp_axis")
                  or ("aoa" if "AoA_interpolation" in path else "mach")),
            # Endpoint angles, so an AoA sweep can be labelled by its RANGE.
            # `aoa` alone is the first bundle's angle, which reads as "AoA = 0°"
            # for a 0→4° sweep — true of the start point, misleading as a title.
            aoa0=float(cfg.get("aoa0_deg", cfg.get("aoa_deg", 0.0) or 0.0)),
            aoa1=float(cfg.get("aoa1_deg", cfg.get("aoa_deg", 0.0) or 0.0)),
            # The interpolated Mach RANGE.  With one pair (2.00→2.50) this was
            # constant and could be ignored; across a 0.80→3.00 survey it is the
            # main independent variable, and pooling pairs would average over
            # three different flow regimes.
            mach0=float(cfg.get("mach0_inlet", float("nan")) or float("nan")),
            mach1=float(cfg.get("mach1_inlet", float("nan")) or float("nan")),
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
            dcl = np.asarray(am.get("dC_L", []), dtype=float)
            cl  = np.asarray(am.get("C_L", []), dtype=float)
            row[f"{method}_cd"]           = np.asarray(am.get("C_D", []), dtype=float)
            row[f"{method}_cl"]           = cl
            row[f"{method}_dcd"]          = dcd
            row[f"{method}_dcl"]          = dcl
            row[f"{method}_cd_err_mean"]  = float(np.mean(np.abs(dcd))) if dcd.size else np.nan
            row[f"{method}_cd_err_max"]   = float(np.max(np.abs(dcd))) if dcd.size else np.nan
            # Lift error measured against the reference the same way drag is, rather
            # than against an assumed C_L=0: at AoA=0 the reference C_L is small but
            # NOT zero (~4.5e-4, ~3% of C_D), and that residual is the noise floor.
            row[f"{method}_cl_err_mean"]  = float(np.mean(np.abs(dcl))) if dcl.size else np.nan
            row[f"{method}_cl_err_max"]   = float(np.max(np.abs(dcl))) if dcl.size else np.nan
            row[f"{method}_cl_abs_max"]   = float(np.max(np.abs(cl))) if cl.size else np.nan
            # cost split: offline (IPFP solve / FFD registration) + online (query)
            row[f"{method}_recon_s"]   = float(recon.get(method, np.nan))
            row[f"{method}_offline_s"] = float(offline.get(method, np.nan))
            row[f"{method}_total_s"]   = (row[f"{method}_offline_s"]
                                          + (row[f"{method}_recon_s"] if np.isfinite(row[f"{method}_recon_s"]) else 0.0))
        row["interp_setup_s"] = float(tim.get("interp_setup_s", np.nan))
        row["interp_s"]       = float(tim.get("interp_s", np.nan))
        # ── Transport metrics (shock placement / sharpness / admissibility) ──
        tr = m.get("transport") or {}
        trm = tr.get("methods", {}) or {}
        trr = tr.get("ref", {}) or {}
        row["beta_analytic"] = np.asarray(trr.get("angle_analytic_deg", []), dtype=float)
        row["beta_ref"]      = np.asarray(trr.get("angle_deg", []), dtype=float)
        row["gradmax_ref"]   = np.asarray(trr.get("grad_max", []), dtype=float)
        row["has_transport"] = bool(tr)
        _gref = float(np.nanmean(row["gradmax_ref"])) if row["gradmax_ref"].size else np.nan
        for method in METHODS:
            d = trm.get(method, {})
            for key, col in (("l2_band", "l2band"), ("angle_err_deg", "dbeta"),
                             ("rh_res", "rh")):
                v = np.asarray(d.get(key, []), dtype=float)
                row[f"{method}_{col}"] = v
                with np.errstate(all="ignore"):
                    row[f"{method}_{col}_mean"] = (float(np.nanmean(v))
                                                   if v.size and np.any(np.isfinite(v)) else np.nan)
            gm = np.asarray(d.get("grad_max", []), dtype=float)
            row[f"{method}_gradmax"] = gm
            with np.errstate(all="ignore"):
                # sharpness RELATIVE to the reference normalises out the mesh's
                # own resolution limit, so it is comparable across the h ladder.
                row[f"{method}_sharpness"] = (float(np.nanmean(gm)) / _gref
                                              if gm.size and np.isfinite(_gref) and _gref > 0
                                              else np.nan)
        row["cd_ref"]  = np.asarray((aero or {}).get("C_D_ref", []), dtype=float)
        row["cl_ref"]  = np.asarray((aero or {}).get("C_L_ref", []), dtype=float)
        row["cd_ref_mean"] = float(np.mean(np.abs(row["cd_ref"]))) if row["cd_ref"].size else np.nan
        # At AoA=0 a symmetric body should give C_L=0 exactly; whatever the reference
        # actually shows is solver/mesh asymmetry and bounds what any C_L comparison
        # can resolve.
        row["cl_ref_mean"] = float(np.mean(np.abs(row["cl_ref"]))) if row["cl_ref"].size else np.nan
        row["has_aero"] = bool(aero)
        rows.append(row)
    return rows


def case_label(rows_or_row):
    """Human label for a (axis, aoa) group — a RANGE for an angle sweep.

    An AoA sweep spans 0→4°, so labelling it by its first bundle's angle ("AoA =
    0°") describes the start point rather than the experiment.
    """
    r = rows_or_row[0] if isinstance(rows_or_row, list) else rows_or_row
    if r["axis"] == "aoa":
        m = r.get("mach0", float("nan"))
        at = f" @ M{m:g}" if m == m else ""
        return f"AoA sweep {r['aoa0']:g}°→{r['aoa1']:g}°{at}"
    m0, m1 = r.get("mach0", float("nan")), r.get("mach1", float("nan"))
    rng = f" {m0:g}→{m1:g}" if m0 == m0 and m1 == m1 else ""
    return f"Mach sweep{rng} @ AoA {r['aoa']:g}°"


def _pair_key(r):
    """Everything that makes two runs the SAME interpolation problem.

    Before the Mach survey this was just (axis, aoa) because there was one pair;
    now 22 Mach pairs share an axis and an AoA, so keying without the endpoints
    would pool three flow regimes into one curve and let each pair overwrite the
    previous one's figures.
    """
    return (r["axis"], r["aoa"],
            round(float(r.get("mach0", 0.0) or 0.0), 4),
            round(float(r.get("mach1", 0.0) or 0.0), 4),
            round(float(r.get("aoa0", 0.0) or 0.0), 4),
            round(float(r.get("aoa1", 0.0) or 0.0), 4))


def _pair_tag(r):
    """Filename suffix for a pair.  EMPTY for the legacy pairs, so the figures of
    the 576 finished runs keep their exact names instead of being duplicated."""
    if r["axis"] == "aoa":
        t = "_axisaoa"
        a0, a1 = float(r.get("aoa0", 0.0)), float(r.get("aoa1", 4.0))
        if not (abs(a0) < 1e-9 and abs(a1 - 4.0) < 1e-9):
            t += f"_A{a0:g}-{a1:g}"
        return t
    aoa = float(r["aoa"])
    t = "" if abs(aoa) < 1e-9 else f"_aoa{aoa:g}"
    m0, m1 = float(r.get("mach0", 2.0)), float(r.get("mach1", 2.5))
    if not (abs(m0 - 2.0) < 1e-9 and abs(m1 - 2.5) < 1e-9):
        t += f"_M{m0:g}-{m1:g}"
    return t


def backfill_ffd_control(rows):
    """Share each case's FFD-control numbers across the drifts in that case.

    The FFD control is the raw registration used directly as an interpolator, so it
    depends ONLY on the two marginals and the mesh — not on the reference drift
    driving the kernel, and not on γ.  (Confirmed in the data: FFD_l2_mean is
    identical across all 12 γ within a case.)  But it is only *computed* on runs
    that actually build the registration, which historically meant the ffd-drift
    runs alone — leaving the FFD column NaN on every null and oblique row, i.e.
    exactly where "does SB beat the raw registration" most needed answering.

    Rather than spend ~24 GPU-h re-running those, copy the value within each
    (hmode, h, axis, aoa) group, which is the granularity at which the marginals
    are identical.  Runs that computed their own FFD keep it untouched, and the
    ``ffd_backfilled`` flag marks the borrowed ones so a shared value is never
    mistaken for a measured one.

    Caveat measured in the data: runs of the SAME case that each computed their own
    FFD disagree in the 4th significant figure (e.g. 0.0022563 vs 0.0022586, ~0.1%).
    The registration is an 800-step Adam optimisation, and these runs executed on
    different GPU models across different jobs, so reduction order differs.  That is
    ~400× smaller than the SB-vs-FFD gap being measured and changes no conclusion,
    but it means the FFD control is reproducible only to ~0.1% across hardware.
    Donor choice is deterministic (first in sorted-path order) so re-running the
    analysis is at least stable.
    """
    groups = {}
    for r in rows:
        # The endpoints belong in the key: the FFD control is a function of the two
        # marginals, so sharing it between a 0.80→0.90 bridge and a 2.90→3.00 one
        # (same hmode/h/axis/aoa) would fabricate a number, not back-fill one.
        groups.setdefault((r["hmode"], r["h"], r["axis"], r["aoa"],
                           r.get("mach0"), r.get("mach1"),
                           r.get("aoa0"), r.get("aoa1")), []).append(r)
    n_filled = 0
    for grp in groups.values():
        donor = next((r for r in grp
                      if np.isfinite(r.get("FFD_l2_mean", np.nan))), None)
        if donor is None:
            continue
        ffd_keys = [k for k in donor if k.startswith("FFD_")]
        for r in grp:
            if r is donor or np.isfinite(r.get("FFD_l2_mean", np.nan)):
                continue
            for k in ffd_keys:
                r[k] = donor[k]
            r["ffd_backfilled"] = True
            n_filled += 1
    if n_filled:
        print(f"  FFD control back-filled onto {n_filled} runs "
              f"(γ- and drift-independent; shared within each case)")
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

def plot_err_vs_mach(rows, tag, outdir, drifts=None):
    """Error vs the interpolated Mach BAND — the deliverable of the Mach survey.

    Every other figure holds the pair fixed and varies gamma or h.  This one holds
    gamma and h fixed and walks the pair across 0.8 -> 3.0, which is the only view
    that answers the survey's actual question: does the ranking of BaryCDI / FFD /
    Linear depend on the flow regime, or is the supersonic result universal?

    The three shaded bands are the physics, and they are not cosmetic.  Below M=1
    the diamond carries NO shock, so the Hessian marginal keys on smooth-field
    curvature and the whole transport premise is different.  Between 1.0 and ~1.25
    the leading-edge shock is DETACHED (a bow shock), which is also where the
    analytic oblique drift ceases to exist.  Only the right-hand band is the
    regime the pipeline was built and validated for -- read the left two as
    exploratory, not as equal-confidence evidence.
    """
    rows = [r for r in rows if r["axis"] == "mach"
            and np.isfinite(r.get("mach0", np.nan))
            and np.isfinite(r.get("mach1", np.nan))]
    pairs = sorted({(r["mach0"], r["mach1"]) for r in rows})
    if len(pairs) < 2:
        print(f"  [skip] err_vs_mach: only {len(pairs)} Mach pair(s) present")
        return
    if drifts is None:
        drifts = [d for d in DRIFT_ORDER if any(r["drift"] == d for r in rows)]
    gammas = sorted({r["gamma"] for r in rows})
    gam = gammas[0] if gammas else None          # the tightest rung reached

    fig, axes = plt.subplots(1, len(drifts), figsize=(4.6*len(drifts), 4.6),
                             squeeze=False, sharey=True)
    for ax, drift in zip(axes[0], drifts):
        sub_d = [r for r in rows if r["drift"] == drift]
        for method in METHODS:
            xs, ys = [], []
            for m0, m1 in pairs:
                sub = [r for r in sub_d if r["mach0"] == m0 and r["mach1"] == m1]
                r = _nearest_gamma(sub, gam) if sub else None
                if r is None or not np.isfinite(r[f"{method}_l2_mean"]):
                    continue
                xs.append(0.5*(m0 + m1))         # the band's midpoint
                ys.append(r[f"{method}_l2_mean"])
            if xs:
                ax.semilogy(xs, ys, "o-", ms=4, color=METHOD_COLOR[method],
                            label=method)
        # regime bands
        ax.axvspan(0.0, 1.0, color="tab:red",    alpha=0.07)
        ax.axvspan(1.0, 1.25, color="tab:orange", alpha=0.07)
        ax.axvspan(1.25, 4.0, color="tab:green",  alpha=0.05)
        ax.set_xlim(min(0.5*(a+b) for a, b in pairs) - 0.05,
                    max(0.5*(a+b) for a, b in pairs) + 0.05)
        ax.set_xlabel("Mach band midpoint")
        ax.set_title(f"drift = {drift}")
        ax.grid(True, which="both", alpha=0.3)
    axes[0][0].set_ylabel(r"mean$_t$  L2 error")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle(f"Interpolation error across the Mach range — {tag}"
                 rf"  ($\gamma_{{\min}}\approx${gam:g})"
                 "\n"
                 "red: no shock (M<1)   orange: detached bow shock   "
                 "green: attached oblique shock (validated regime)",
                 fontsize=10)
    plt.tight_layout()
    f = os.path.join(outdir, f"err_vs_mach_{tag}.png")
    plt.savefig(f, dpi=160, bbox_inches="tight");  plt.close()
    print(f"  wrote {f}")


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
    fig.suptitle(f"Interpolation error vs. mesh resolution\n{run_label(rows)}")
    plt.tight_layout()
    f = os.path.join(outdir, f"err_vs_h_{hmode}.png")
    plt.savefig(f, dpi=160, bbox_inches="tight");  plt.close()
    print(f"  wrote {f}")


def plot_transport(rows, hmode, outdir):
    """The four transport metrics vs γ, plus shock angle vs the ANALYTIC value.

    These answer what the domain-wide norms cannot.  ~68% of cells are undisturbed
    freestream, so a global L2 mostly scores a region every method gets free; and
    C_D/C_L integrate over the WALL, where the true transport is the identity, so
    they reward doing nothing.  Here: did the shock land in the right PLACE
    (Δβ, and L2 restricted to the shock band), stay SHARP (gradient vs the
    reference's), and remain a PHYSICALLY VALID jump (RH residual)?

    Δβ and the RH residual are measured against θ-β-M / Rankine-Hugoniot, i.e.
    against physics rather than against a reference snapshot — so unlike every
    other metric here they carry no reference-discretisation noise floor.
    """
    rows = [r for r in rows if r.get("has_transport")]
    if not rows:
        print("  [skip] transport: no run carries a transport block")
        return
    panels = [("l2band_mean", r"mean$_t$  L2 on the shock band", True),
              ("dbeta_mean",  r"mean$_t$  $|\beta-\beta_{\theta\beta M}|$  [deg]", True),
              ("sharpness",   r"mean $\|\nabla M\|$ / reference", False),
              ("rh_mean",     r"mean$_t$  Rankine-Hugoniot residual", True)]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    for ax, (key, ylab, logy) in zip(axes, panels):
        for method in METHODS:
            pts = sorted(((r["gamma"], r.get(f"{method}_{key}", np.nan)) for r in rows),
                         key=lambda z: z[0])
            xs = [a for a, b in pts if np.isfinite(b)]
            ys = [b for a, b in pts if np.isfinite(b)]
            if xs:
                (ax.loglog if logy else ax.semilogx)(
                    xs, ys, "o-", color=METHOD_COLOR[method], label=method)
        if key == "sharpness":
            ax.axhline(1.0, color="grey", ls=":", lw=1.5, label="reference sharpness")
        ax.set_xlabel(r"$\gamma_{\min}$");  ax.set_ylabel(ylab)
        ax.grid(True, which="both", alpha=0.3);  ax.legend(fontsize=8)
    fig.suptitle(f"Transport-quality metrics\n{run_label(rows)}")
    plt.tight_layout()
    f = os.path.join(outdir, f"transport_{hmode}.png")
    plt.savefig(f, dpi=160, bbox_inches="tight");  plt.close()
    print(f"  wrote {f}")


def plot_aero(rows, hmode, outdir):
    """C_D and C_L against the reference, plus their errors vs mesh resolution.

    These are wall quantities: both are ∮p·n over the body, so they are decided by
    the pressure ON THE SURFACE and catch a method that displaces the wall even when
    its field-wide L2 looks good.  (That is not hypothetical — the FFD registration
    carries ~0.6 cell-widths of drift at the wall cells despite the body being
    geometrically identical at both endpoint Mach numbers, and it is last on C_D
    while being first on L2.)

    C_L caveat: at AoA=0 a symmetric body should give C_L=0 exactly.  The reference
    does not — it shows ~4.5e-4, about 3% of C_D — and that residual is solver/mesh
    asymmetry.  It is the noise floor: no C_L difference below it means anything.
    The floor is drawn on the plot so it cannot be read past by accident.
    """
    rows = [r for r in rows if r["has_aero"]]
    if not rows:
        print("  [skip] aero: no run carries an aero block")
        return
    hs = sorted({r["h"] for r in rows if np.isfinite(r["h"])})

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))

    best = min(rows, key=lambda r: (r["h"], r["gamma"]))
    t = best["t_ref"]
    sub_ttl = rf"(h={best['h']:g}, $\gamma$={best['gamma']:g})"

    for col, (coef, ref_key, err_key, sym) in enumerate([
            ("cd", "cd_ref", "cd_err_mean", "C_D"),
            ("cl", "cl_ref", "cl_err_mean", "C_L")]):
        # Top row: coefficient vs t, reference overlaid
        ax = axes[0][col]
        if best[ref_key].size:
            ax.plot(t, best[ref_key], "ko-", lw=2, label="reference", zorder=5)
        for method in METHODS:
            v = best[f"{method}_{coef}"]
            if v.size:
                ax.plot(t, v, "s--", color=METHOD_COLOR[method], alpha=0.85, label=method)
        ax.set_xlabel("t");  ax.set_ylabel(rf"${sym}$")
        ax.set_title(rf"${sym}$  {sub_ttl}")

        # Bottom row: mean error vs mesh resolution, smallest γ per mesh
        ax = axes[1][col]
        for method in METHODS:
            xs, ys = [], []
            for h in hs:
                cand = [r for r in rows if r["h"] == h
                        and np.isfinite(r[f"{method}_{err_key}"])]
                if not cand:
                    continue
                r = min(cand, key=lambda rr: rr["gamma"])
                xs.append(h);  ys.append(r[f"{method}_{err_key}"])
            if xs:
                ax.loglog(xs, ys, "o-", color=METHOD_COLOR[method], label=method)
        if coef == "cl":
            floor = np.nanmean([r["cl_ref_mean"] for r in rows
                                if np.isfinite(r.get("cl_ref_mean", np.nan))])
            if np.isfinite(floor) and hs:
                ax.axhline(floor, color="grey", ls=":", lw=1.5,
                           label=rf"reference $|C_L|$ = {floor:.1e} (noise floor)")
        ax.set_xlabel("cell size $h$")
        ax.set_ylabel(rf"mean$_t$  $|{sym} - {sym}^{{\mathrm{{ref}}}}|$")
        ax.set_title(rf"${sym}$ error vs. mesh resolution (smallest $\gamma$ per mesh)")

    for ax in axes.ravel():
        ax.legend(fontsize=8);  ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"Aerodynamic coefficients\n{run_label(rows)}")
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
    fig.suptitle(f"Interpolation error vs. annealed γ_min\n{run_label(rows)}",
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
    fig.suptitle(f"BaryCDI L2(t) colour-graded by γ_min\n{run_label(rows)}",
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
    fig.suptitle(f"IPFP convergence\n{run_label(rows)}", fontsize=13)
    fig.tight_layout()
    f = os.path.join(outdir, f"convergence_{hmode}.png")
    fig.savefig(f, dpi=200, bbox_inches="tight");  plt.close(fig)
    print(f"  wrote {f}")


def plot_gain_heatmap(rows, hmode, outdir):
    """BaryCDI/Linear ratio per (drift, gamma).  <1 means the bridge beats a blend.

    Three things this has to get right, all of which the fixed 0..2 scale and the
    full DRIFT_ORDER row list got wrong once the survey covered many Mach bands:

    * **No empty rows/columns.** Most bands were never run with the exploratory
      SBsquared variants, so those rows were blank white strips carrying no
      information while squeezing the rows that do.
    * **No saturation.** Ratios reach ~3.3 at gamma=1e-2; clipping at vmax=2
      painted 3.24, 3.29 and 3.32 the identical dark red, hiding the ordering
      exactly where the spread is largest.
    * **1.0 is the meaningful centre** (parity with Linear), so the diverging
      colormap is pinned there rather than at the midpoint of the data range.
    """
    from matplotlib.colors import TwoSlopeNorm

    gammas = sorted({r["gamma"] for r in rows})
    M = np.full((len(DRIFT_ORDER), len(gammas)), np.nan)
    for i, drift in enumerate(DRIFT_ORDER):
        for r in _by_drift(rows, drift):
            j = gammas.index(r["gamma"])
            if np.isfinite(r["BaryCDI_l2_mean"]) and np.isfinite(r["Linear_l2_mean"]):
                M[i, j] = r["BaryCDI_l2_mean"] / r["Linear_l2_mean"]

    keep_r = [i for i in range(M.shape[0]) if np.any(np.isfinite(M[i, :]))]
    keep_c = [j for j in range(M.shape[1]) if np.any(np.isfinite(M[:, j]))]
    if not keep_r or not keep_c:
        print(f"  [skip] gain_heatmap {hmode}: no finite cells")
        return
    M = M[np.ix_(keep_r, keep_c)]
    drifts = [DRIFT_ORDER[i] for i in keep_r]
    gammas = [gammas[j] for j in keep_c]

    # Dropping empty rows/columns is not enough: the exploratory SBsquared
    # variants were never run at the six tightest gamma rungs, leaving a solid
    # white block.  Simply dropping those columns would discard the extended
    # anneal that shows null degrading and oblique/ffd hitting their floor.
    # Instead, group columns by WHICH drifts they cover and emit one fully
    # populated heatmap per group -- no white cells, and nothing thrown away.
    groups = {}
    for j in range(M.shape[1]):
        key = tuple(np.isfinite(M[:, j]))
        groups.setdefault(key, []).append(j)
    order = sorted(groups.items(), key=lambda kv: -(sum(kv[0]) * len(kv[1])))

    for n, (pattern, cols) in enumerate(order):
        r_ok = [i for i, ok in enumerate(pattern) if ok]
        if not r_ok:
            continue
        Mg = M[np.ix_(r_ok, cols)]
        d_g = [drifts[i] for i in r_ok]
        g_g = [gammas[j] for j in cols]
        # Suffix every block after the main one so they never overwrite.
        suffix = "" if n == 0 else f"_blk{n}"

        lo, hi = float(np.nanmin(Mg)), float(np.nanmax(Mg))
        norm = TwoSlopeNorm(vmin=min(lo, 0.95), vcenter=1.0, vmax=max(hi, 1.05))

        fig, ax = plt.subplots(figsize=(1.25 * len(g_g) + 3.2,
                                        0.52 * len(d_g) + 1.9))
        im = ax.imshow(Mg, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(len(g_g)), [f"{g:g}" for g in g_g], rotation=45)
        ax.set_yticks(range(len(d_g)), d_g)
        ax.set_xlabel(r"$\gamma_{\min}$")
        for i in range(Mg.shape[0]):
            for j in range(Mg.shape[1]):
                # Label colour from the cell's own luminance rather than a fixed
                # distance from 1.0, which went unreadable on the deep reds.
                r_, g_, b_, _ = im.cmap(norm(Mg[i, j]))
                lum = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
                ax.text(j, i, f"{Mg[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if lum < 0.5 else "black")
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label("BaryCDI / Linear   ($<1$ beats linear)", fontsize=9)
        extra = "" if n == 0 else "   (extended $\\gamma$, subset of drifts)"
        ax.set_title(f"SB gain over linear\n{run_label(rows)}{extra}", fontsize=10)
        fig.tight_layout()
        f = os.path.join(outdir, f"gain_heatmap_{hmode}{suffix}.png")
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
    fig.suptitle(f"Sweep diagnostics\n{run_label(rows)}", fontsize=13)
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
    fig.suptitle(f"Cost vs γ\n{run_label(rows)}", fontsize=13)
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
    marker = {"null": "o", "oblique": "s", "ffd": "^",
              "SBsquared": "D", "SBsquared_exact": "v",
              "SBsquared+ffd": "P", "SBsquared_exact+ffd": "X"}

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
    ax.set_title(f"Cost vs accuracy  (lower-left is better)\n{run_label(rows)}")
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
    fig.suptitle(f"Schrödinger bridge vs. raw FFD registration  (<1 = SB better)\n"
                 f"{run_label(rows)}", fontsize=13)
    fig.tight_layout()
    f = os.path.join(outdir, f"sb_vs_ffd_{hmode}.png")
    fig.savefig(f, dpi=200, bbox_inches="tight");  plt.close(fig)
    print(f"  wrote {f}")


# ─────────────────────────────────────────────────────────────────────────────
#  CSV
# ─────────────────────────────────────────────────────────────────────────────

CSV_COLS = ["hmode", "h", "axis", "aoa", "mach0", "mach1", "n_cells", "drift", "gamma", "n_stages",
            # accuracy — SB vs the two γ-independent baselines
            "BaryCDI_l2_mean", "Linear_l2_mean", "FFD_l2_mean",
            "BaryCDI_linf_max", "Linear_linf_max", "FFD_linf_max",
            "BaryCDI_w2_mean", "Linear_w2_mean", "FFD_w2_mean",
            # scalar physics metric: wave drag error, and the C_L symmetry residual
            "BaryCDI_cd_err_mean", "Linear_cd_err_mean", "FFD_cd_err_mean",
            "BaryCDI_cl_err_mean", "Linear_cl_err_mean", "FFD_cl_err_mean",
            "BaryCDI_cl_abs_max", "Linear_cl_abs_max", "FFD_cl_abs_max",
            "cd_ref_mean", "cl_ref_mean", "ffd_backfilled",
            # transport metrics: shock placement, sharpness, physical admissibility
            "BaryCDI_l2band_mean", "Linear_l2band_mean", "FFD_l2band_mean",
            "BaryCDI_dbeta_mean", "Linear_dbeta_mean", "FFD_dbeta_mean",
            "BaryCDI_sharpness", "Linear_sharpness", "FFD_sharpness",
            "BaryCDI_rh_mean", "Linear_rh_mean", "FFD_rh_mean",
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
    # Two roots, because Mach-axis and AoA-axis runs live in sibling trees.  Both
    # are scanned by default so one table can compare them; the `axis` column keeps
    # them from ever being pooled into a single plot.
    ap.add_argument("--root", nargs="+",
                    default=["outputs/Mach_interpolation/diamond/iso/hessian",
                             "outputs/AoA_interpolation/diamond/iso/hessian"],
                    help="one or more dirs containing <hmode>/…/gmin*/metrics.json")
    ap.add_argument("--out", default=None,
                    help="summary output dir (default <first root>/_sweep)")
    args = ap.parse_args()

    roots = [r for r in args.root if os.path.isdir(r)]
    if not roots:
        raise SystemExit(f"none of these roots exist: {args.root}")
    outdir = args.out or os.path.join(roots[0], "_sweep")
    os.makedirs(outdir, exist_ok=True)

    # Per-axis figure destination: an AoA sweep's figures belong under the
    # AoA_interpolation root, not filed beside the Mach sweeps merely because that
    # root sorted first.  An explicit --out overrides and puts everything together.
    def outdir_for(axis):
        if args.out:
            return outdir
        for r in roots:
            if (axis == "aoa") == ("AoA_interpolation" in r):
                d = os.path.join(r, "_sweep")
                os.makedirs(d, exist_ok=True)
                return d
        return outdir

    rows = []
    for r in roots:
        got = load_runs(r)
        print(f"  {len(got):4d} runs under {r}")
        rows += got
    rows = backfill_ffd_control(rows)
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
        groups = sorted({_pair_key(r) for r in sub_h})
        for gkey in groups:
          sub_a = [r for r in sub_h if _pair_key(r) == gkey]
          axis = gkey[0]
          atag = _pair_tag(sub_a[0])
          # Figures follow their run's own root: an angle sweep is a different
          # experiment and belongs under AoA_interpolation, not filed beside the
          # Mach sweeps just because that root happened to be listed first.
          gdir = outdir_for(axis)
          for h in sorted({r["h"] for r in sub_a if np.isfinite(r["h"])}):
            sub = [r for r in sub_a if r["h"] == h]
            tag = (f"{hmode}_h{h:g}" if len(all_hs) > 1 else hmode) + atag
            print(f"── hessian_mode = {hmode}, {case_label(sub)}, "
                  f"h = {h:g}  ({len(sub)} runs) ──")
            plot_err_vs_gamma(sub, tag, gdir)
            plot_err_vs_t(sub, tag, gdir)
            plot_convergence(sub, tag, gdir)
            plot_gain_heatmap(sub, tag, gdir)
            plot_diagnostics(sub, tag, gdir)
            plot_timing(sub, tag, gdir)
            plot_cost_accuracy(sub, tag, gdir)
            plot_sb_vs_ffd(sub, tag, gdir)

        for gkey in groups:
            sub_a = [r for r in sub_h if _pair_key(r) == gkey]
            axis = gkey[0]
            atag = _pair_tag(sub_a[0])
            gdir = outdir_for(axis)
            print(f"── hessian_mode = {hmode}, {case_label(sub_a)}: "
                  f"cross-resolution ──")
            plot_err_vs_h(sub_a, hmode + atag, gdir)
            plot_aero(sub_a, hmode + atag, gdir)
            for h in sorted({r["h"] for r in sub_a if np.isfinite(r["h"])}):
                sub_ah = [r for r in sub_a if r["h"] == h]
                plot_transport(sub_ah, f"{hmode}{atag}_h{h:g}", gdir)

        # Cross-PAIR view: pools the Mach pairs that every loop above deliberately
        # kept apart, so it must sit outside them.  Grouped by (h, AoA) because
        # those are what have to be held fixed for a Mach-band comparison to mean
        # anything.  Silently skips when only one pair is present, so it is a no-op
        # on the pre-survey tree.
        for h in sorted({r["h"] for r in sub_h if np.isfinite(r["h"])}):
            for aoa in sorted({r["aoa"] for r in sub_h if r["axis"] == "mach"}):
                sub_m = [r for r in sub_h if r["axis"] == "mach"
                         and r["h"] == h and r["aoa"] == aoa]
                if not sub_m:
                    continue
                mtag = (f"{hmode}_h{h:g}" if len(all_hs) > 1 else hmode)
                if abs(aoa) > 1e-9:
                    mtag += f"_aoa{aoa:g}"
                plot_err_vs_mach(sub_m, mtag, outdir_for("mach"))

        write_csv(sub_h, os.path.join(outdir, f"sweep_summary_{hmode}.csv"))

    write_csv(rows, os.path.join(outdir, "sweep_summary.csv"))
    print(f"Done → {outdir}/")


if __name__ == "__main__":
    main()
