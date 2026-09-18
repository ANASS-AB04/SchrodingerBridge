#!/usr/bin/env python3
"""Cross-band summary figures for the results deck (bump + naca0012).

    uv run python SB/deck_figs.py

`interval_analysis.py` plots one metric at a time and `sweep_analysis.py` plots
one band at a time, so neither produces the two figures a deck needs: L2 AND
Linf on one axis, and the aerodynamic coefficients across the whole Mach sweep
(its `aero_*.png` is per band, with a mesh-convergence panel that holds a single
point when only one h was run).  Everything here is read from the
`sweep_summary_eig.csv` that `sweep_analysis.py` already wrote -- no GPU, no
re-run.

Also folds the warm-start tolerance sweeps (`outputs/warmstart/wstol_*.csv`,
written by Euler/postproc/warmstart_analysis.py) into one recap panel.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "deck"
GAMMA = "0.0001"

# The bands the physics is actually validated on.  Below M=1 there is no shock
# at all and the Hessian marginal has nothing to transport; 1.0-1.25 is a
# detached bow shock.  Shaded, not dropped -- the reader should see them.
REGIMES = [(0.0, 1.0, "#d62728", "no shock"),
           (1.0, 1.25, "#ff7f0e", "detached bow shock"),
           (1.25, 9.9, "#2ca02c", "attached oblique shock")]

STYLE = {"BaryCDI": dict(color="tab:green", marker="o", ls="-",  label="BaryCDI (SB)"),
         "Linear":  dict(color="black",     marker="o", ls="--", label="Linear", mfc="none"),
         "FFD":     dict(color="tab:red",   marker="o", ls=":",  label="FFD", mfc="none")}


def load(case):
    p = ROOT / f"outputs/Mach_interpolation/{case}/iso/hessian/_sweep/sweep_summary_eig.csv"
    return [r for r in csv.DictReader(open(p)) if r["gamma"] == GAMMA]


def series(rows, aoa, drift, col):
    """(mid, value) sorted by band midpoint, skipping blanks."""
    out = []
    for r in rows:
        if r["aoa"] != aoa or r["drift"] != drift or not r[col]:
            continue
        out.append((0.5 * (float(r["mach0"]) + float(r["mach1"])), float(r[col])))
    return np.array(sorted(out)).T if out else (np.array([]), np.array([]))


def best_drift(rows, aoa):
    """The drift with the lowest band-mean BaryCDI L2 -- reported, not assumed."""
    agg = defaultdict(list)
    for r in rows:
        if r["aoa"] == aoa and r["BaryCDI_l2_mean"]:
            agg[r["drift"]].append(float(r["BaryCDI_l2_mean"]))
    return min(agg, key=lambda d: np.mean(agg[d]))


def shade(ax):
    for lo, hi, c, _ in REGIMES:
        ax.axvspan(lo, hi, color=c, alpha=0.07, lw=0)


def panel(ax, rows, aoa, drift, cols, title, ylabel, logy=True):
    shade(ax)
    for m, col in cols.items():
        x, y = series(rows, aoa, drift, col)
        if len(x):
            ax.plot(x, y, ms=4, **STYLE[m])
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Mach band midpoint")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=.3, which="both")
    ax.set_xlim(0.78, 3.02)


def fig_error(case, rows, aoa, tag):
    drift = best_drift(rows, aoa)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    panel(axes[0], rows, aoa, drift,
          {m: f"{m}_l2_mean" for m in STYLE}, r"$L^2$ error", r"mean$_t$  $L^2$")
    panel(axes[1], rows, aoa, drift,
          {m: f"{m}_linf_max" for m in STYLE}, r"$L^\infty$ error", r"max$_t$  $L^\infty$")
    axes[0].legend(fontsize=9)
    fig.suptitle(f"{case} — Mach-interval sweep · h=0.025 · "
                 rf"$\gamma_{{\min}}=10^{{-4}}$ · AoA {float(aoa):.0f}° · drift = {drift}"
                 "\nred: no shock   orange: detached bow shock   green: attached oblique shock",
                 fontsize=10, y=0.995, va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = OUT / f"sweep_l2linf_{tag}.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print("wrote", p)


def fig_aero(case, rows, aoa, tag):
    drift = best_drift(rows, aoa)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    panel(axes[0], rows, aoa, drift,
          {m: f"{m}_cd_err_mean" for m in STYLE},
          r"drag: $\overline{|C_D-C_D^{\rm ref}|}$", r"mean$_t$  $|\Delta C_D|$")
    panel(axes[1], rows, aoa, drift,
          {m: f"{m}_cl_err_mean" for m in STYLE},
          r"lift: $\overline{|C_L-C_L^{\rm ref}|}$", r"mean$_t$  $|\Delta C_L|$")
    # The reference level the errors should be read against.
    for ax, col in zip(axes, ("cd_ref_mean", "cl_ref_mean")):
        x, y = series(rows, aoa, drift, col)
        if len(x):
            ax.plot(x, np.abs(y), color="grey", ls="-", lw=1, alpha=.6,
                    label=f"|{col[:2].upper()} ref|")
            ax.legend(fontsize=8)
    fig.suptitle(f"{case} — aerodynamic coefficients across the Mach sweep · "
                 rf"$\gamma_{{\min}}=10^{{-4}}$ · AoA {float(aoa):.0f}° · drift = {drift}",
                 fontsize=10, y=0.995, va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = OUT / f"sweep_aero_{tag}.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print("wrote", p)


def fig_speedup(prefix="ws_ffd", t_mid=0.5):
    """Speedup FACTOR collapses with the tolerance; iterations SAVED does not."""
    p = ROOT / f"outputs/warmstart/wstol_{prefix}.csv"
    if not p.exists():
        print("skip speedup:", p, "missing")
        return
    rows = [r for r in csv.DictReader(open(p)) if float(r["t"]) == t_mid]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for v in ("BaryCDI", "FFD", "Linear"):
        sel = [r for r in rows if r["variant"] == v and r["speedup"]]
        if not sel:
            continue
        tol = [float(r["tol"]) for r in sel]
        axes[0].loglog(tol, [float(r["speedup"]) for r in sel], ms=4, **STYLE[v])
        axes[1].semilogx(tol, [int(r["steps_saved"]) for r in sel], ms=4, **STYLE[v])
    for ax, yl, ti in ((axes[0], "speedup vs cold start",
                        "Speedup FACTOR depends on where you stop"),
                       (axes[1], "solver iterations saved",
                        "Iterations SAVED is the stable quantity")):
        ax.set_xlabel("stationarity tolerance"); ax.set_ylabel(yl)
        ax.set_title(ti); ax.invert_xaxis(); ax.grid(alpha=.3, which="both")
    axes[1].set_ylim(bottom=0)
    axes[0].legend(fontsize=9)
    fig.suptitle(f"Warm-start benefit · diamond h=0.025 · M 2.00→2.50 · t={t_mid} · "
                 f"seeds from the {prefix.replace('ws_', '')} run", fontsize=10, y=0.995, va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    q = OUT / "speedup_recap.png"
    fig.savefig(q, dpi=140); plt.close(fig)
    print("wrote", q)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bump = load("bump")
    fig_error("bump", bump, "0.0", "bump")
    fig_aero("bump", bump, "0.0", "bump")
    naca = load("naca0012")
    for aoa, tag in (("0.0", "naca_aoa0"), ("2.0", "naca_aoa2")):
        fig_error("naca0012", naca, aoa, tag)
        fig_aero("naca0012", naca, aoa, tag)
    fig_speedup()


if __name__ == "__main__":
    main()
