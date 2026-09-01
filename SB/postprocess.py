"""
postprocess.py — figures and animations built from finished SB run directories.

The per-run PNG dump answers "what did frame 7 look like"; it cannot answer "does
the shock actually move", "where does this method put the shock relative to the
reference", or "how does the whole study compare". Those need the fields and the
metrics re-read, which is what this module does.

    uv run python SB/postprocess.py --run <run_dir>          # one run: all artefacts
    uv run python SB/postprocess.py --all                    # every run under outputs/
    uv run python SB/postprocess.py --summary                # cross-case report only

Four artefacts:

  animate      GIF sweeping t=0→1 per method.  Falls back to stitching the existing
               Mach_<method>_t*.png when fields.npz is absent, so it works on runs
               made before field persistence existed.
  compare      reference | BaryCDI | FFD | Linear at one t, ONE shared colour scale,
               shock locus overlaid.  Needs fields.npz.
  overlay      measured shock loci of reference + each method on one axes, with the
               analytic θ-β-M ray.  Needs the transport block's `locus`.
  summary      error and the four transport metrics vs AoA, per drift, per mesh.
               Needs only metrics.json, so it works on everything.

No new dependencies: PIL writes the GIFs (imageio and ffmpeg are not installed, so
MP4 is not an option here).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

METHODS = ["BaryCDI", "Linear", "FFD"]
METHOD_COLOR = {"BaryCDI": "tab:green", "Linear": "black", "FFD": "tab:red",
                "reference": "tab:blue"}


# ─────────────────────────────────────────────────────────────────────────────
#  Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_run(run_dir):
    """(metrics, fields|None) for a finished run directory."""
    with open(os.path.join(run_dir, "metrics.json")) as fh:
        m = json.load(fh)
    fp = os.path.join(run_dir, "fields.npz")
    return m, (np.load(fp, allow_pickle=False) if os.path.isfile(fp) else None)


def _mesh_for(metrics):
    """Load the run's mesh — needed for tripcolor, which wants the connectivity."""
    import sys
    for p in (".", "Euler"):
        if p not in sys.path:
            sys.path.insert(0, p)
    from Euler.jax_fvm.src.mesh import Mesh
    mesh = Mesh()
    mesh.load_mesh(metrics["config"]["mesh"])
    return mesh


def case_tag(metrics):
    """Short label identifying the experiment this run belongs to."""
    c = metrics["config"]
    axis = c.get("interp_axis") or "mach"
    if axis == "aoa":
        return f"AoA {c.get('aoa0_deg', 0):g}->{c.get('aoa1_deg', 0):g}deg @ M{c.get('mach0_inlet')}"
    return f"Mach {c.get('mach0_inlet')}->{c.get('mach1_inlet')} @ AoA {c.get('aoa_deg', 0):g}deg"


# ─────────────────────────────────────────────────────────────────────────────
#  1. Animation
# ─────────────────────────────────────────────────────────────────────────────

def animate_run(run_dir, fps=4, out_name=None):
    """GIF per method sweeping t=0→1.

    Prefers stitching the run's existing per-frame PNGs: they are already rendered
    with consistent axes, it needs no mesh reload, and crucially it works on runs
    that predate fields.npz — which is every run currently on disk.
    """
    from PIL import Image
    made = []
    for method in METHODS:
        frames = sorted(glob.glob(os.path.join(run_dir, f"Mach_{method}_t*.png")))
        if len(frames) < 2:
            continue
        imgs = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in frames]
        out = out_name or os.path.join(run_dir, f"anim_{method}.gif")
        imgs[0].save(out, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0, optimize=True)
        made.append(out)
    return made


# ─────────────────────────────────────────────────────────────────────────────
#  2. Side-by-side comparison at one t
# ─────────────────────────────────────────────────────────────────────────────

def compare_at_t(run_dir, t=0.5, out=None):
    """reference | BaryCDI | FFD | Linear at one t, on ONE shared colour scale.

    The per-run PNGs each carry their own colour scale, so differences between
    methods are invisible when flipping between files — a method can look identical
    to the reference purely because its colourbar rescaled.  One scale across all
    panels is what makes the comparison honest.
    """
    metrics, fields = load_run(run_dir)
    if fields is None:
        print(f"  [skip] compare: no fields.npz in {run_dir}")
        return None
    mesh = _mesh_for(metrics)
    names = [str(x) for x in fields["method_names"]]
    t_arr = fields["t"]
    idx = int(np.argmin(np.abs(t_arr - t)))

    # Reference at this t, if the run carried reference bundles.
    t_ref = metrics.get("t_ref") or []
    ref_field = None
    if t_ref:
        j = int(np.argmin(np.abs(np.asarray(t_ref) - t_arr[idx])))
        rp = metrics["config"].get("ref_bundles")
        if rp is None:                       # not echoed in config; rebuild the path
            rp = []
        if j < len(rp):
            with np.load(rp[j]) as d:
                ref_field = d["mach"].astype(float)

    panels = ([("reference", ref_field)] if ref_field is not None else []) + \
             [(n, fields["mach"][k][idx]) for k, n in enumerate(names)]
    vmin = min(float(np.min(v)) for _, v in panels)
    vmax = max(float(np.max(v)) for _, v in panels)

    tri = mtri.Triangulation(np.asarray(mesh.points)[:, 0],
                             np.asarray(mesh.points)[:, 1],
                             np.asarray(mesh.tris))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.8),
                             squeeze=False)
    for ax, (name, fld) in zip(axes[0], panels):
        tpc = ax.tripcolor(tri, facecolors=np.asarray(fld), cmap="viridis",
                           vmin=vmin, vmax=vmax)
        ax.set_aspect("equal");  ax.set_title(name);  ax.set_xlabel("x")
        _overlay_locus(ax, metrics, name, t_arr[idx])
    axes[0][0].set_ylabel("y")
    fig.colorbar(tpc, ax=axes[0].tolist(), label="M", shrink=0.85)
    fig.suptitle(f"{case_tag(metrics)}   t={t_arr[idx]:.2f}   "
                 f"(shared colour scale {vmin:.3f}–{vmax:.3f})")
    out = out or os.path.join(run_dir, f"compare_t{t_arr[idx]:.2f}.png")
    plt.savefig(out, dpi=160, bbox_inches="tight");  plt.close()
    return out


def _overlay_locus(ax, metrics, name, t_val):
    """Draw the measured shock locus for one method at this t, if recorded."""
    tr = metrics.get("transport") or {}
    if not tr:
        return
    tr_t = np.asarray(tr.get("t_ref", []), dtype=float)
    if tr_t.size == 0:
        return
    j = int(np.argmin(np.abs(tr_t - t_val)))
    src = tr["ref"] if name == "reference" else tr["methods"].get(name, {})
    loci = src.get("locus") or []
    if j < len(loci) and loci[j]:
        pts = np.asarray(loci[j], dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], "-", color="white", lw=2.0, alpha=0.9)
        ax.plot(pts[:, 0], pts[:, 1], "--", color="red", lw=1.2)


# ─────────────────────────────────────────────────────────────────────────────
#  3. Shock-locus overlay
# ─────────────────────────────────────────────────────────────────────────────

def shock_overlay(run_dir, t=0.5, out=None):
    """Measured shock loci of reference + each method on one axes.

    Turns the Δβ number back into geometry: an angle error of 2° is abstract, two
    lines diverging across the domain is not.  The analytic θ-β-M ray is drawn too,
    so the reference's own discretisation error is visible alongside the methods'.
    """
    metrics, _ = load_run(run_dir)
    tr = metrics.get("transport") or {}
    if not tr:
        print(f"  [skip] overlay: no transport block in {run_dir}")
        return None
    tr_t = np.asarray(tr["t_ref"], dtype=float)
    j = int(np.argmin(np.abs(tr_t - t)))

    sources = [("reference", tr["ref"])] + \
              [(n, tr["methods"][n]) for n in METHODS if n in tr["methods"]]
    if not any((s.get("locus") or [None] * (j + 1))[j] for _, s in sources):
        # Runs predating locus capture would otherwise render a plot containing
        # only the analytic ray, which reads as "no method has a shock".
        print(f"  [skip] overlay: {run_dir} has no recorded shock loci "
              f"(run predates locus capture — re-run to get them)")
        return None

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, src in sources:
        loci = src.get("locus") or []
        if j < len(loci) and loci[j]:
            pts = np.asarray(loci[j], dtype=float)
            ang = src.get("angle_deg", [float("nan")] * (j + 1))[j]
            ax.plot(pts[:, 0], pts[:, 1], "o-", ms=3, color=METHOD_COLOR[name],
                    label=f"{name}  β={ang:.2f}°")
    a_exact = tr["ref"].get("angle_analytic_deg", [float("nan")] * (j + 1))[j]
    if np.isfinite(a_exact):
        xs = np.array([0.0, 1.6])
        ax.plot(1.0 + xs, 2.0 + xs * np.tan(np.radians(a_exact)), "k:", lw=1.5,
                label=f"θ-β-M exact  β={a_exact:.2f}°")
    ax.set_xlabel("x");  ax.set_ylabel("y");  ax.set_aspect("equal")
    ax.grid(True, alpha=0.3);  ax.legend(fontsize=9)
    ax.set_title(f"Leading-edge shock locus — {case_tag(metrics)}, t={tr_t[j]:.2f}")
    out = out or os.path.join(run_dir, f"shock_overlay_t{tr_t[j]:.2f}.png")
    plt.savefig(out, dpi=160, bbox_inches="tight");  plt.close()
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  4. Cross-case summary
# ─────────────────────────────────────────────────────────────────────────────

def summary_report(roots, outdir, hmode="eig", gamma=1e-4):
    """Error + the four transport metrics vs AoA, per drift, per mesh.

    The study's one-page answer: does the drift matter, does incidence change that,
    and do the geometric metrics agree with the field norms.  Reads metrics.json
    only, so it needs no fields and works on every run ever produced.
    """
    rows = []
    for root in roots:
        for f in sorted(glob.glob(os.path.join(root, "**", "metrics.json"),
                                  recursive=True)):
            try:
                m = json.load(open(f))
            except Exception:                                   # noqa: BLE001
                continue
            c = m["config"]
            if c.get("hessian_mode") != hmode:
                continue
            if abs(float(c["gamma_final"]) - gamma) > 1e-12:
                continue
            e = m.get("errors") or {};  tr = m.get("transport") or {}
            a = m.get("aero") or {}
            if not tr:
                continue
            axis = c.get("interp_axis") or ("aoa" if "AoA_interp" in f else "mach")
            mn = lambda d, k: (float(np.nanmean(np.asarray(d[k], dtype=float)))
                               if d.get(k) else np.nan)
            rows.append(dict(
                axis=axis, aoa=float(c.get("aoa_deg", 0.0) or 0.0),
                h=float(str(c["mesh"]).split("_h")[1][:-4]),
                drift=c.get("reference_drift", "null"),
                l2={k: mn(e.get(k, {}), "l2") for k in METHODS},
                dbeta=mn(tr["methods"].get("BaryCDI", {}), "angle_err_deg"),
                rh=mn(tr["methods"].get("BaryCDI", {}), "rh_res"),
                l2band=mn(tr["methods"].get("BaryCDI", {}), "l2_band"),
                dcl=(float(np.mean(np.abs(a["methods"]["BaryCDI"]["dC_L"])))
                     if a else np.nan)))
    if not rows:
        print("  [skip] summary: no runs with a transport block matched")
        return None

    drifts = sorted({r["drift"] for r in rows})
    hs = sorted({r["h"] for r in rows})
    panels = [("l2", "mean$_t$ L2 (BaryCDI)"), ("l2band", "L2 on shock band"),
              ("dbeta", r"$|\beta-\beta_{\theta\beta M}|$  [deg]"),
              ("rh", "Rankine-Hugoniot residual"), ("dcl", r"$|\Delta C_L|$")]
    fig, axes = plt.subplots(len(hs), len(panels),
                             figsize=(4.2 * len(panels), 3.7 * len(hs)),
                             squeeze=False)
    for i, h in enumerate(hs):
        for j, (key, ylab) in enumerate(panels):
            ax = axes[i][j]
            for drift in drifts:
                sel = sorted((r for r in rows
                              if r["h"] == h and r["drift"] == drift
                              and r["axis"] == "mach"),
                             key=lambda r: r["aoa"])
                if not sel:
                    continue
                y = [(r["l2"]["BaryCDI"] if key == "l2" else r[key]) for r in sel]
                ax.plot([r["aoa"] for r in sel], y, "o-", label=drift)
            # The AoA-axis case has no meaningful x position on an "AoA" axis —
            # mark it as a horizontal reference line instead of faking a point.
            for r in rows:
                if r["axis"] == "aoa" and r["h"] == h and r["drift"] == "ffd":
                    v = r["l2"]["BaryCDI"] if key == "l2" else r[key]
                    if np.isfinite(v):
                        ax.axhline(v, color="tab:purple", ls="--", lw=1.2,
                                   label="AoA-axis sweep (ffd)")
            ax.set_yscale("log")
            ax.set_xlabel("AoA [deg]");  ax.grid(True, which="both", alpha=0.3)
            if j == 0:
                ax.set_ylabel(f"h = {h:g}\n{ylab}")
            else:
                ax.set_ylabel(ylab)
            if i == 0 and j == len(panels) - 1:
                ax.legend(fontsize=7)
    fig.suptitle(f"AoA study summary — hessian_mode={hmode}, γ={gamma:g}")
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"summary_{hmode}_gmin{gamma:g}.png")
    plt.savefig(out, dpi=160, bbox_inches="tight");  plt.close()
    print(f"  wrote {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────

def process_run(run_dir, t=0.5, do=("animate", "compare", "overlay")):
    made = []
    if "animate" in do:
        made += animate_run(run_dir)
    if "compare" in do:
        r = compare_at_t(run_dir, t);  made += [r] if r else []
    if "overlay" in do:
        r = shock_overlay(run_dir, t);  made += [r] if r else []
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="a single run directory")
    ap.add_argument("--all", action="store_true",
                    help="every run directory under --roots")
    ap.add_argument("--summary", action="store_true", help="cross-case report only")
    ap.add_argument("--roots", nargs="+",
                    default=["outputs/Mach_interpolation/diamond/iso/hessian",
                             "outputs/AoA_interpolation/diamond/iso/hessian"])
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=1e-4)
    ap.add_argument("--hmode", default="eig")
    ap.add_argument("--out", default=None, help="summary output dir")
    args = ap.parse_args()

    roots = [r for r in args.roots if os.path.isdir(r)]

    if args.run:
        for f in process_run(args.run, args.t):
            print(f"  wrote {f}")
    elif args.all:
        dirs = [os.path.dirname(p) for r in roots
                for p in glob.glob(os.path.join(r, "**", "metrics.json"),
                                   recursive=True)]
        print(f"processing {len(dirs)} run directories …")
        for i, d in enumerate(dirs, 1):
            made = process_run(d, args.t)
            print(f"  [{i}/{len(dirs)}] {d}: {len(made)} artefacts")

    if args.summary or args.all:
        outdir = args.out or (os.path.join(roots[0], "_sweep") if roots else ".")
        summary_report(roots, outdir, hmode=args.hmode, gamma=args.gamma)


if __name__ == "__main__":
    main()
