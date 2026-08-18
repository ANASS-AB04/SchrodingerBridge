"""
Backfill the FFD-as-interpolator baseline into the EXISTING 108-run sweep.

The sweep predates the FFD-interpolation control, but every drift_ffd output dir
cached the registration displacement `beta`, and beta depends only on the two
marginals (i.e. on hessian_mode) — not on gamma or drift.  So we can rebuild the
FFD interpolation exactly, with no IPFP: one registration map per hessian mode.

  T_ffd = x + beta                       (forward,  mu0 -> mu1)
  S_ffd = flow_map(x, beta, sign=-1)     (backward; the old cache has no `alpha`,
                                          so use the reflected inverse flow of the
                                          same frozen field — what the SB pipeline
                                          itself uses for phi_bwd)

Then the SAME CDI reconstruction the SB uses, and L2/Linf vs the same 9 references.
W2 is left as NaN (it needs the jax Sinkhorn; too slow to redo on CPU here).

Writes errors["FFD"] into each drift_ffd/*/metrics.json so sweep_analysis.py picks
it up, and prints a head-to-head table.

RUN IT ON THE CLUSTER (where the sweep outputs live), otherwise a later
`rsync … outputs/ …` pull will overwrite the backfilled files with the cluster's
un-backfilled copies:

    uv run python SB/backfill_ffd_baseline.py

Only needed for runs produced BEFORE the FFD-interpolation control was added —
newer runs compute errors["FFD"] natively in run_mach_interpolation_case.
"""
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "Euler")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

from Euler.jax_fvm.src.mesh import Mesh                     # noqa: E402
from SB import drift as drift_mod                           # noqa: E402
from SB.utils import (build_cdi_interpolators,              # noqa: E402
                      reconstruct_mach_barycentric_cdi)

ROOT = os.path.join(REPO, "outputs/Mach_interpolation/diamond/iso/hessian")
T_REF = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def backfill(hmode):
    run_dirs = sorted(glob.glob(os.path.join(ROOT, hmode, "drift_ffd", "gmin*")))
    if not run_dirs:
        print(f"  no drift_ffd runs for {hmode}");  return None
    with open(os.path.join(run_dirs[0], "metrics.json")) as fh:
        cfg = json.load(fh)["config"]

    betas = sorted(glob.glob(os.path.join(run_dirs[0], "ffd_beta_*.npz")))
    if not betas:
        print(f"  no cached beta in {run_dirs[0]}");  return None
    beta = np.load(betas[0])["beta"]

    mesh = Mesh();  mesh.load_mesh(os.path.join(REPO, cfg["mesh"]))
    bary = np.asarray(mesh.barycenter)
    d0 = np.load(os.path.join(REPO, cfg["bundle0"]));  mach0 = d0["mach"].astype(float);  d0.close()
    d1 = np.load(os.path.join(REPO, cfg["bundle1"]));  mach1 = d1["mach"].astype(float);  d1.close()

    inside = drift_mod.make_inside_body(mesh)
    T_ffd = bary + beta
    S_ffd = np.asarray(drift_mod.flow_map(bary, beta, sign=-1.0, inside_body=inside))
    print(f"  [{hmode}] |beta| max={np.linalg.norm(beta,axis=1).max():.3f}  "
          f"|T-x| mean={np.linalg.norm(T_ffd-bary,axis=1).mean():.4f}")

    from scipy.spatial import Delaunay
    i0, i1, ib, lo, hi = build_cdi_interpolators(bary, mach0, mach1, Delaunay(bary), mesh)

    area = np.asarray(mesh.area)
    l2, linf = [], []
    # reference bundle paths are recorded per-run only implicitly; rebuild from the
    # same Mach ladder the sweep used (M2.05 … M2.45)
    ref_glob = os.path.dirname(os.path.join(REPO, cfg["bundle0"]))
    for t, m in zip(T_REF, ("2.05", "2.10", "2.15", "2.20", "2.25",
                            "2.30", "2.35", "2.40", "2.45")):
        hits = [g for g in glob.glob(os.path.join(ref_glob, f"AOA0.00_M{m}_*.npz"))
                if "summary" not in os.path.basename(g)]
        ref = np.load(sorted(hits)[-1]);  mref = ref["mach"].astype(float);  ref.close()
        pred = reconstruct_mach_barycentric_cdi(float(t), T_ffd, S_ffd, bary,
                                                mach0, mach1, i0, i1, ib, lo, hi)
        diff = pred - mref
        l2.append(float(np.sqrt(np.sum(diff**2 * area)) /
                        max(float(np.sqrt(np.sum(mref**2 * area))), 1e-30)))
        linf.append(float(np.abs(diff).max()))

    for d in run_dirs:                       # write into every ffd run of this hmode
        mp = os.path.join(d, "metrics.json")
        with open(mp) as fh:
            m = json.load(fh)
        if m.get("errors") is None:
            continue
        m["errors"]["FFD"] = {"l2": l2, "linf": linf, "w2": [float("nan")] * 9}
        m.setdefault("timings", {}).setdefault("recon_s", {})["FFD"] = float("nan")
        m["timings"].setdefault("offline_s", {})["FFD"] = float("nan")
        with open(mp, "w") as fh:
            json.dump(m, fh, indent=1)
    print(f"  [{hmode}] wrote FFD errors into {len(run_dirs)} ffd runs")
    return np.array(l2), np.array(linf)


def summarise(hmode, ffd):
    """Head-to-head vs the SB/Linear numbers already in the sweep."""
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, hmode, "drift_*", "gmin*"))):
        with open(os.path.join(d, "metrics.json")) as fh:
            m = json.load(fh)
        if not m.get("errors"):
            continue
        rows.append((m["config"]["reference_drift"], m["config"]["gamma_final"],
                     float(np.mean(m["errors"]["BaryCDI"]["l2"])),
                     float(np.mean(m["errors"]["Linear"]["l2"]))))
    lin = np.mean([r[3] for r in rows])
    print(f"\n  ── {hmode}: mean-t L2 ──   FFD = {ffd[0].mean():.4e}   "
          f"Linear = {lin:.4e}   (FFD/Linear = {ffd[0].mean()/lin:.2f})")
    print(f"  {'drift':>8} {'gamma':>9} {'SB L2':>11} {'SB/FFD':>8} {'SB/Linear':>10}")
    best = None
    for drift, g, sb, _ in sorted(rows, key=lambda r: (r[0], -r[1])):
        ratio = sb / ffd[0].mean()
        print(f"  {drift:>8} {g:>9.0e} {sb:>11.4e} {ratio:>8.2f} {sb/lin:>10.2f}")
        if best is None or sb < best[2]:
            best = (drift, g, sb)
    print(f"  BEST SB: {best[0]} @ γ={best[1]:.0e} → L2={best[2]:.4e}  "
          f"(vs FFD {ffd[0].mean():.4e} → SB/FFD = {best[2]/ffd[0].mean():.2f})")


if __name__ == "__main__":
    for hmode in ("eig", "det"):
        print(f"=== {hmode} ===")
        out = backfill(hmode)
        if out:
            summarise(hmode, out)
