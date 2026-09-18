"""
ref_aero_curve.py — reference C_D/C_L vs Mach straight from the Euler bundles.

Computed exactly as SB's compute_aero_coefficients does (inviscid ∮p·n on the
mesh the bundles live on, q∞ = ½γp∞M²), so it shows what every SB run will use
as its reference.  Consecutive Mach values whose C_D differs by more than
--jump percent are flagged: on a well-resolved corpus C_D(M) is smooth, and a
flag is the signature of an under-resolved leading-edge bow shock (see the
"Rounded-LE airfoils" note in CLAUDE.md).

    JAX_PLATFORMS=cpu uv run python SB/ref_aero_curve.py --case naca0012 --h-tag h0.025le7
    JAX_PLATFORMS=cpu uv run python SB/ref_aero_curve.py --case naca0012 --h-tag h0.025 h0.025le7 --mach 1.6 2.1
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "Euler"))
sys.path.insert(0, os.path.join(_HERE, ".."))
import jax.numpy as jnp                                        # noqa: E402
from jax_fvm.src.mesh import Mesh                              # noqa: E402
from jax_fvm.src import helper                                 # noqa: E402


def curve(root, case, h_tag, aoa, mach_lo, mach_hi, gamma=1.4, p_inf=1.0):
    mesh = Mesh()
    mesh.load_mesh(os.path.join(root, "meshes", case, f"{case}_{h_tag}.npy"))
    L_ref = float(mesh.metadata.get("obstacle_length", 1.0))
    out = {}
    for f in glob.glob(os.path.join(root, "results", case, h_tag, f"AOA{aoa:.2f}_M*.npz")):
        m = re.search(r"_M([0-9.]+)_", os.path.basename(f))
        if not m or not (mach_lo - 1e-9 <= float(m.group(1)) <= mach_hi + 1e-9):
            continue
        d = np.load(f)
        if int(np.asarray(d["n_cells"]).ravel()[0]) != len(mesh.tris):
            print(f"  [skip] {os.path.basename(f)}: n_cells does not match the mesh")
            continue
        M = float(np.asarray(d["mach_in"]).ravel()[0])
        cd, cl = helper.get_force_coefficients_from_pressure(
            jnp.asarray(np.asarray(d["primitives"], dtype=float)[:, 3]), mesh,
            0.5 * gamma * p_inf * M ** 2, L_ref)
        out[round(M, 4)] = (float(cd), float(cl))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--h-tag", nargs="+", required=True, help="one or more tags, shown side by side")
    ap.add_argument("--aoa", type=float, default=0.0)
    ap.add_argument("--mach", type=float, nargs=2, default=(0.0, 9.9), metavar=("LO", "HI"))
    ap.add_argument("--jump", type=float, default=3.0, help="flag |ΔC_D| above this percent")
    ap.add_argument("--root", default=os.path.join(_HERE, ".."))
    a = ap.parse_args()

    curves = {t: curve(a.root, a.case, t, a.aoa, *a.mach) for t in a.h_tag}
    machs = sorted(set().union(*curves.values()))
    if not machs:
        raise SystemExit("no bundles found")
    print(f"{a.case}  AoA {a.aoa:g}   C_D / C_L per tag   (* = |ΔC_D| > {a.jump:g}% vs previous Mach)")
    print(f"{'M':>6}" + "".join(f"  {t:>28s}" for t in a.h_tag))
    prev = {t: None for t in a.h_tag}
    n_flag = {t: 0 for t in a.h_tag}
    for M in machs:
        line = f"{M:6.2f}"
        for t in a.h_tag:
            if M not in curves[t]:
                line += f"  {'—':>28s}"
                continue
            cd, cl = curves[t][M]
            flag = ""
            if prev[t] is not None and abs(cd - prev[t]) / abs(prev[t]) * 100 > a.jump:
                flag = f"* {100 * (cd - prev[t]) / prev[t]:+.1f}%"
                n_flag[t] += 1
            line += f"  {cd:9.5f} {cl:+9.5f} {flag:>8s}"
            prev[t] = cd
        print(line)
    print("flags: " + ", ".join(f"{t}: {n_flag[t]}" for t in a.h_tag))


if __name__ == "__main__":
    main()
