"""
make_config_for_mesh.py — derive an SB Config for a different mesh resolution.

The Euler snapshot name embeds the final time (`…_HLLC_MUSCL_SRK2_t4.00.npz`),
and the solver's stationarity check stops different Mach runs at different times
(e.g. M2.35 → t2.74).  So the bundle paths for a refined mesh cannot be written
by hand — this script GLOBS the actual files produced by the Euler batch and
rewrites the `[run.mach.<case>]` block accordingly.

Everything else in the base config (drift, γ-schedule, density mode, …) is copied
verbatim, line for line, so the sweep sbatch can still `sed` its keys on top.

    uv run python SB/make_config_for_mesh.py --h-tag h0.0125 --out SB/Config_h0.0125.toml
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

# ── Interpolation axes ──────────────────────────────────────────────────────
# axis="mach": AoA fixed, Mach runs 2.00 → 2.50 (the original study).
# axis="aoa" : Mach fixed, AoA runs 0 → 4 deg.  Endpoints + 9 evenly spaced
#              references, matching the 11-frame reconstruction grid, so the
#              reference at t=k/10 is the physical snapshot at that parameter.
# Angle of attack is the axis on which LIFT is a real signal: at AoA=0 the
# reference C_L is mesh-asymmetry noise (~1e-4, and it flips sign between
# meshes), while over 0→4 deg C_L spans ~0 → 0.16.
MACH_ENDPOINTS = ("2.00", "2.50")
MACH_REFS = ("2.05", "2.10", "2.15", "2.20", "2.25", "2.30", "2.35", "2.40", "2.45")

AOA_ENDPOINTS = ("0.00", "4.00")
AOA_REFS = ("0.40", "0.80", "1.20", "1.60", "2.00", "2.40", "2.80", "3.20", "3.60")


def find_bundle(results_dir: str, mach: str, aoa: str = "0.00",
                case: str = "diamond") -> str:
    """The one .npz bundle for this (AoA, Mach), whatever final-time suffix it carries.

    The filename convention is case-dependent: ``Euler/config.py`` prefixes the AoA
    only for diamond, so bump snapshots are ``M2.00_...`` with no ``AOA`` field.
    """
    # Ask Euler itself how it names a snapshot rather than re-deriving the rule.
    # The AoA prefix applies to diamond AND every airfoil (naca0012, rae2822,
    # oneraD, ...) but not to bump; duplicating that list here would silently rot
    # the moment a geometry is added upstream.
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Euler"))
        from config import format_condition_tag           # type: ignore
        stem = format_condition_tag({"Mach": float(mach), "case": case,
                                     "aoa": float(aoa)})
    except Exception:                                      # noqa: BLE001
        stem = f"AOA{aoa}_M{mach}" if case != "bump" else f"M{mach}"
    pattern = os.path.join(results_dir, f"{stem}_*.npz")
    hits = sorted(g for g in glob.glob(pattern) if "summary" not in os.path.basename(g))
    if not hits:
        raise SystemExit(
            f"ERROR: no bundle for M{mach} in {results_dir}\n"
            f"       (pattern: {pattern})\n"
            f"       Run the Euler batch for this mesh first.")
    if len(hits) > 1:
        print(f"  [warn] {len(hits)} bundles for M{mach}, taking newest: "
              f"{os.path.basename(hits[-1])}", file=sys.stderr)
    return hits[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h-tag", required=True, help="e.g. h0.0125")
    ap.add_argument("--case", default="diamond")
    ap.add_argument("--base", default="SB/Config.toml")
    ap.add_argument("--out", default=None, help="default SB/Config_<h-tag>.toml")
    ap.add_argument("--aoa", default="0.00",
                    help="axis=mach: the fixed AoA. Ignored when axis=aoa.")
    ap.add_argument("--axis", choices=("mach", "aoa"), default="mach",
                    help="which parameter the bridge interpolates along")
    ap.add_argument("--mach", default="2.00",
                    help="axis=aoa: the fixed Mach. Ignored when axis=mach.")
    # Arbitrary endpoint pairs.  The 9 references are generated as equispaced
    # INTERIOR points, so the reference at t=k/10 is the physical snapshot at that
    # parameter — the same convention the hardcoded ladders above encode.
    ap.add_argument("--p0", default=None,
                    help="interpolation start (Mach if axis=mach, AoA if axis=aoa)")
    ap.add_argument("--p1", default=None, help="interpolation end")
    args = ap.parse_args()

    def _ladder(p0, p1):
        """(endpoints, 9 interior refs) as %.2f strings, matching bundle names."""
        a, b = float(p0), float(p1)
        pts = [a + (b - a) * k / 10.0 for k in range(11)]
        fmt = [f"{x:.2f}" for x in pts]
        return (fmt[0], fmt[-1]), tuple(fmt[1:-1])

    if (args.p0 is None) != (args.p1 is None):
        # Silently falling back to the 2.00->2.50 default here would build a
        # config for the wrong bridge and label it as the caller's pair.
        raise SystemExit("ERROR: --p0 and --p1 must be given together")

    if args.p0 is not None and args.p1 is not None:
        ends, refs_lad = _ladder(args.p0, args.p1)
        if args.axis == "aoa":
            AOA_ENDPOINTS, AOA_REFS = ends, refs_lad
        else:
            MACH_ENDPOINTS, MACH_REFS = ends, refs_lad

    if args.p0 is None or args.p1 is None:      # module defaults
        MACH_ENDPOINTS, MACH_REFS = globals()["MACH_ENDPOINTS"], globals()["MACH_REFS"]
        AOA_ENDPOINTS, AOA_REFS   = globals()["AOA_ENDPOINTS"], globals()["AOA_REFS"]

    out_path = args.out or f"SB/Config_{args.h_tag}.toml"
    results_dir = os.path.join("results", args.case, args.h_tag)
    mesh_path = os.path.join("meshes", args.case, f"{args.case}_{args.h_tag}.npy")
    if not os.path.isfile(mesh_path):
        raise SystemExit(f"ERROR: mesh not found: {mesh_path}")

    if args.axis == "aoa":
        b0 = find_bundle(results_dir, args.mach, AOA_ENDPOINTS[0], args.case)
        b1 = find_bundle(results_dir, args.mach, AOA_ENDPOINTS[1], args.case)
        refs = [find_bundle(results_dir, args.mach, a, args.case) for a in AOA_REFS]
    else:
        b0 = find_bundle(results_dir, MACH_ENDPOINTS[0], args.aoa, args.case)
        b1 = find_bundle(results_dir, MACH_ENDPOINTS[1], args.aoa, args.case)
        refs = [find_bundle(results_dir, m, args.aoa, args.case) for m in MACH_REFS]

    with open(args.base) as fh:
        lines = fh.readlines()

    # A geometry new to Config.toml has no [run.mach.<case>] block, and the
    # rewrite below would report "0/4 keys" instead of doing anything useful.
    # Clone the diamond block under the new name: every key it holds is
    # overwritten just below anyway, so the clone only has to supply the shape.
    if not any(ln.strip() == f"[run.mach.{args.case}]" for ln in lines):
        src_hdr = "[run.mach.diamond]"
        try:
            i0 = next(i for i, ln in enumerate(lines) if ln.strip() == src_hdr)
        except StopIteration:
            raise SystemExit(f"ERROR: no {src_hdr} in {args.base} to clone from")
        i1 = next((i for i in range(i0 + 1, len(lines))
                   if lines[i].strip().startswith("[")
                   and lines[i].strip().endswith("]")), len(lines))
        block = [f"\n[run.mach.{args.case}]\n"] + lines[i0 + 1:i1]
        lines = lines + block
        print(f"  [new] cloned {src_hdr} -> [run.mach.{args.case}]", file=sys.stderr)

    # Only rewrite inside [run.mach.<case>].  The keys bundle0/bundle1/mesh/
    # ref_bundles appear identically in every case block, so a section-blind
    # regex would stamp diamond paths into [run.mach.bump] as well.
    target_section = f"[run.mach.{args.case}]"
    section = None
    out, i, in_refs, n_rewritten = [], 0, False, 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped
        if in_refs:                                   # skip old ref_bundles body
            if stripped.startswith("]"):
                in_refs = False
            i += 1
            continue
        if section != target_section:
            out.append(line)
            i += 1
            continue
        # The inlet Machs must ALWAYS track the endpoints.  They feed the analytic
        # oblique drift and are what the analysis reads back to identify the pair,
        # so leaving the base config's 2.00/2.50 in place for a 0.80->0.90 bridge
        # would build the drift for the wrong flow and mislabel the run.  On the
        # AoA axis Mach is fixed, so both take the same value.
        if re.match(r"^mach0_inlet\s*=", line):
            m0 = args.mach if args.axis == "aoa" else MACH_ENDPOINTS[0]
            out.append(f'mach0_inlet  = {float(m0)}    # freestream Mach for bundle0\n')
        elif re.match(r"^mach1_inlet\s*=", line):
            m1 = args.mach if args.axis == "aoa" else MACH_ENDPOINTS[1]
            out.append(f'mach1_inlet  = {float(m1)}    # freestream Mach for bundle1\n')
        elif re.match(r"^bundle0\s*=", line):
            out.append(f'bundle0      = "{b0}"\n');  n_rewritten += 1
        elif re.match(r"^bundle1\s*=", line):
            out.append(f'bundle1      = "{b1}"\n');  n_rewritten += 1
        elif re.match(r"^mesh\s*=", line):
            out.append(f'mesh         = "{mesh_path}"\n');  n_rewritten += 1
        elif re.match(r"^ref_bundles\s*=", line):
            out.append("ref_bundles = [\n")
            out.extend(f'    "{r}",\n' for r in refs)
            out.append("]\n")
            in_refs = not line.rstrip().endswith("]")
            n_rewritten += 1
        else:
            out.append(line)
        i += 1

    if n_rewritten != 4:
        raise SystemExit(
            f"ERROR: rewrote {n_rewritten}/4 keys in {target_section} of {args.base}.\n"
            f"       Expected bundle0, bundle1, mesh and ref_bundles — check that the\n"
            f"       section exists and its keys start at column 0.")

    with open(out_path, "w") as fh:
        fh.writelines(out)

    print(f"Wrote {out_path}")
    print(f"  mesh    : {mesh_path}")
    print(f"  bundle0 : {b0}")
    print(f"  bundle1 : {b1}")
    print(f"  refs    : {len(refs)} bundles ({os.path.basename(refs[0])} … "
          f"{os.path.basename(refs[-1])})")


if __name__ == "__main__":
    main()
