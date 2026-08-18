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

# Mach values of the interpolation endpoints + the 9 references, in order.
MACH_ENDPOINTS = ("2.00", "2.50")
MACH_REFS = ("2.05", "2.10", "2.15", "2.20", "2.25", "2.30", "2.35", "2.40", "2.45")


def find_bundle(results_dir: str, mach: str, aoa: str = "0.00") -> str:
    """The one .npz bundle for this Mach, whatever final-time suffix it carries."""
    pattern = os.path.join(results_dir, f"AOA{aoa}_M{mach}_*.npz")
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
    ap.add_argument("--aoa", default="0.00")
    args = ap.parse_args()

    out_path = args.out or f"SB/Config_{args.h_tag}.toml"
    results_dir = os.path.join("results", args.case, args.h_tag)
    mesh_path = os.path.join("meshes", args.case, f"{args.case}_{args.h_tag}.npy")
    if not os.path.isfile(mesh_path):
        raise SystemExit(f"ERROR: mesh not found: {mesh_path}")

    b0 = find_bundle(results_dir, MACH_ENDPOINTS[0], args.aoa)
    b1 = find_bundle(results_dir, MACH_ENDPOINTS[1], args.aoa)
    refs = [find_bundle(results_dir, m, args.aoa) for m in MACH_REFS]

    with open(args.base) as fh:
        lines = fh.readlines()

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
        if re.match(r"^bundle0\s*=", line):
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
