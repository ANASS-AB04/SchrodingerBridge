#!/usr/bin/env python3
"""Measure how much solver time an interpolated snapshot saves as a warm start.

For each target Mach it runs the solver from four initial states -- cold (uniform
freestream), Linear-, FFD- and BaryCDI-seeded -- and reports how many iterations
each needs to reach a given stationarity tolerance.

    uv run python slurm/run_warmstart_bench.py \
        --case diamond --mesh-path meshes/diamond/diamond_h0.025.npy \
        --warmstart outputs/.../warmstart.npz --t 0.1:0.9:0.1

Design note -- why every run goes to tf instead of stopping early.

`Euler/main.py` steps inside a `jax.lax.scan` of STATIC length N.  Early stopping
only flips a flag: the body takes a cheap `lax.cond` branch, but the loop still
makes all N trips.  A skipped step is not free (measured: 3.2 ms against 38.4 ms
for a real one, 8.4 %), so an early-stopped wall time understates the saving and
caps the reportable speedup at roughly 1/0.084 ~ 12x no matter how good the seed
is.  Restructuring the loop would fix that at the cost of touching a solver that
produced ~700 existing bundles.

So this driver does not time early-stopped runs at all.  Every variant runs the
SAME fixed N with `--stationarity-threshold 0`, which means:

  * identical wall time, identical compile, identical trip count -- the scan tail
    confound cannot enter, because nothing stops early;
  * `--save-convergence` records the whole stationarity curve, so
    "iterations to reach tolerance X" is read off afterwards for ANY X, rather
    than committing up front to a threshold that may sit below the solver's
    achievable floor (it does: see --tol);
  * the per-step cost is measured directly as wall_time/N and is the same for
    every variant, so the wall-clock saving is that step count times a measured
    constant.

The cost is that each run is a full solve; the benefit is that the headline
number has no confound in it.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EULER_DIR = REPO_ROOT / "Euler"
for _p in (REPO_ROOT, EULER_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def parse_range(spec: str):
    """start:stop:step | comma list | single value  (same grammar as run_euler_grid)."""
    if ":" in spec:
        start, stop, step = (float(x) for x in spec.split(":"))
        n = int(round((stop - start) / step))
        return [round(start + i * step, 10) for i in range(n + 1)]
    if "," in spec:
        return [round(float(x), 10) for x in spec.split(",") if x.strip()]
    return [round(float(spec), 10)]


def freestream_W(mesh, mach, aoa_deg, cfg, helper, jnp):
    """The state initialize() would build -- used only to price the shared dt."""
    gamma, rho_inf, p_inf = cfg["gamma"], cfg["rho_inf"], cfg["p_inf"]
    c = (gamma * p_inf / rho_inf) ** 0.5
    a = np.deg2rad(aoa_deg)
    prim = jnp.asarray([rho_inf, mach * c * np.cos(a), mach * c * np.sin(a), p_inf])
    return helper.getConserved(jnp.tile(prim[None, :], (mesh.area.shape[0], 1)),
                               gamma=gamma, M=1.0)


def shared_dt(mesh, mach, aoa, cfg, seeds, helper, jnp):
    """One dt for every variant of this Mach.

    `helper.get_dt` takes a global min over cells, so each seed would otherwise
    pick its own dt -- and then "iterations" would be counted on different clocks
    and `stopping_step * dt` would mix the two things being compared.  Taking the
    min over all candidate seeds is CFL-stable for all of them.  Legitimate
    because for SRK2 the discrete fixed point satisfies R(W*)=0 independently of
    dt: pinning dt changes the path, not the destination.
    """
    cands = [freestream_W(mesh, mach, aoa, cfg, helper, jnp)] + list(seeds)
    return min(float(helper.get_dt(jnp.asarray(W), mesh, CFL=cfg["CFL"],
                                   gamma=cfg["gamma"], M=1.0)) for W in cands)


def steps_to_tol(step, rel, tol):
    """First recorded check at which `rel` drops to `tol`; None if it never does."""
    hit = np.flatnonzero(np.asarray(rel) <= tol)
    return int(np.asarray(step)[hit[0]]) if hit.size else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="diamond",
                    help="case name passed to Euler/main.py --case "
                         "(diamond, bump, naca0012, rae2822, ...)")
    ap.add_argument("--mesh-path", required=True)
    ap.add_argument("--warmstart", required=True, help="warmstart.npz written by SB")
    ap.add_argument("--t", default="0.1:0.9:0.1", help="interpolation times to test")
    ap.add_argument("--methods", default=None,
                    help="comma list; default = every method in the warmstart.npz")
    ap.add_argument("--case-name", default=None,
                    help="value for --case when it is not a bench choice "
                         "(naca0012, rae2822, ...); defaults to --case")
    ap.add_argument("--axis", choices=("mach", "aoa"), default="mach",
                    help="which quantity the SB run interpolated along. 'mach' "
                         "sweeps Mach at fixed AoA, 'aoa' sweeps AoA at fixed "
                         "Mach -- warmstart.npz carries mach_target AND "
                         "aoa_target, and the solver must be told the condition "
                         "that matches the seed or the loader's guard rejects it")
    ap.add_argument("--aoa", type=float, default=0.0,
                    help="fixed AoA for --axis mach (ignored for --axis aoa, "
                         "which reads aoa_target from the warmstart file)")
    ap.add_argument("--mach", type=float, default=None,
                    help="fixed Mach for --axis aoa (defaults to the file's "
                         "mach_target, which is constant along an AoA sweep)")
    ap.add_argument("--check-every", type=int, default=25,
                    help="stationarity check interval; also sets the resolution of "
                         "the convergence curve AND its achievable floor (an "
                         "oscillatory residual averages out over a longer block)")
    ap.add_argument("--tol", type=float, default=None,
                    help="tolerance for the headline table. Default: calibrated "
                         "from the cold runs as 3x the best rel any cold run "
                         "reaches, so it is inside the solver's achievable range")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--tag-prefix", default="ws",
                    help="prefix for each run's --out-tag. Set it per reference "
                         "drift (e.g. ws_ffd, ws_oblique): the BaryCDI seed "
                         "depends on the drift, but the variant name does not, "
                         "so two drifts would otherwise write to the same "
                         "results/<case>/<h>/warmstart/<tag>/ and silently "
                         "overwrite each other's A/B")
    ap.add_argument("--out-csv", default="warmstart_bench.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from jax_fvm.src.mesh import Mesh                       # noqa: E402
    import jax_fvm.src.helper as helper                     # noqa: E402
    import jax.numpy as jnp                                 # noqa: E402
    from config import load_default_config, format_h        # noqa: E402

    cfg = load_default_config()
    mesh = Mesh()
    mesh.load_mesh(args.mesh_path)

    ws = np.load(args.warmstart, allow_pickle=False)
    ws_names = [str(s) for s in np.asarray(ws["method_names"]).ravel()]
    methods = ([m.strip() for m in args.methods.split(",")] if args.methods
               else ws_names)
    unknown = [m for m in methods if m not in ws_names]
    if unknown:
        raise SystemExit(f"methods {unknown} not in {args.warmstart} (has {ws_names})")
    t_all = np.asarray(ws["t"], dtype=float).ravel()
    mach_all = np.asarray(ws["mach_target"], dtype=float).ravel()
    aoa_all = (np.asarray(ws["aoa_target"], dtype=float).ravel()
               if "aoa_target" in ws.files else np.full_like(t_all, args.aoa))
    case_name = args.case_name or args.case

    res_root = REPO_ROOT / "results" / args.case / format_h(mesh) / "warmstart"
    variants = ["cold"] + methods
    runs = []

    for t_val in parse_range(args.t):
        j = int(np.argmin(np.abs(t_all - t_val)))
        if abs(t_all[j] - t_val) > 1e-9:
            print(f"[skip] t={t_val} not in warmstart.npz {t_all.tolist()}")
            continue
        # The seed carries the condition it was built for, and the loader refuses
        # a mismatch, so read BOTH targets from the file rather than assuming
        # which one varies.
        if args.axis == "aoa":
            aoa = float(aoa_all[j])
            mach = float(args.mach if args.mach is not None else mach_all[j])
        else:
            mach = float(mach_all[j])
            aoa = float(args.aoa)

        seeds = [np.asarray(ws["conservatives"])[ws_names.index(m), j] for m in methods]
        dt = shared_dt(mesh, mach, aoa, cfg, seeds, helper, jnp)
        print(f"\n=== t={t_val:.2f}  M={mach:.4f}  AoA={aoa:.2f}  dt={dt:.6e} "
              f"(shared by {len(variants)} variants) ===")

        for rep in range(args.repeat):
            for var in variants:
                tag = (f"{args.tag_prefix}_t{t_val:.2f}_{var}"
                       + (f"_r{rep}" if args.repeat > 1 else ""))
                cmd = [
                    "uv", "run", "python", "Euler/main.py",
                    "--case", case_name, "--mach", f"{mach:.10g}",
                    "--aoa", f"{aoa:.10g}", "--mesh-path", args.mesh_path,
                    "--flux", "HLLC", "--time-scheme", "SRK2",
                    "--reconstruction", "MUSCL",
                    # threshold 0 => run the full N; the curve carries the answer
                    "--stationarity-threshold", "0",
                    "--stationarity-check-every", str(args.check_every),
                    "--dt", f"{dt:.12e}",
                    "--save-convergence", "--summary",
                    "--out-tag", tag, "--quiet",
                ]
                if var != "cold":
                    cmd += ["--init-state", args.warmstart,
                            "--init-method", var, "--init-t", f"{t_val:.10g}"]
                print(f"  [{var:8s}] {tag}")
                if args.dry_run:
                    print("    " + " ".join(cmd))
                    continue
                r = subprocess.run(cmd, cwd=REPO_ROOT)
                if r.returncode != 0:
                    print(f"  [WARN] {tag} FAILED (rc={r.returncode}) — continuing")
                    continue
                runs.append({"t": t_val, "mach": mach, "aoa": aoa, "variant": var,
                             "repeat": rep, "dt": dt, "tag": tag})

    if args.dry_run:
        return

    # ── collect ───────────────────────────────────────────────────────────────
    rows = []
    for run in runs:
        d = res_root / run["tag"]
        conv = sorted(d.glob("convergence_*.npz"))
        summ = sorted(d.glob("summary_*.json"))
        if not conv:
            print(f"  [WARN] no convergence trace in {d}")
            continue
        c = np.load(conv[0])
        row = dict(run)
        row["n_steps_total"] = int(np.asarray(c["n_steps_total"]).ravel()[0])
        row["wall_time_s"] = float(np.asarray(c["wall_time_s"]).ravel()[0])
        row["per_step_ms"] = 1e3 * row["wall_time_s"] / max(1, row["n_steps_total"])
        row["rel_min"] = float(np.min(c["rel"]))
        row["_step"], row["_rel"] = c["step"], c["rel"]
        if summ:
            s = json.loads(summ[0].read_text())
            for k in ("cd", "cl", "deltaS", "init_repair_count",
                      "init_rho_min", "init_p_min"):
                row[k] = s.get(k)
        rows.append(row)

    if not rows:
        raise SystemExit("no runs produced a convergence trace")

    # Calibrate the tolerance INSIDE the reachable range.  The solver has a
    # residual floor (MUSCL limiter chatter), so a tolerance below it would be
    # reached by nobody and the table would be all None.
    if args.tol is not None:
        tol = args.tol
    else:
        cold_floor = max(r["rel_min"] for r in rows if r["variant"] == "cold")
        tol = 3.0 * cold_floor
        print(f"\nCalibrated tolerance = 3 x (worst cold floor {cold_floor:.3e}) "
              f"= {tol:.3e}")

    # Per-step cost is shared across the variants of one t, NOT taken per run.
    # Every variant executes the identical program (same mesh, same N, same dt,
    # same static args), so any spread is measurement noise -- and it is biased,
    # not random: the scan's XLA compile happens inside the timed region, and the
    # first run at each t pays it while the rest are served from
    # JAX_COMPILATION_CACHE_DIR.  Cold runs first, so charging each run its own
    # wall time would systematically inflate the cold baseline and manufacture a
    # speedup out of a cache hit.  The minimum is the compile-free estimate.
    for r in rows:
        r["tol"] = tol
        r["steps_to_tol"] = steps_to_tol(r.pop("_step"), r.pop("_rel"), tol)
    by_t = {}
    for r in rows:
        by_t.setdefault(r["t"], []).append(r["per_step_ms"])
    for r in rows:
        r["per_step_ms_raw"] = r["per_step_ms"]
        r["per_step_ms"] = min(by_t[r["t"]])
        r["time_to_tol_s"] = (None if r["steps_to_tol"] is None
                              else r["steps_to_tol"] * r["per_step_ms"] / 1e3)

    cold_by_t = {(r["t"], r["repeat"]): r for r in rows if r["variant"] == "cold"}
    for r in rows:
        c = cold_by_t.get((r["t"], r["repeat"]))
        r["speedup"] = (None if not (c and r["steps_to_tol"] and c["steps_to_tol"])
                        else c["steps_to_tol"] / r["steps_to_tol"])

    cols = ["t", "mach", "aoa", "variant", "repeat", "dt", "n_steps_total", "tol",
            "steps_to_tol", "speedup", "per_step_ms", "per_step_ms_raw",
            "time_to_tol_s", "wall_time_s", "rel_min", "cd", "cl", "deltaS",
            "init_repair_count", "init_rho_min", "init_p_min", "tag"]
    out = REPO_ROOT / args.out_csv
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["t"], r["variant"], r["repeat"])):
            w.writerow(r)
    print(f"\nWrote {out}  ({len(rows)} runs)")

    print(f"\n{'t':>5} {'M':>7} {'variant':>9} {'steps→tol':>10} {'speedup':>8} "
          f"{'C_D':>9} {'repair':>7}")
    for r in sorted(rows, key=lambda r: (r["t"], r["variant"])):
        sp = "  n/a" if r["speedup"] is None else f"{r['speedup']:6.2f}x"
        st = "   never" if r["steps_to_tol"] is None else f"{r['steps_to_tol']:8d}"
        cd = "      n/a" if r.get("cd") is None else f"{r['cd']:9.5f}"
        print(f"{r['t']:5.2f} {r['mach']:7.4f} {r['variant']:>9} {st} {sp:>8} "
              f"{cd} {str(r.get('init_repair_count')):>7}")


if __name__ == "__main__":
    main()
