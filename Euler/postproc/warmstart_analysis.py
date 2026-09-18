#!/usr/bin/env python3
"""How does the warm-start speedup depend on the tolerance you ask for?

    uv run python Euler/postproc/warmstart_analysis.py \
        --root results/diamond/h0.025/warmstart --prefix ws_ffd

Reads the `convergence_*.npz` traces a benchmark run leaves behind and re-reads
"iterations to reach tolerance X" for a whole sweep of X.  No solver runs: the
traces already contain the entire stationarity history, which is why the
benchmark records them instead of committing to one threshold.

Why this matters.  The speedup RATIO is not a property of the seed -- it is a
property of the tolerance you stop at, and it varies by more than an order of
magnitude across the reachable range:

    tol      cold  warm   speedup
    1.0e-4   1250    25     50.0x
    3.0e-5   3075   275     11.2x
    1.0e-5   4575  1775      2.6x
    8.0e-6   5150  2150      2.4x   <- at the solver's residual floor

Quoting a single number without its tolerance is meaningless.  What IS stable is
the iterations SAVED (~2800-3000 in that example, at every tolerance), so the
tables below report both and the plot shows the ratio collapsing.

The solver does not converge to machine zero -- MUSCL limiter chatter leaves
`rel` on a floor -- so tolerances below that floor are unreachable by ANY
variant, cold included.  The sweep is clipped to the reachable range and the
floor is reported.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

TAG_RE = re.compile(r"^(?P<prefix>.+)_t(?P<t>[0-9]*\.?[0-9]+)_(?P<var>[A-Za-z0-9]+)"
                    r"(?:_r(?P<rep>\d+))?$")


def load_traces(root: Path, prefix: str | None):
    """{(t, variant, rep): (step, rel)} from every convergence_*.npz under root."""
    out = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = TAG_RE.match(d.name)
        if not m or (prefix and m["prefix"] != prefix):
            continue
        files = sorted(d.glob("convergence_*.npz"))
        if not files:
            continue
        c = np.load(files[0])
        out[(round(float(m["t"]), 4), m["var"], int(m["rep"] or 0))] = (
            np.asarray(c["step"]), np.asarray(c["rel"]),
            float(np.asarray(c["wall_time_s"]).ravel()[0]),
            int(np.asarray(c["n_steps_total"]).ravel()[0]))
    return out


def steps_to_tol(step, rel, tol):
    hit = np.flatnonzero(rel <= tol)
    return int(step[hit[0]]) if hit.size else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="e.g. results/diamond/h0.025/warmstart")
    ap.add_argument("--prefix", default=None,
                    help="tag prefix to select one drift, e.g. ws_ffd")
    ap.add_argument("--n-tol", type=int, default=24)
    ap.add_argument("--baseline", default="cold")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-png", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    traces = load_traces(root, args.prefix)
    if not traces:
        raise SystemExit(f"no convergence traces under {root} (prefix={args.prefix})")

    ts = sorted({k[0] for k in traces})
    variants = sorted({k[1] for k in traces})
    print(f"{len(traces)} traces | t = {ts} | variants = {variants}")

    # Reachable range: nobody can go below the worst floor; nothing above the
    # loosest starting value is a convergence test at all.
    floor = max(float(r.min()) for _, r, _, _ in traces.values())
    start = min(float(r[0]) for _, r, _, _ in traces.values() if r.size)
    lo, hi = floor * 1.05, start
    if not (hi > lo):
        raise SystemExit(f"no reachable tolerance band (floor {floor:.3e}, start {start:.3e})")
    tols = np.geomspace(hi, lo, args.n_tol)
    print(f"residual floor = {floor:.3e}   sweep {hi:.3e} -> {lo:.3e}\n")

    rows = []
    for tol in tols:
        for t in ts:
            base = traces.get((t, args.baseline, 0))
            if base is None:
                continue
            nb = steps_to_tol(base[0], base[1], tol)
            for v in variants:
                tr = traces.get((t, v, 0))
                if tr is None:
                    continue
                n = steps_to_tol(tr[0], tr[1], tol)
                rows.append({
                    "tol": tol, "t": t, "variant": v, "steps_to_tol": n,
                    "baseline_steps": nb,
                    "speedup": (nb / n) if (n and nb) else None,
                    "steps_saved": (nb - n) if (n is not None and nb is not None) else None,
                    "per_step_ms": 1e3 * tr[2] / max(1, tr[3]),
                })

    out_csv = Path(args.out_csv or (root / "warmstart_tolerance_sweep.csv"))
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}  ({len(rows)} rows)")

    # ── console table at the mid-range t, where a seed has to be genuinely good
    t_mid = ts[len(ts) // 2]
    print(f"\n=== t = {t_mid} (mid-range) ===")
    hdr = f"{'tol':>10} {args.baseline:>7}"
    for v in variants:
        if v != args.baseline:
            hdr += f" | {v:>8} {'x':>7} {'saved':>7}"
    print(hdr)
    print("-" * len(hdr))
    for tol in tols[::max(1, len(tols) // 12)]:
        sel = [r for r in rows if r["t"] == t_mid and abs(r["tol"] - tol) < 1e-18]
        b = next((r["baseline_steps"] for r in sel), None)
        line = f"{tol:10.2e} {str(b):>7}"
        for v in variants:
            if v == args.baseline:
                continue
            r = next((r for r in sel if r["variant"] == v), None)
            n = r["steps_to_tol"] if r else None
            sp = r["speedup"] if r else None
            sv = r["steps_saved"] if r else None
            line += (f" | {str(n):>8} {(f'{sp:.1f}' if sp else '-'):>7} "
                     f"{(str(sv) if sv is not None else '-'):>7}")
        print(line)

    # ── the headline: ratio is tolerance-dependent, saved iterations are not
    print("\n=== stability of each summary, over the reachable tolerance band ===")
    print(f"{'variant':>9} {'speedup min':>12} {'speedup max':>12} {'ratio':>7} "
          f"{'saved min':>10} {'saved max':>10} {'spread':>7}")
    for v in variants:
        if v == args.baseline:
            continue
        sp = [r["speedup"] for r in rows if r["variant"] == v and r["speedup"]]
        sv = [r["steps_saved"] for r in rows if r["variant"] == v
              and r["steps_saved"] is not None]
        if not sp or not sv:
            continue
        print(f"{v:>9} {min(sp):12.1f} {max(sp):12.1f} {max(sp)/min(sp):6.1f}x "
              f"{min(sv):10d} {max(sv):10d} {max(sv)/max(1,min(sv)):6.1f}x")
    print("\nA large 'ratio' with a small 'spread' means the speedup FACTOR is an "
          "artefact of\nwhere you stop, while the iterations SAVED is the real, "
          "quotable quantity.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, v in enumerate(variants):
        if v == args.baseline:
            continue
        sel = [r for r in rows if r["variant"] == v and r["t"] == t_mid
               and r["speedup"]]
        if not sel:
            continue
        axes[0].loglog([r["tol"] for r in sel], [r["speedup"] for r in sel],
                       "o-", color=colors[i], label=v, ms=3)
        axes[1].semilogx([r["tol"] for r in sel], [r["steps_saved"] for r in sel],
                         "o-", color=colors[i], label=v, ms=3)
    axes[0].axvline(floor, ls=":", c="k", lw=1)
    axes[0].set_xlabel("stationarity tolerance")
    axes[0].set_ylabel("speedup vs cold")
    axes[0].set_title(f"Speedup FACTOR is tolerance-dependent (t={t_mid})")
    axes[0].invert_xaxis()
    axes[0].legend()
    axes[0].grid(alpha=.3)
    axes[1].axvline(floor, ls=":", c="k", lw=1, label="residual floor")
    axes[1].set_xlabel("stationarity tolerance")
    axes[1].set_ylabel("iterations saved")
    axes[1].set_title("Iterations SAVED is stable")
    axes[1].invert_xaxis()
    axes[1].legend()
    axes[1].grid(alpha=.3)
    fig.tight_layout()
    out_png = Path(args.out_png or (root / "warmstart_tolerance_sweep.png"))
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
