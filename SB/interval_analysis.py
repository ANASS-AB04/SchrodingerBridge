"""
interval_analysis.py — mean interpolation error per Mach / AoA INTERVAL.

The sweep plots answer "how does error depend on gamma / on h" for one fixed
interpolation pair.  This answers the complementary question the big grid was
run for: holding gamma and the mesh fixed, how does each method's error depend
on WHICH interval is being interpolated — the Mach band, or the AoA band?

That distinction matters because the three Mach regimes are different physics
(no shock below M=1, detached bow shock to ~1.25, attached oblique above), and
because the AoA axis is a rotation about the leading edge rather than a
downstream sweep.  Pooling them, as a single mean-over-everything would, hides
exactly the effect being looked for.

    uv run python SB/interval_analysis.py                    # tables + figures
    uv run python SB/interval_analysis.py --gamma 0.0001     # one gamma rung
    uv run python SB/interval_analysis.py --metric l2 --csv-only

Outputs (to <root>/_sweep/ by default):
    err_by_mach_interval.csv    per (band, drift, method): mean/std over t
    err_by_aoa_interval.csv     same for the AoA axis
    regime_summary.csv          the three Mach regimes aggregated
    err_vs_mach_<gamma>.png     error vs Mach band, one panel per drift
    err_vs_aoa_<gamma>.png      error vs AoA band
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_analysis as SA                                    # noqa: E402

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402


# The physics bands.  Below M=1 the diamond carries no shock at all; from 1.0 to
# the detachment Mach (~1.25 for the 5 deg half-wedge) the LE shock is a detached
# bow shock; only above that is it the attached oblique shock the pipeline was
# built and validated for.
REGIMES = [("subsonic",   0.00, 1.00, "no shock"),
           ("detached",   1.00, 1.25, "detached bow shock"),
           ("attached",   1.25, 9.99, "attached oblique shock (validated)")]


def regime_of(mach_mid):
    for name, lo, hi, _ in REGIMES:
        if lo <= mach_mid < hi:
            return name
    return "?"


def pick(rows, gamma):
    """Rows at the gamma rung nearest `gamma` within each interpolation pair."""
    out = []
    for key in {SA._pair_key(r) for r in rows}:
        sub = [r for r in rows if SA._pair_key(r) == key]
        for drift in {r["drift"] for r in sub}:
            sd = [r for r in sub if r["drift"] == drift]
            r = SA._nearest_gamma(sd, gamma)
            if r is not None:
                out.append(r)
    return out


def rows_table(rows, metric, axis):
    """One record per (interval, drift, method)."""
    recs = []
    for r in rows:
        if r["axis"] != axis:
            continue
        if axis == "mach":
            lo, hi = r.get("mach0"), r.get("mach1")
        else:
            lo, hi = r.get("aoa0"), r.get("aoa1")
        if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        for method in SA.METHODS:
            series = r.get(f"{method}_{metric}")
            if series is None or not np.size(series):
                continue
            v = np.asarray(series, dtype=float)
            if not np.any(np.isfinite(v)):
                continue
            recs.append({
                "axis": axis, "lo": float(lo), "hi": float(hi),
                "mid": 0.5 * (float(lo) + float(hi)),
                "width": round(float(hi) - float(lo), 6),
                "interval": (f"M{lo:.2f}-{hi:.2f}" if axis == "mach"
                             else f"A{lo:.2f}-{hi:.2f}"),
                "aoa": float(r["aoa"]), "h": float(r["h"]),
                "hmode": r.get("hmode", ""), "n_cells": r.get("n_cells", 0),
                "drift": r["drift"], "gamma": float(r["gamma"]),
                # The driftless kernel can OSCILLATE instead of converging at
                # small gamma (residual jumping O(1) between iterations), and the
                # reported field is then whichever iterate hit the iteration cap.
                # Those points are numerical artifacts, not measurements.
                "ipfp_res": float(r.get("ipfp_res", float("nan"))),
                "converged": bool(r.get("ipfp_converged", False)),
                "method": method,
                "mean": float(np.nanmean(v)), "std": float(np.nanstd(v)),
                "max": float(np.nanmax(v)),
                "regime": regime_of(0.5 * (float(lo) + float(hi))) if axis == "mach" else "-",
            })
    return recs


def write_csv(recs, path, cols):
    if not recs:
        print(f"  [skip] {os.path.basename(path)}: no rows")
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        # the regime summary has no "mid" — sort on whatever ordering key exists
        recs = sorted(recs, key=lambda d: (d.get("aoa", 0.0), d.get("mid", 0.0),
                                           d.get("regime", ""), d.get("drift", ""),
                                           d.get("method", "")))
        for r in recs:
            w.writerow(r)
    print(f"  wrote {path}  ({len(recs)} rows)")


def plot_intervals(recs, axis, gamma, outdir, metric, aoa_filter=None):
    """Error vs interval midpoint, one panel per drift, one line per method."""
    recs = [r for r in recs if r["axis"] == axis]
    if aoa_filter is not None:
        recs = [r for r in recs if abs(r["aoa"] - aoa_filter) < 1e-9]
    if not recs:
        return
    drifts = [d for d in SA.DRIFT_ORDER if any(r["drift"] == d for r in recs)]
    drifts += sorted({r["drift"] for r in recs} - set(drifts))
    if not drifts:
        return

    fig, axes = plt.subplots(1, len(drifts), figsize=(4.5 * len(drifts), 4.4),
                             squeeze=False, sharey=True)
    for ax, drift in zip(axes[0], drifts):
        sub_d = [r for r in recs if r["drift"] == drift]
        # Distinct dash patterns and z-order, because two series can COINCIDE:
        # on the AoA axis the FFD registration recovers ~zero displacement, so
        # FFD lies exactly on Linear.  With identical solid strokes the one drawn
        # first is invisible and the plot silently under-reports a real result.
        style = {"BaryCDI": ("-", 4, 3), "Linear": ("--", 5, 2), "FFD": (":", 6, 1)}
        for method in SA.METHODS:
            pts = sorted((r["mid"], r["mean"]) for r in sub_d
                         if r["method"] == method)
            if pts:
                xs, ys = zip(*pts)
                ls, ms, z = style.get(method, ("-", 4, 1))
                ax.semilogy(xs, ys, ls, marker="o", ms=ms, lw=1.5, zorder=z,
                            mfc="none" if method != "BaryCDI" else None,
                            color=SA.METHOD_COLOR[method], label=method)
        if axis == "mach":
            ax.axvspan(0.0, 1.00, color="tab:red", alpha=0.07)
            ax.axvspan(1.00, 1.25, color="tab:orange", alpha=0.07)
            ax.axvspan(1.25, 9.99, color="tab:green", alpha=0.05)
            ax.set_xlim(min(r["mid"] for r in recs) - 0.06,
                        max(r["mid"] for r in recs) + 0.06)
            ax.set_xlabel("Mach band midpoint")
        else:
            ax.set_xlabel("AoA band midpoint (deg)")
        ax.set_title(f"drift = {drift}", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
    axes[0][0].set_ylabel(rf"mean$_t$  {metric.upper()} error")
    axes[0][-1].legend(fontsize=8)
    what = "Mach" if axis == "mach" else "AoA"
    extra = "" if aoa_filter is None else f", AoA {aoa_filter:g}°"
    sub = ("\nred: no shock (M<1)   orange: detached bow shock   "
           "green: attached oblique shock (validated regime)"
           if axis == "mach" else "")
    h = recs[0].get("h", float("nan"))
    hm = recs[0].get("hmode", "")
    meta = (f"{hm}   ·   h={h:g}" if h == h else hm)
    fig.suptitle(f"{metric.upper()} error vs {what} interval   ·   {meta}"
                 rf"   ·   $\gamma_{{\min}}\approx${gamma:g}{extra}{sub}",
                 fontsize=11)
    plt.tight_layout()
    tag = f"{axis}_{gamma:g}" + ("" if aoa_filter is None else f"_aoa{aoa_filter:g}")
    f = os.path.join(outdir, f"err_vs_{tag}.png")
    plt.savefig(f, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  wrote {f}")


def one_width(recs, want=None):
    """Keep a single interval WIDTH.

    The legacy pair is 2.00->2.50, i.e. 0.50 wide, while the survey bands are
    0.10 wide.  A 5x wider interval is a harder interpolation -- its error sits
    ~3x above its neighbours -- so plotting them on one axis puts a spurious
    spike at the legacy midpoint (M=2.25) and inflates every mean that includes
    it.  Width is a confound, not a signal, so hold it fixed.
    """
    if not recs:
        return recs, None
    widths = {}
    for r in recs:
        widths[r["width"]] = widths.get(r["width"], 0) + 1
    if want is None:
        want = max(widths, key=lambda w: widths[w])       # the modal width
    dropped = {w: n for w, n in widths.items() if abs(w - want) > 1e-9}
    if dropped:
        print(f"  [width] keeping {want:g}-wide intervals; dropping "
              + ", ".join(f"{n} rows at width {w:g}" for w, n in sorted(dropped.items())))
    return [r for r in recs if abs(r["width"] - want) < 1e-9], want


def plot_all_drifts_vs_mach(recs, gamma, outdir, metric, aoa=0.0):
    """All drifts on ONE axes, against the two gamma-independent baselines.

    The per-drift panel figure answers "does the bridge beat Linear here"; this
    answers "which drift, and does any of them beat the registration" -- which
    needs every curve on a shared axis.  Linear and FFD are drawn once, not once
    per panel, because neither depends on the drift: FFD is the raw registration
    used directly as an interpolator, Linear is the plain blend.

    x is the band MIDPOINT, so the 2.50-2.60 bridge is plotted at M = 2.55.
    """
    recs = [r for r in recs if r["axis"] == "mach"
            and abs(r["aoa"] - aoa) < 1e-9]
    if not recs:
        return
    drifts = [d for d in SA.DRIFT_ORDER if any(r["drift"] == d for r in recs)]
    if not drifts:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    # One BaryCDI curve per drift.
    for drift in drifts:
        pts = sorted((r["mid"], r["mean"]) for r in recs
                     if r["drift"] == drift and r["method"] == "BaryCDI")
        if pts:
            xs, ys = zip(*pts)
            ax.semilogy(xs, ys, "-o", ms=4, lw=1.5,
                        color=SA.DRIFT_COLOR.get(drift, "grey"),
                        label=f"SB / {drift}")
            # Ring the points whose IPFP never converged: at small gamma the
            # driftless kernel oscillates (residual O(1) between iterations) and
            # the "error" is just wherever the iteration cap landed.  Without
            # this the reader takes those spikes for a physical effect.
            bad = [(r["mid"], r["mean"]) for r in recs
                   if r["drift"] == drift and r["method"] == "BaryCDI"
                   and r.get("ipfp_res", 0) > 1e-3]
            if bad:
                bx, by = zip(*sorted(bad))
                ax.semilogy(bx, by, "o", ms=13, mfc="none", mew=1.8,
                            color="crimson", zorder=10,
                            label="_nolegend_" if drift != drifts[0] else
                                  "IPFP not converged (res $>10^{-3}$)")
    # The two baselines, once each: take whichever drift carries them (they are
    # identical across drifts by construction, so any row will do).
    for method, style in (("Linear", SA.LINEAR_STYLE), ("FFD", SA.FFD_STYLE)):
        seen, pts = set(), []
        for r in sorted(recs, key=lambda d: d["mid"]):
            if r["method"] == method and r["mid"] not in seen:
                seen.add(r["mid"]);  pts.append((r["mid"], r["mean"]))
        if pts:
            xs, ys = zip(*pts)
            st = dict(style);  st.pop("label", None)
            ax.semilogy(xs, ys, marker="s", ms=4, zorder=5,
                        label=method + (" registration" if method == "FFD" else ""),
                        **st)

    ax.axvspan(0.0, 1.00, color="tab:red", alpha=0.06)
    ax.axvspan(1.00, 1.25, color="tab:orange", alpha=0.06)
    ax.axvspan(1.25, 9.99, color="tab:green", alpha=0.04)
    ax.set_xlim(min(r["mid"] for r in recs) - 0.06,
                max(r["mid"] for r in recs) + 0.06)
    ax.set_xlabel("Mach band midpoint   (2.55 $=$ the $2.50\\!\\to\\!2.60$ bridge)")
    ax.set_ylabel(rf"mean$_t$  {metric.upper()} error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    h = recs[0].get("h", float("nan"))
    hm = recs[0].get("hmode", "")
    meta = (f"{hm}   ·   h={h:g}" if h == h else hm)
    ax.set_title(rf"{metric.upper()} error vs Mach band — all drifts"
                 "\n"
                 rf"{meta}   ·   AoA {aoa:g}$^\circ$   ·   "
                 rf"$\gamma_{{\min}}\approx${gamma:g}"
                 "\nred: no shock   orange: detached bow shock   "
                 "green: attached oblique shock", fontsize=11)
    plt.tight_layout()
    tag = f"{gamma:g}" + ("" if abs(aoa) < 1e-9 else f"_aoa{aoa:g}")
    f = os.path.join(outdir, f"err_vs_mach_alldrifts_{tag}.png")
    plt.savefig(f, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  wrote {f}")


def common_bands(recs):
    """Restrict to intervals covered by EVERY drift, per AoA.

    Without this the drift comparison is confounded: `oblique` has no solution
    below the detachment Mach, so it is missing the 5 lowest bands -- including
    M1.20-1.30, the hardest band that still counts as attached.  Averaging each
    drift over whatever bands it happens to have then credits oblique for the
    bands it skipped.  The tell is the Linear column, which is drift-INDEPENDENT
    by construction: if it differs between two drift rows, those rows are not
    averaging the same set of bands and the comparison is invalid.
    """
    kept = []
    for aoa in {r["aoa"] for r in recs}:
        sub = [r for r in recs if abs(r["aoa"] - aoa) < 1e-9]
        drifts = {r["drift"] for r in sub}
        per_drift = {d: {r["interval"] for r in sub if r["drift"] == d}
                     for d in drifts}
        n_all = len({r["interval"] for r in sub})
        # A drift that was only ever run on one exploratory pair would otherwise
        # drag the intersection down to that single band and destroy the
        # comparison for the drifts that DO span the grid.  Drop those instead,
        # and say so -- a 1-band "comparison" is worse than a narrower drift set.
        sparse = {d for d, iv in per_drift.items() if len(iv) < 0.5 * n_all}
        if sparse:
            print(f"    [aoa {aoa:g}] dropping sparse drift(s) "
                  f"{sorted(sparse)}: <50% band coverage")
            per_drift = {d: iv for d, iv in per_drift.items() if d not in sparse}
            sub = [r for r in sub if r["drift"] not in sparse]
        shared = set.intersection(*per_drift.values()) if per_drift else set()
        kept += [r for r in sub if r["interval"] in shared]
    return kept


def regime_summary(recs, path, restrict=True):
    """Aggregate the Mach bands into the three physical regimes."""
    out = []
    mach = [r for r in recs if r["axis"] == "mach"]
    if restrict:
        before = len({(r["aoa"], r["interval"]) for r in mach})
        mach = common_bands(mach)
        after = len({(r["aoa"], r["interval"]) for r in mach})
        if after < before:
            print(f"  [common bands] {before} -> {after} (aoa, band) combinations, "
                  "so every drift is averaged over the SAME intervals")
    for name, _, _, desc in REGIMES:
        for aoa in sorted({r["aoa"] for r in mach}):
            for drift in sorted({r["drift"] for r in mach}):
                for method in SA.METHODS:
                    v = [r["mean"] for r in mach
                         if r["regime"] == name and r["method"] == method
                         and r["drift"] == drift and abs(r["aoa"] - aoa) < 1e-9]
                    if not v:
                        continue
                    out.append({"regime": name, "description": desc, "aoa": aoa,
                                "drift": drift, "method": method,
                                "n_bands": len(v),
                                "mean": float(np.mean(v)),
                                "std": float(np.std(v)),
                                "min": float(np.min(v)),
                                "max": float(np.max(v))})
    write_csv(out, path, ["regime", "description", "aoa", "drift", "method",
                          "n_bands", "mean", "std", "min", "max"])
    return out


def print_regime_table(summary, metric):
    """The headline numbers, as a terminal table."""
    if not summary:
        return
    for aoa in sorted({r["aoa"] for r in summary}):
        print(f"\n── mean {metric.upper()} by regime, AoA {aoa:g}° "
              "─────────────────────────────")
        drifts = sorted({r["drift"] for r in summary})
        print(f"  {'regime':10s} {'drift':22s} " +
              "".join(f"{m:>12s}" for m in SA.METHODS) + f"{'bands':>7s}")
        for name, _, _, _ in REGIMES:
            for drift in drifts:
                cells, n = [], 0
                for m in SA.METHODS:
                    hit = [r for r in summary
                           if r["regime"] == name and r["drift"] == drift
                           and r["method"] == m and abs(r["aoa"] - aoa) < 1e-9]
                    cells.append(f"{hit[0]['mean']:12.4e}" if hit else f"{'-':>12s}")
                    n = max(n, hit[0]["n_bands"] if hit else 0)
                if n:
                    print(f"  {name:10s} {drift:22s} " + "".join(cells) + f"{n:7d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", nargs="+",
                    default=["outputs/Mach_interpolation/diamond/iso/hessian",
                             "outputs/AoA_interpolation/diamond/iso/hessian"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--gamma", type=float, default=1e-4,
                    help="gamma rung to hold fixed (nearest match per pair)")
    ap.add_argument("--metric", default="l2", choices=("l2", "linf", "w2"))
    ap.add_argument("--hmode", default="eig")
    ap.add_argument("--h", type=float, default=0.025)
    ap.add_argument("--csv-only", action="store_true")
    ap.add_argument("--all-widths", action="store_true",
                    help="do NOT hold the interval width fixed (mixes the "
                         "0.50-wide legacy pair with the 0.10-wide survey bands)")
    ap.add_argument("--width", type=float, default=None,
                    help="interval width to keep (default: the modal width)")
    ap.add_argument("--all-bands", action="store_true",
                    help="do NOT restrict the regime table to bands common to "
                         "every drift (the comparison is then confounded)")
    args = ap.parse_args()

    rows = []
    for r in args.root:
        got = SA.load_runs(r)
        print(f"  {len(got):5d} runs under {r}")
        rows += got
    rows = SA.backfill_ffd_control(rows)
    rows = [r for r in rows if r["hmode"] == args.hmode
            and np.isfinite(r["h"]) and abs(r["h"] - args.h) < 1e-9]
    if not rows:
        raise SystemExit("No runs matched the hmode/h filter")
    print(f"Using {len(rows)} runs (hmode={args.hmode}, h={args.h:g})")

    sel = pick(rows, args.gamma)
    print(f"  {len(sel)} at the gamma rung nearest {args.gamma:g}")

    outdir = args.out or os.path.join(args.root[0], "_sweep")
    os.makedirs(outdir, exist_ok=True)

    mach_recs = rows_table(sel, args.metric, "mach")
    aoa_recs = rows_table(sel, args.metric, "aoa")
    if not args.all_widths:
        mach_recs, _ = one_width(mach_recs, args.width)
        aoa_recs, _ = one_width(aoa_recs, None)
    cols = ["axis", "interval", "lo", "hi", "mid", "width", "regime", "aoa", "h",
            "drift", "gamma", "method", "mean", "std", "max",
            "ipfp_res", "converged"]
    write_csv(mach_recs, os.path.join(outdir, "err_by_mach_interval.csv"), cols)
    write_csv(aoa_recs, os.path.join(outdir, "err_by_aoa_interval.csv"), cols)

    # Two views, because one table cannot serve both questions honestly.
    # (a) restricted to bands every drift covers -> the drifts ARE comparable,
    #     but oblique's absence below detachment confines it to the attached
    #     regime; (b) every band each drift actually ran -> full Mach coverage
    #     including subsonic/detached, but NOT a valid cross-drift comparison.
    summary = regime_summary(mach_recs, os.path.join(outdir, "regime_summary.csv"),
                             restrict=not args.all_bands)
    print("\n=== (a) DRIFT COMPARISON — common bands only ===")
    print_regime_table(summary, args.metric)

    if not args.all_bands:
        full = regime_summary(mach_recs,
                              os.path.join(outdir, "regime_summary_allbands.csv"),
                              restrict=False)
        print("\n=== (b) FULL MACH COVERAGE — each drift over its own bands ===")
        print("    Cross-drift differences here are NOT meaningful: the band sets")
        print("    differ (watch the Linear column, which cannot depend on drift).")
        print_regime_table(full, args.metric)

    if not args.csv_only:
        for aoa in sorted({r["aoa"] for r in mach_recs}):
            plot_intervals(mach_recs, "mach", args.gamma, outdir, args.metric,
                           aoa_filter=aoa)
            plot_all_drifts_vs_mach(mach_recs, args.gamma, outdir, args.metric,
                                    aoa=aoa)
        # An angle sweep is a different experiment and belongs under its own
        # root, not filed beside the Mach sweeps because that root was listed
        # first.  Fall back to `outdir` when only one root was given.
        aoa_out = outdir
        for r in args.root:
            if "AoA_interpolation" in r:
                aoa_out = os.path.join(r, "_sweep")
                os.makedirs(aoa_out, exist_ok=True)
                break
        plot_intervals(aoa_recs, "aoa", args.gamma, aoa_out, args.metric)
    print(f"\nDone → {outdir}/")


if __name__ == "__main__":
    main()
