---
name: naca0012-bow-shock-underresolved
description: naca0012 reference C_D/C_L jump with Mach because the nose bow shock spans 1.3–5 cells at h0.025 — affects every rounded-LE airfoil
metadata:
  type: project
---

The naca0012 HR reference snapshots (h0.025, `h_min_factor=5`) have a **sawtooth
C_D(M)**: smooth decay, then a +7–18% jump (AoA 0 at M1.69, 2.04, 2.81; AoA 2 at 1.71,
1.81, 2.07, 2.49), and C_L ≠ 0 at AoA 0 (−0.006…+0.011) flipping sign at the same
Mach values. Found 2026-09-11 when the per-run aero plot "exploded at t=0.9".

**Cause:** detached bow shock at the rounded LE. Nose radius 0.016c, physical standoff
(Billig) 0.038c at M1.6 → 0.010c at M3, nose cells ~0.0078 → the subsonic pocket is
1.3–5 cells, i.e. about the numerical shock width. The captured shock jumps from one cell
row to the next as M changes; low-C_D states show a stagnation-line shock spike, and the
asymmetric mesh (`symmetric=False` in airfoils.py) makes them lopsided → fake C_L.
Ruled out: Euler array chunk boundaries (jumps are mid-chunk, same task/GPU/code), and
end-time overshoot (loop runs exactly N=int(tf/dt)+1 steps). Diamond and bump are clean
(sharp LE / weak shocks).

**Fix implemented 2026-09-11 (uncommitted, not yet validated):** nose-refined tags
`h0.025leN` — Euler solves on a nested solver mesh and exports bundles on the standard
mesh (details in CLAUDE.md "Rounded-LE airfoils"). Chosen over refining SB's mesh
because SB's explicit kernel would cost ~21x per run. Verified locally: plain meshes
unchanged, local CPU reruns reproduce the corpus C_D/C_L to 5 decimals (so the jump is
deterministic). The le7 solves were stopped before finishing; validation is the cluster
smoke test `CASE=naca0012 H_TAG=h0.025le7 sbatch --array=11,15 run_euler_grid_big.sbatch`
(tasks 11/15 = M1.68–1.75 / M2.00–2.07 at AoA 0), checked with
`SB/ref_aero_curve.py --h-tag h0.025 h0.025le7`. If it passes: make h0.025le7 the
airfoil default in run_all_geoms.sh and rerun naca0012 (the old naca0012 SB runs need
redoing).

**How to apply:** until then, the aero (ΔC_D/ΔC_L) metrics of the h0.025 naca0012 runs
in bands containing a jump measure the reference's step, not method error; L2/L∞ are
barely affected. Run naca2412, rae2822, oneraD, oa209 with the le7 tag, never plain
h0.025. Related: [[sb-does-not-beat-ffd-baseline]], [[cali3-storage]].
