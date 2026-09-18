---
name: aoa-study-findings
description: Completed 576-run AoA study — SB never beats FFD on any axis; AoA-axis transport is amplitude-dominated and SB loses even to Linear there
metadata:
  type: project
---

Completed 2026-08-27: full matrix (Mach axis @ AoA 0/2/4° + AoA axis 0→4°@M2.00) ×
{null, oblique, ffd} × 12 γ × {h0.025, h0.0175, h0.0125} = 576 runs total.

**Mach axis:** SB beats Linear everywhere (SB/Lin 0.34–0.47) but never beats the FFD
registration (SB/FFD 1.24–2.08), and the gap widens with refinement at every AoA.
Drift degeneracy holds at incidence (null/oblique/ffd within ~3% at 2° and 4°).
`SBsquared` wins ONLY at the bottom of the ladder (γ≤5e-4: ~11% under oblique/ffd at
all meshes, cleanest same-batch comparison at γ=2e-4) — across the mid-ladder
(5e-2…2e-3) it is the WORST drift and at γ=2e-2 worse than Linear (ratio 1.19 vs
0.85), with a genuine non-monotonicity (5e-3 worse than 1e-2 at all 3 meshes): the
early bootstrap drift is 2γ∇g-dominated, huge and void-contaminated, and only the
fixed point cleans it up. Its gain heatmap row is visibly the bad one. Fixes worth
trying: seed from oblique/ffd instead of heat, or engage the bootstrap only below
γ≈1e-3. beta_delta converges mesh-independently (~0.008). Never run at AoA≠0 or on
the AoA axis.

**AoA axis (the big one):** transport at fixed Mach is AMPLITUDE-dominated, not
displacement: the FFD registration finds |β|_max = 0.0000 exactly at h0.025 and
h0.0175 (upper shock weakens as α−AoA→1°, lower strengthens — shocks barely move).
Consequences: Linear beats SB everywhere on this axis (SB/Lin 1.2–2.4), and FFD ==
Linear at coarse meshes. Only at h0.0125 does the registration resolve the rotation
(|β|=0.23) and then FFD beats Linear (0.00151 vs 0.00183). Oblique is the best SB
drift there (0.00225, 25% better than null/ffd) — analytic physics helps most for
rotation.

**C_L:** Linear wins by 12–16× at incidence (2.1e-3 vs 1.2e-4 at 2°). Structural:
C_L is a linear functional of p, the wall never moves, so any transport displacement
at the wall is spurious. At 4° SB's shock angle error (3–4°) is ~2× Linear's — the
L2 win at strong incidence comes from the bulk, not the shocks (weak-shock caveat:
the boundary-follower is less reliable on the nearly-degenerate upper shock).

**How to apply:** the data points to a displacement/amplitude DECOMPOSITION as the
next method step — geometry from registration/SB, amplitude interpolated along the
transported coordinates (or unbalanced OT / Wasserstein–Fisher–Rao). Also untested:
gate_to_flow A/B (implemented, never concluded), SBsquared_exact (implemented,
adjoint-verified 7e-15, never run), T=id wall constraint for the C_L pathology,
Scharfetter–Gummel flux for the Péclet penalty. See [[sb-does-not-beat-ffd-baseline]].
