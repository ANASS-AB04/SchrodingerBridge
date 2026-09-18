---
name: sb-does-not-beat-ffd-baseline
description: The raw FFD registration beats the SB bridge built on it at every gamma and every mesh — and the drift-independence result shows why
metadata:
  node_type: memory
  type: project
  originSessionId: e09870cc-5949-4eb4-91d8-3162b8557c57
  modified: 2026-08-20T08:22:02.547Z
---

On the diamond case (eig marginal), the raw FFD registration used directly as an
interpolator beats the Schrödinger bridge built on top of it at **every** γ and at
**every** mesh resolution. Mean-t L2 at h=0.025: 0.0023 (FFD) vs 0.0032 (BaryCDI) vs
0.0084 (Linear). The gap *widens* with refinement — SB/FFD = 1.41 → 1.60 → 1.76 at
h = 0.025 → 0.0175 → 0.0125 (γ=1e-4).

**Why (established 2026-08-20 from the mesh-study sweep):** at h=0.025, the `null`,
`oblique` and `ffd` drifts all converge to the SAME error (3.12–3.30e-3, <2% apart at
their optima). A crude analytic θ-β-M formula and a fitted 200-DOF registration give
identical SB answers. β contributes no accuracy — it only keeps the kernel from
collapsing (null diverges below γ=2e-4, the other two don't). This is expected from the
math: β enters the SB objective as KL against the reference path measure weighted by γ,
so **the γ-anneal to the OT limit systematically erases the very information that made
FFD good**. SB/FFD bottoms out at γ≈1e-4 and rises again below it — there is a floor,
and it sits above FFD.

Second mechanism, refinement-specific: the marginal ρ ∝ |λ_max(H_M)|^(1/3) with
λ_max ~ ΔM/h², so refining drives the marginals toward curve-supported measures. The
log-domain kernel blows up accordingly (den_T_max 5.9e178 → 2.1e273 across the h ladder
at fixed γ=1e-4) and the trust gate reverts 14.5% → 41.5% of cells. `_trust_gate`
(SB/utils.py) reverts to the **identity**, not to Φ_β, even in the drifted branch —
so those cells become the Linear baseline rather than the FFD flow. That is why
BaryCDI's h-scaling tracks Linear's (+36%) instead of FFD's (+9%).

**How to apply:** treat "does SB beat FFD" as the success criterion, not "does SB beat
Linear" (it always does). Untested levers, cheapest first: (1) gate fallback to Φ_β
instead of identity; (2) γ ∝ h² ladder instead of the fixed prefix — γ=1e-4 at h0.025
corresponds to 2.5e-5 at h0.0125; (3) stop annealing at the finite-γ optimum (γ≈2e-4 at
h0.025) rather than driving to OT; (4) `SBsquared` at fine meshes — it is the one
mechanism that breaks the β-erasure, and `run_mesh_study.sbatch` defaults to
`DRIFTS=ffd` so it has never been run there. Also untested: β is a 200-DOF Bernstein
field on a fixed 144² grid independent of the mesh, so `grid_n`/`n_cp` is a knob.
See [[mesh-resolution-study-diamond]].

**Replicated on naca0012 (2026-09-08, 387 SB runs, h0.025, 22 Mach bands × AoA 0/2°,
γ=1e-4):** same ordering, third geometry. Mean-t L2 in the attached-shock regime @ AoA 0:
FFD 4.01e-3 < BaryCDI 5.91e-3 < Linear 6.57e-3. Per band, SB beats Linear in about half
the bands (19–24 of ~43) and beats FFD in only 2–3 of them, for every drift. Drift again
barely matters (mean SB L2 8.77e-3 null / 9.08e-3 ffd / 8.78e-3 SBsquared_exact+ffd,
pooled over all bands) — consistent with the β-erasure mechanism above, now on a smooth
blunt-nosed body rather than a wedge.
