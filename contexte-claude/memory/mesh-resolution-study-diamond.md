---
name: mesh-resolution-study-diamond
description: "Ongoing diamond mesh-convergence study — ladder, what Phase 0 fixed, what still needs the cluster"
metadata: 
  node_type: memory
  type: project
  originSessionId: e09870cc-5949-4eb4-91d8-3162b8557c57
  modified: 2026-08-18T09:01:05.869Z
---

Started 2026-08-18: testing whether the FFD-guided SB drift survives mesh refinement.
Ladder is **h ∈ {0.025, 0.0175, 0.0125}** (20190 / 41106 / 80889 cells), diamond only,
`hessian_mode="eig"`, `reference_drift="ffd"`, γ_min ∈ {1e-2, 1e-3, 1e-4}.

**Trap that cost real time:** the fine meshes shipped in the repo before this date were
α=10° half-wedge (`height=0.176`) while `h0.025` is α=5° (`height=0.0875`) — a refinement
study across them silently compared two different airfoils. The 10° set is archived in
`meshes/diamond/alpha10/`; the 5° ladder is rebuilt with
`uv run python Euler/diamond.py --h 0.0175 0.0125`.

**How to apply:** whenever a new diamond mesh appears, assert
`metadata["height"] == 0.087488663525924` before using it. Cost scales ≈1/h⁴, but the 5°
meshes are far smaller than the 10° ones were (near-body refinement scales with `height`),
so h0.0125 is only ~×16 the h0.025 cost, not ×36.

Remaining work is cluster-side: Euler snapshots (`run_euler_refined.sbatch`, 11-Mach
array per h) then `run_sb_sweep_refined.sbatch --array=5,8,11`. See
[[sb-does-not-beat-ffd-baseline]] for what the results are being judged against.
