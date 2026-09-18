---
name: mach-range-physics-limits
description: The diamond SB pipeline is only valid for M>=1.3 — oblique drift is undefined below M~1.25 and there is no shock at all below M=1
metadata: 
  node_type: memory
  type: project
  originSessionId: e09870cc-5949-4eb4-91d8-3162b8557c57
  modified: 2026-08-28T13:15:35.174Z
---

The diamond SB pipeline (Hessian shock marginal + oblique/FFD drift) is built for
**attached oblique shocks**, which for the 5° half-wedge means **M ≥ 1.3**. Measured
from `SB/drift.beta_le` on 2026-08-28:

| M | 0.8–1.0 | 1.05–1.20 | ≥1.30 |
|---|---|---|---|
| θ-β-M | no solution | **raises** `deflection θ=5.00° exceeds θ_max=1.52° at M=1.100 → shock detaches` | β_le = 59.96° … 23.13° |

Three regimes, only one validated:
- **M < 1.0** — no shock exists at all. The `iso_hessian` marginal keys on |λ_max(H_M)|,
  so it latches onto smooth-field curvature instead; the transport premise is different.
- **1.0 < M < ~1.25** — detached bow shock. `oblique` is unavailable (it *raises*, it
  does not return NaN — guard with try/except, not `isfinite` alone).
- **M ≥ 1.3** — attached, the regime every result so far comes from.

**How to apply:** when designing a Mach sweep, expect `oblique` to skip the bottom 5
of the 22 0.10-wide bands from 0.80. Do not read sub-1.25 results as equal-confidence
to supersonic ones. `run_sb_grid_big.sbatch` enforces this by calling `beta_le` itself
(so the guard follows the geometry rather than a hardcoded 1.25), and
`plot_err_vs_mach` shades the three bands red/orange/green.
See [[sb-does-not-beat-ffd-baseline]] and [[aoa-study-findings]].
