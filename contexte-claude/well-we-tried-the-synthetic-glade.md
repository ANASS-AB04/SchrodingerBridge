# PART VI — Airfoil defects: body predicate + marginal  ← ACTIVE

## ROOT CAUSE 1 (primary) — `inside_body` is identically False on every airfoil

`drift.py:206-232` `make_inside_body` has an **exact** rhombus predicate behind
`if case == "diamond"`, and a KD-tree fallback for everything else:

```python
thr = 4.0 * median(nearest-neighbour distance)
inside(P) := distance(P, nearest fluid barycentre) > thr
```

It is a **void detector, not a body detector** — it can only fire where fluid
cells are absent over a radius larger than `thr`. Measured on the diamond by
forcing the fallback:

```
thr = 4 x median NN            = 0.07746
10139 points verified INSIDE the body
  max distance to nearest barycentre = 0.04956
  detected by the fallback           = 0 / 10139
```

Any body thinner than `thr` is invisible. naca0012's max half-thickness is
**0.060**, thinner still toward the TE, and wall cells are refined to `h/5`, so
`inside_body` returns **all-False for every airfoil**. The whole
body-protection machinery is dead code for the geometries now running.

### The causal chain to the shear

1. `utils.py:1397-1400` — the CDI fallback `bad0 = inside_body(W) | ~isfinite(M0)`
   is empty. Meanwhile the interpolator is `Delaunay(barycenters)`
   (`utils.py:1358`), whose convex hull **fills the airfoil hole with sliver
   triangles joining upper- and lower-surface wall cells**. Query points inside
   the body are therefore neither NaN nor flagged, so the CDI returns a **linear
   blend of upper- and lower-surface Mach and pressure across the solid**. Worst
   at the razor-thin TE where the slivers are longest. The header comment at
   `utils.py:1352-1356` claims this is handled — that guarantee holds only when
   `inside_body` works.
2. `drift.py:266-284` — the bisection retraction in `flow_map` never fires, so
   `phi_fwd`/`phi_bwd` pass through the body and `NearestNDInterpolator` samples
   β from whichever surface is closer.
3. `utils.py:1319-1330` — `bias_T`/`bias_S` therefore acquire a phantom
   displacement on every wall cell (Φ crosses the body, the no-flux kernel
   cannot) and it is subtracted from `T`/`S` right where the shock feet are.
4. `utils.py:1336-1339` — the trust cap is a global max over `|Φ − x|`, inflated
   by those body-crossing trajectories, so the gate degenerates to a no-op.

### Two diagnostics that hid it

- `utils.py:2396-2399` counts gate reverts as `T_map == barycenters` exactly, but
  with `gate_to_flow=True` (the default) the fallback is `phi_fwd`, **not**
  `bary` — so `gate_reverts_T/S` is structurally **always 0** and reports nothing.
- `advdiff_solver.compute_n_steps_advdiff` calls only `check_resolution`, never
  `check_cfl_stability` — so the drifted branch, which is what every airfoil run
  uses, never emits the instability warning.

### Fix

Replace the fallback with a real point-in-polygon test built from the mesh's own
wall faces (`face_markers == 2`), which exist for every geometry. That is exact
for diamond, bump and all airfoils, and removes the `case == "diamond"` special
case entirely.

---

## Ranked list of live mechanisms (airfoil runs, cfl_pct=0)

| # | mechanism | site | shear/noise | status |
|---|---|---|---|---|
| 1 | **CDI interpolates across the solid body.** `Delaunay(barycenters)` spans the airfoil hole; `inside_body` all-False so the fallback never fires. Sliver triangles at the razor-thin TE give long wake streaks. | `utils.py:1358,1397` + `drift.py:229` | **shear**, large amplitude, directly in the output | VERIFIED (0/10139) |
| 2 | **TPFA has no non-orthogonality correction.** `(φ_j−φ_i)/d_ij` is used as ∂φ/∂n; the error is a spurious tangential flux ≈ artificial advection with Pe ≈ sinθ = O(1) — **independent of γ and dx, so refinement does not fix it**. Mesh-locked direction in the stretched LE/TE rows. Affects every semigroup solve. | `heat_solver.py:180-183` (reused by advdiff) | **shear**, everywhere | structural |
| 3 | **The marginal is mesh-weighted.** Green–Gauss gain (Σl_f/A)² = (1/dx)² varies ~20× across the mesh; then `feat**(1/3)` is concave, so it boosts near-zero Hessian garbage ~10⁴× in relative terms. | `utils.py:68-75` ×2, `utils.py:207` | **noise** — this is the `Density_mu0` speckle you saw | matches observation |
| 4 | **FFD registration rasterises to a fixed uniform 144² grid**, coarser than the h/5 LE cells, so the near-body ridge is aliased away before β is fitted. That blind β then drives the whole ladder and Φ_β. | `ffd_drift.py:63,84-97` | shear, smooth/large-scale | structural |
| 5 | **Trust-gate cap is `jnp.max` over all cells** + 4σ — one outlier opens the gate globally. And `gate_reverts_T/S` is structurally always 0 (compares against `bary` while the fallback is `phi_fwd`). | `utils.py:1335-1339`, `2396-2399` | shear, binary/coherent | real |
| — | Explicit-diffusion instability (19.9× past the limit at `pct=10`) | `heat_solver.py:107` | noise | **already fixed** by `cfl_pct=0`; only the *warning* is missing on the drifted path (`compute_n_steps_advdiff` calls `check_resolution` but not `check_cfl_stability`) |

Fix order: **1** (clean, self-contained, biggest win), then **3** (switch the airfoil
marginal to `field/grad_mach`), then reassess. **2** and **4** are structural and
should not be attempted before 1 and 3 are measured.

---

## ROOT CAUSE 2 (secondary) — the marginal is mesh-weighted

## Context

The first naca0012 SB runs produce visible shear and high-frequency noise. You
localised it by eye and it is decisive: **`Mach_M0_raw` (the Euler field) is
clean, `Density_mu0` (the Hessian marginal) is noisy.** So the defect is in the
marginal construction — not the Euler solve, not the bridge, not the CDI.

## Diagnosis

`build_hessian_density` ([SB/utils.py:184](SB/utils.py#L184)) builds the marginal as

```
gM  = compute_cell_gradient(mesh, mach)      # first pass
gMx = compute_cell_gradient(mesh, gM[:,0])   # second pass  -> H
feat = |λ_max(H)|
raw  = feat ** (lp/(2lp+2))                  # = feat^(1/3) at lp=2
raw  = raw * (|M − M∞| > percentile(floor_pct))
```

with **no normalisation by cell size anywhere**. Three compounding effects, all
of which are amplified by the airfoil mesh and were invisible on the diamond:

1. **Mesh-size weighting — this is the "shear".** For a shock, λ_max ~ ΔM/h²
   (already recorded in [[sb-does-not-beat-ffd-baseline]]). On the diamond the
   cell-size spread is dx_10th/dx_min = **1.64**, so 1/h² is nearly a global
   constant and normalisation removes it. naca0012's spread is **4.46**, because
   `mesh_utils.sample_airfoil_boundary` samples the surface down to `h_min = h/5`
   where curvature demands it. That is a **4.46² ≈ 20×** spatially varying
   weight, and ρ ∝ feat^(1/3) turns it into ~2.7× spurious mass wherever cells
   are small — i.e. along the refined airfoil surface. The marginal therefore
   tracks the *mesh refinement pattern* rather than the shocks, which is exactly
   what directional streaking looks like.

2. **Gradient noise squared — this is the "noise".** Two successive
   `compute_cell_gradient` passes; the second differentiates an already-noisy
   field. The codebase already knows this operator is dirty — the comment near
   [SB/utils.py:565](SB/utils.py#L565) says its noise "makes the drift-CDI shear
   along shocks", which is why the barycentric CDI was made gradient-free. The
   marginal still uses it, twice.

3. **The active-zone gate does not exclude the surface on a curved body.** It
   keeps cells with |M − M∞| above the `floor_pct` percentile. On the diamond the
   straight legs leave the flow near-freestream except at shocks, so the gate is
   effective. An airfoil accelerates flow over its whole upper surface, so the
   gate keeps the entire surface region — precisely where (1) and (2) are worst.

None of this is a coding typo. It is the Alauzet–Loseille metric, discretised,
meeting a mesh it was never validated on.

## What we changed for the airfoils (none of it causes this)

Checked against `git diff HEAD -- SB/`: the only edits to `SB/utils.py` are the
`cfl_pct` threading and the `is_wedge` flag. Specifically:

| change | effect | can it cause shear? |
|---|---|---|
| `cfl_pct` 10 → 0 | `n_steps` only — *more* steps, i.e. more accurate | no |
| `LADDER_START` 0.5 → 0.02 | real numerical change, but affects the MAPS; the marginal is built before any IPFP | no (but untested — see V5) |
| `compute_w2` off | reporting only | no |
| `oblique` dropped | correct: no wedge on a rounded LE | no |
| `le_shock_geometry` → α=NaN | metrics only (angle/RH reported as NaN) | no |

## Fix

### A. Move the airfoil marginal off the second derivative (recommended)

`build_field_density` ([SB/utils.py:229](SB/utils.py#L229)) is already
implemented, already wired through `Config.toml`, and already has its own output
directory level (`field/<field_source>/`):

- `field_source="pert_mach"` — |M − M∞|, **zero** derivatives, fully
  mesh-independent
- `field_source="grad_mach"` — ‖∇M‖, **one** derivative (1/h, not 1/h²), still
  keys on shocks because a shock is a gradient spike

Use `density_mode="field", field_source="grad_mach"` for airfoils, scoped by
case in `run_sb_grid_big.sbatch` exactly like `CFL_PCT` and `LADDER_START`.
Diamond and bump keep `iso_hessian`, so the existing 576-run baseline is
untouched.

### B. Reduce the mesh cell-size spread (complementary)

`h_min_factor` is now exposed on `MeshSizeParams`
([Euler/meshing/mesh_utils.py:21](Euler/meshing/mesh_utils.py#L21)), default 5.0
= current behaviour. Raising it to 2–3 shrinks the spread, which simultaneously:
- reduces the 1/h² weighting that causes (1),
- makes `cfl_pct=10` safe again → ~20× cheaper bridge **and** W₂, so W₂ can come
  back for airfoils.

Measure `dx_min` / `dx_10th` / cell count for `h_min_factor ∈ {5, 3, 2, 1.5}`
before choosing — it trades leading-edge resolution for SB cost.

### C. Only if `iso_hessian` must be kept

Divide `feat` by a local h² (the `dx_i` already computed in
`heat_solver.reference_cell_size`) before the power, making it a true continuous
metric. Riskier: it changes the diamond too, so it must be A/B'd against the
stored baseline rather than adopted blind.

## Verification

1. **Quantify the diagnosis, locally, no GPU.** From one naca0012 bundle, build
   the marginal three ways (`iso_hessian/eig`, `field/grad_mach`,
   `field/pert_mach`) and report the rank correlation between the marginal and
   1/dx² per cell. The prediction is a strong correlation for `iso_hessian` and a
   weak one for the `field` variants.
2. **Control:** same on the diamond. Correlation should be much weaker there
   (spread 1.64 vs 4.46) — that is what makes this an airfoil-specific defect
   rather than a general one.
3. **Visual:** one naca0012 SB run with `field/grad_mach`; compare `Density_mu0`
   and `Mach_BaryCDI_t0.50` against the current ones.
4. **Regression:** a diamond run must be byte-identical — the switch is
   case-scoped, so `iso_hessian` stays the diamond/bump default.
5. **Unrelated but untested:** isolate the `LADDER_START` change by rerunning
   diamond `M2.00-2.10 / ffd / gmin0.01` with `LADDER_START=0.02` (2 stages) and
   comparing BaryCDI mean-L2 against the stored full-ladder value
   **4.1943e-03**. Any difference is the shortened anneal, not the geometry.

---

# PART V — FFD-seeded, exact SB² (reconditioned bootstrap)

## Context

The full-ladder data explained SB²'s heatmap: the bootstrap's stage 0 recovers
`b = 2γ∇g` from a HEAT bridge at γ=0.2 — the huge (|b|≈2.8), void-contaminated field
imaged earlier — and the mid-ladder spends eight rungs washing it out, losing to every
drift (and at 2e-2 to Linear itself), before finally winning ~11% at γ≤5e-4. Two
suspected causes, both proposed by the user: **bad initialization** (heat seed) and
**the frozen t=0.5 collapse** (not exact). The fix: seed the bootstrap with the FFD
registration, keep it fixed down to γ≈1e-3 (where ffd-drift is at or near best anyway),
then engage the **exact time-dependent** bootstrap below.

Status of the machinery: β(t) stacks are fully implemented — kernel indexes them with
Q backward / Q† forward in SB time, verified against the adjoint identity to 7e-15
(negative control fails at 183%) — but `SBsquared_exact` has NEVER executed end-to-end
(its smoke test was interrupted). Seeding and a switch-γ do not exist yet.

## Decisions

- **Two new config knobs**, composable with both bootstrap modes:
  - `drift_bootstrap_seed = "null" | "ffd" | "oblique"` (default `"null"` = current
    behaviour). For `"ffd"`, reuse the registration already computed for the FFD
    control (`_b_ffd` in the [4b] block of `run_mach_interpolation_case`) — do not
    register twice.
  - `drift_bootstrap_start_gamma` (default 0 = engage immediately, current behaviour).
    While γ > this, β stays FIXED at the seed — those stages are literally a
    drift_ffd run; below it, each stage recovers `b = β + 2γ∇g_t` and feeds it
    forward. Recommended value 1e-3: three bootstrapped stages (5e-4, 2e-4, 1e-4),
    enough for the fixed point (beta_delta halved per stage) without re-entering the
    region where the heat-seeded bootstrap was 30-40% worse.
- **Run the 2×2**, not only the combined scheme: {heat, ffd-seed} × {frozen, exact}.
  Heat+frozen exists; 3 new variants × 12 γ at h0.025 ≈ 4 GPU-h. Isolates which
  ingredient (seed vs exactness) carries any win — that decides whether the β(t)
  overhead is worth keeping.
- **Expectation setting**: current SB² endpoint wins ~10% over other drifts but sits
  ~35% above the FFD baseline. This scheme is the most bridge-favourable configuration
  the codebase can express; if it still loses to FFD, that is close to a definitive
  negative for the bridge here. Either outcome is informative.

## STATUS — implementation landed, smoke tests run, one fix outstanding

Seeding + switch-γ are implemented (`drift_bootstrap_seed`,
`drift_bootstrap_start_gamma`), plumbed through `main.py`/`Config.toml`, given
composite output tags (`drift_SBsquared_exact+ffd/`), taught to `sweep_analysis.py`,
and wired into `run_sb_aoa_sweep.sbatch` via `DRIFTS="…+ffd"` + `BOOT_SWITCH`.

**Four CPU smoke runs (3-stage ladder, num_iter=4, γ→2e-3) — all plumbing correct:**

| run | stage sources | \|β\|max | BaryCDI L2 |
|---|---|---|---|
| exact, heat seed, bootstrap from start | heat, bootstrap, bootstrap | **12.96** | 0.005743 |
| exact, ffd seed, switch=0 (never engage) | seed-fixed ×3 | 0.4082 | **0.003685** |
| plain `ffd` (reference) | external ×3 | 0.4082 | **0.003685** |
| exact, ffd seed, switch=0.01 | seed-fixed, seed-fixed, **bootstrap** | 2.7585 | 0.003859 |

- **Degeneracy check passes exactly**: seeded-but-never-engaged ≡ plain `drift_ffd`
  (identical L2 to all 6 digits, identical |β|). The fixed-seed phase is provably the
  ffd kernel.
- **First-ever end-to-end execution of `SBsquared_exact`** — it runs.
- Stage provenance (`drift_source`) reports exactly as designed in every configuration.

**BLOCKER FOUND — the exact mode has a t→1 singularity.** Measured |2γ∇g_t| against t
on a synthetic void+ridge potential:

```
 t      0.00   0.25   0.50   0.75   0.90   1.00
|b|max  1.35   1.50   1.79   2.40   3.70  13.76
```

`g_t = log Q_{1−t}[e^g]`, so at t=1 `Q_0` is the identity and `g_t` is the RAW
log-potential — gradient enormous across the void/ridge boundary. The frozen mode
samples only t=0.5 and never sees it; the exact mode includes it, and `|β|_max` sets
the advective CFL and Péclet for the entire kernel. This is why exactness ALONE made
things worse (12.96 vs the frozen mode's ~2.4), and why even the seeded switch=0.01 run
inflated β 6.7× (0.41 → 2.76) after a single bootstrap stage.

**Fix before sweeping**: add `drift_bootstrap_t_clip` (default 0.1) and sample the
β(t) stack at K times spanning `[clip, 1−clip]` instead of `[0, 1]`. The kernel already
interpolates the stack by index, so the end slices simply clamp to the nearest interior
value — no change to the Q/Q† mirroring. At clip=0.1 the worst slice is ~3.7 rather than
13.8; the profile above is the calibration data if a larger clip is wanted.

## Implementation (done, except the t-clip above)

1. **`SB/utils.py`, annealing loop** (the block at the `bootstrap_drift`/`beta_k`
   state): initialize `beta_k`/`beta_max_k` from the seed when
   `drift_bootstrap_seed != "null"`; gate the recovery block with
   `gk <= drift_bootstrap_start_gamma` (recover only when the NEXT stage is at or
   below the switch; above it β passes through unchanged). `use_drift` starts True
   when seeded. Record `seed`, `start_gamma`, and per-stage `drift_source`
   ("seed-fixed" | "bootstrap") in metrics.
2. **`SB/main.py`**: plumb the two keys; extend the drift dir tag to
   `drift_<mode>+<seed>` when seeded (e.g. `drift_SBsquared_exact+ffd`) so nothing
   collides with existing SB² runs. Pure modes keep their exact current paths.
3. **`SB/Config.toml`**: the two keys + doc comment.
4. **`SB/sweep_analysis.py`**: add the composite names to DRIFT_ORDER/DRIFT_COLOR/
   marker (exact-string filtering would otherwise drop them from every plot).
5. **`run_sb_aoa_sweep.sbatch` / `run_mesh_study.sbatch`**: accept the composite names
   in DRIFTS by splitting on `+` into (reference_drift, seed) and sed-ing both keys.

## Remaining work

1. **`drift_bootstrap_t_clip`** in `SB/utils.py` (the `bootstrap_exact` branch that
   builds the stack via `np.linspace(0.0, 1.0, nt_boot)` → `linspace(clip, 1-clip, …)`),
   plus the Config key and a `sed` line in the sweep script. ~15 lines.
2. **Re-smoke** the exact+ffd configuration and confirm |β|max stays near the seed's
   0.41 rather than jumping to 2.76.
3. **Sweep at h0.025** — the 2×2 so the active ingredient is identifiable:
   ```bash
   DRIFTS="SBsquared+ffd SBsquared_exact+ffd SBsquared_exact" \
     BOOT_SWITCH=0.001 sbatch run_sb_aoa_sweep.sbatch
   ```
   3 variants × 12 γ × 4 cases = 144 tasks, but only the `aoa0.00` case (36) is needed
   for the head-to-head; use `CASES=aoa0.00` to cut it to 36 tasks ≈ 1.5 GPU-h.
4. **Read-out** against the existing ladder: (a) is the mid-ladder penalty gone —
   does the γ=2e-2 ratio drop from 1.19 below 0.85; (b) does the endpoint beat the
   current SB² best of 2.81e-3; (c) does anything beat FFD's 2.26e-3.

## Verification (ordered; sweep only after all pass)

1. **CPU smoke of `SBsquared_exact`** end-to-end (short ladder, num_iter≈5,
   compute_w2=false) — first-ever full execution of the exact path; confirm the
   (K,N,2) stack flows through recovery→smoothing→kernel→maps→metrics.
2. **Degeneracy**: seed=ffd with `start_gamma=0` never reached (i.e. switch below the
   ladder floor) must reproduce a plain `drift_ffd` run's numbers to round-off —
   proves the fixed-seed phase is exactly the ffd kernel.
3. **Continuity**: seed=null, start_gamma=0 must reproduce the existing `SBsquared`
   behaviour (same recovery at every stage).
4. **Endpoint exactness**: t=0/t=1 reconstructions equal the endpoint bundles.
5. **Sweep**: 3 variants × 12 γ at h0.025 via the array script; then the ladder table
   as before — the read-out is (a) is the mid-ladder penalty gone, (b) does the
   endpoint beat 2.81e-3 (current SB² best), (c) does anything finally beat the FFD
   baseline 2.26e-3.

---

# PART IV — Refined-mesh AoA runs + post-processing

## Context

The h0.025 AoA matrix (144 runs) is complete and the AoA-axis interpolation looks right
by eye. Two things follow: extend the matrix to the refined meshes, and build the
post-processing that the per-run PNG dump cannot give — animations, method comparison,
shock-locus geometry, and a cross-case summary.

The h0.025 results already say something sharp, which shapes what the post-processing
needs to show:

- **On the AoA axis, SB is worse than doing nothing** — BaryCDI 0.00182 vs Linear
  0.00119, the first case where the bridge loses to a plain blend. And
  `FFD == Linear == 0.00119` exactly, meaning the registration found ~zero displacement
  and degenerated into Linear.
- **The drift degeneracy survives incidence** — at 2° null/oblique differ by 3%, at 4°
  by ~3%. β still buys no accuracy.
- **C_L discriminates now, and Linear wins by 12–16×** once lift is real (2° and 4°),
  versus a tie at 0° where both sit under the noise floor.

## Decisions taken

- Full 144-task matrix on **both** refined meshes (~264 GPU-h, ~22 h wall at 12 concurrent).
- All four post-processing artefacts: animations, side-by-side at fixed t, shock-locus
  overlay, cross-case summary.

---

## A. Refined-mesh runs — submissions only, no code change

```bash
H_TAG=h0.0175 sbatch run_euler_grid_aoa.sbatch      # 19 new of 41 (AOA0/AOA2 exist)
H_TAG=h0.0125 sbatch run_euler_grid_aoa.sbatch
# then, once each finishes:
H_TAG=h0.0175 sbatch run_sb_aoa_sweep.sbatch                    # 144, 6 skip
H_TAG=h0.0125 sbatch --time=08:00:00 run_sb_aoa_sweep.sbatch    # 144, 6 skip
```

**Raise the walltime at h0.0125.** The default `--time=04:00:00` is sized for h0.025;
the worst measured task there (ffd, γ=1e-4, 12 stages) was 968 s of IPFP, ~32 min with
W₂ — fine — but `null` can stall at the `num_iter` cap on every stage, and 8 h keeps
the margin without making the tasks materially harder to schedule.

## B. Persist what post-processing needs — `SB/utils.py`

Nothing but scalars survives a run today (`metrics.json` has no fields), so two of the
four artefacts have nothing to work from. Two small additions:

1. **`fields.npz` per run** — the reconstructed Mach field for each method at all 11
   frames, float32. 2.7 MB at h0.025, 10.7 MB at h0.0125; ~2.7 GB across the whole
   study. Behind `save_fields` (default true) so it can be turned off.
2. **Shock loci into the `transport` block** — `compute_transport_metrics` already fits
   a line per method per `t_ref` via `fit_shock_curve`; store the `sample_shock_curve`
   output (200 points) alongside the angles. A few kB, and it makes the overlay plot
   possible without loading fields at all.

Existing h0.025 runs have neither. Re-running that mesh to backfill is ~13 GPU-h —
cheap next to the 264 the refined meshes cost — but the animations do not need it (see
below), so it is optional.

## C. `SB/postprocess.py` — new module, operates on finished run directories

No new dependencies: **PIL is available, `imageio` and `ffmpeg` are not**, so animations
are GIF via `PIL.Image.save(append_images=…)` rather than MP4.

| artefact | source | works on existing h0.025 runs? |
|---|---|---|
| `animate_run` — GIF sweeping t=0→1 per method | stitches the existing `Mach_<method>_t*.png` | **yes**, no re-run |
| `compare_at_t` — reference \| BaryCDI \| FFD \| Linear, shared colour scale, shock overlaid | `fields.npz` | needs re-run (PNG-montage fallback) |
| `shock_overlay` — fitted loci of reference + each method on one axes | `transport` block (B2) | needs re-run |
| `summary_report` — error + 4 transport metrics vs AoA, per drift, per mesh | `metrics.json` only | **yes**, no re-run |

Reuse `fit_shock_curve` / `sample_shock_curve` ([SB/utils.py:826](SB/utils.py#L826)) and
`plot.py`'s existing tripcolor helpers rather than writing new mesh plotting.

The animation matters most for the reason you have already been using it: the smearing
in the SB² frames and the AoA-axis result both showed up by eye before any metric caught
them. A GIF makes the shock's motion legible in a way 11 separate PNGs do not.

## D. Pending fixes to fold in

- **`backfill_ffd_control` in `sweep_analysis.py`** — this edit was aborted mid-apply and
  never landed (`want_ffd_control` in `utils.py` did). The FFD control is γ- and
  drift-independent, confirmed in the data (FFD_l2_mean constant across all 12 γ), so
  copy it within each `(hmode, h, axis, aoa)` group instead of re-running 96 jobs.
  Mark back-filled rows with an `ffd_backfilled` column so a shared value is never
  mistaken for a measured one.
- **AoA-axis group mislabelled** — reported as `axis=aoa, AoA = 0°` for a 0→4° sweep,
  because `aoa_deg` takes the first bundle's angle. Should print the range.
- **AoA-axis figures land under `Mach_interpolation/…/_sweep/`** — `outdir` defaults to
  the first root. The `_axisaoa` suffix stops them colliding, but they are filed under
  the wrong root.

## Verification

- **Euler:** `--array=0` first at each H_TAG; then 41 bundles per mesh, `n_cells`
  matching, all at `t4.00`.
- **SB:** `sacct -j <id> -X -n -o State | sort | uniq -c` → 144 COMPLETED per mesh.
- **fields.npz:** shape `(3, 11, N)` and dtype float32; the t=0 and t=1 slices must equal
  the endpoint bundles exactly (the CDI is exact at the endpoints — this is the check
  that catches a wiring error).
- **Animations:** generate for one existing h0.025 run *before* touching anything else,
  since that path needs no re-run and proves the PNG-stitching route.
- **`backfill_ffd_control`:** re-run `sweep_analysis.py` on the untouched tree; the 40
  pre-existing shared columns must stay byte-identical, and `FFD_l2_mean` must become
  non-NaN on null/oblique rows with `ffd_backfilled=True`.
- **Shock overlay:** the reference locus must sit on the analytic θ-β-M ray to within the
  ~0.1° the metric already reports.

---

# PART III — AoA study: drifts across incidence, and transport along the AoA axis

## Context

Everything so far interpolates in **Mach at zero incidence**, and that has quietly
capped what the metrics can say. At AoA=0 the diamond is symmetric, so the reference
C_L is pure mesh-asymmetry noise (~1e-4, and it *flips sign* between meshes) — lift
cannot discriminate between interpolators at all. Meanwhile `null`, `oblique` and `ffd`
all land within 2% of each other, so the drift comparison has no dynamic range either.

Two additions open both up:

1. **Mach interpolation at several fixed AoA** (0°, 2°, 4°) × the three drifts. Lift goes
   from meaningless (≈1e-4) at 0° to ~0.08 at 2° and ~0.16 at 4° — a signal ~1000×
   the noise floor — so C_L becomes a real metric, and we learn whether the
   drift-independence result survives incidence.
2. **Interpolation along the AoA axis itself** (0°→4° at fixed Mach). A genuinely
   different transport problem: the shocks *rotate* about the leading edge rather than
   sweeping downstream, and the upper and lower surfaces move in opposite directions.

## Decisions taken

- AoA values **0, 2, 4°**; AoA-axis sweep at **fixed M = 2.00** (shares 3 snapshots with
  the Mach-axis runs, so the two axes touch and can be cross-checked).
- Output: keep `Mach_interpolation` exactly as it is, add a **sibling
  `AoA_interpolation` root** for the AoA-axis runs.
- **Array jobs, not the 2-GPU pool.** At `--gres=gpu:1` per task SLURM spreads work
  across every free L40 and H100 GPU instead of the 2 a single node can hold, each task
  writes its own directory so there are no write races, and short tasks schedule far
  more easily than one 24 h reservation.

## Snapshot accounting (verified against disk)

| | count |
|---|---|
| Mach axis: 3 AoA × 11 Mach | 33 |
| AoA axis: 11 AoA @ M2.00 | 11 (3 shared) |
| **union needed per mesh** | **41** |
| already on disk at h0.025 | 13 |
| **new Euler solves at h0.025** | **28** |

`AOA0.00` is complete (11/11). `AOA2.00` has only M1.20/1.30/2.00/2.50 — the two
endpoints are reusable, the 9 references are not.

---

## 1. Output directories

`_mach_output_dir` ([SB/main.py](SB/main.py)) currently hardcodes `"Mach_interpolation"`.
Make the root and the case level follow the interpolation axis:

```
Mach axis, AoA 0   →  outputs/Mach_interpolation/diamond/iso/hessian/eig/h0.025/drift_ffd/gmin<γ>/
Mach axis, AoA 2   →  outputs/Mach_interpolation/…/h0.025/aoa2.00/drift_ffd/gmin<γ>/
AoA  axis @ M2.00  →  outputs/AoA_interpolation/…/h0.025/M2.00/drift_ffd/gmin<γ>/
```

Two properties that matter: **nothing existing moves** (all 135 completed runs keep
their paths), and the AoA-axis tag now carries the fixed Mach — my current
`aoa0.00-4.00` does not, so two AoA sweeps at different Mach would have collided.

The axis is already detected in `run_mach_interpolation_case` (it compares `aoa_in` from
the two endpoint bundles and records `interp_axis` in `metrics.json`); `_aoa_tag` just
needs to consume the same information and return the root alongside the tag.

**`sweep_analysis.py` must scan both roots** — this is the acknowledged cost of the
sibling-root layout. Change `--root` to accept multiple paths (default: both), and add
an `axis` column from `cfg["interp_axis"]` so the two never pool in one plot.

## 2. `run_euler_grid_aoa.sbatch` — HR generation (new)

Array job, one task per (AoA, Mach) pair, `--gres=gpu:1`, `--mem=8G`,
`--partition=gpu-h100,gpu-l40`, `--array=0-40%12`.

- Task index → the 41-pair union, built once in the script so both axes are covered
  without duplicating the 3 shared corners.
- Reuse the **stale-bundle check** already in `run_mesh_study.sbatch`: compare each
  existing bundle's recorded `n_cells` against the mesh and quarantine mismatches. It
  caught the α=10°/5° mesh swap and will catch any repeat.
- Reuse `FULL_TIME=1` → `--stationarity-threshold 0`. Non-negotiable here: the residual
  is normalised by *step count*, so it tightens with dt and different runs would
  otherwise stop at different physical times.
- Keep `export LC_ALL=C` — bash `printf "%.2f"` emits `2,00` under fr_FR and would
  corrupt every filename.

Skips the 13 already-present pairs, so h0.025 costs 28 solves (~15–30 s each).

## 3. `run_sb_aoa_sweep.sbatch` — the SB matrix (new)

Array job, one task per (case, drift, γ). `--gres=gpu:1`, `--mem=8G`.

```
cases  = aoa0 | aoa2 | aoa4 | axis-aoa@M2.00      (4)
drifts = null | oblique | ffd                      (3)
gammas = the 12-rung ladder prefix                (12)
                                             → 144 tasks per mesh
```

Index decode mirrors `run_sb_sweep_refined.sbatch`'s existing scheme. Each task calls
`make_config_for_mesh.py` with `--axis mach --aoa <a>` or `--axis aoa --mach 2.00`
(both flags already exist), `sed`s in its drift and γ prefix, and runs `SB/main.py`.

Resume-skip on `metrics.json` as now — and with one task per output directory there is
no concurrent-write hazard, unlike submitting the pooled script several times.

**Scope: h0.025 only to start.** 144 tasks × ~5–12 min ≈ 19 GPU-h; across ~12
concurrent GPUs that is ~2 h wall. The finer meshes are ×4 and ×16 (76 and 304 GPU-h) —
`H_TAG` is a variable, so extend once the coarse matrix says which corner is worth it.

## 4. Analysis

`sweep_analysis.py` gains the `axis` column and dual-root scanning. The questions:

1. Does the `null`/`oblique`/`ffd` degeneracy (<2% apart at AoA=0) survive at 2° and 4°?
   If the drifts separate under incidence, β is doing something after all.
2. Does C_L discriminate once it is ~1000× its noise floor — and does the Linear-wins
   result survive, given C_L is a linear functional of p and the wall does not move?
3. Is AoA-axis transport (rotation about the LE) harder or easier than Mach-axis
   transport (downstream sweep)? The new Δβ metric measures exactly this, against
   θ-β-M rather than against a reference.

## Verification

- **Paths:** three synthetic configs (Mach@0, Mach@2, AoA-axis@M2.00) must produce the
  three distinct paths above; `git status outputs/` clean apart from new dirs.
- **Euler array:** `--array=0` alone first; confirm one bundle, correct `AOA<a>_M<m>`
  name, `t=4.00`, and `n_cells` matching the mesh. Then the full 0-40.
- **Index decode:** dry-run the array arithmetic in bash and check all 144 (case, drift,
  γ) triples are distinct and cover the intended grid — the same check that caught the
  locale bug.
- **AoA-axis sanity:** at t=0 and t=1 the reconstruction must reproduce the endpoint
  fields exactly (the CDI is exact at the endpoints), and the measured Δβ at t=0 must
  match θ-β-M for AoA=0 — a shock-angle check that is independent of the reference.
- **Metrics:** confirm the new `transport` block is populated (the four metrics were
  oracle-tested: Δβ 0.117° for the reference vs 2.20° for Linear).

---

# PART II — SB²: a bootstrapped, self-conditioned drift

## Context

Every reference drift so far comes from **outside** the bridge: `oblique` from analytic
θ-β-M relations, `ffd` from a registration of the two marginals. The `ffd` one is also
structurally capped — it is a 200-DOF Bernstein field on a fixed 144² grid that does not
depend on the mesh, so refining cannot sharpen it, and the raw registration still beats the
bridge built on it at every γ (0.0023 vs 0.0032 mean-L2 at h0.025).

**The idea:** let the bridge condition itself. Anneal γ downward as now, but at each stage
take the reference drift from the *previous* stage's own SB solution:

```
stage 0   γ₀ = 0.5    β = 0                    pure-heat bridge  → (f,g)
stage k   γₖ < γₖ₋₁   β = b^(k−1)              drifted Q/Q† bridge → (f,g)
                      where b = β + 2γ∇g_t     (retrieve_b_2d_drift)
```

Two things make this a natural fit rather than a hack:

- `retrieve_b_2d_drift` ([SB/resolution.py:166](SB/resolution.py#L166)) already returns the
  **total** drift `β + 2γ∇g_t`, which is exactly what the next stage should consume — the
  correction accumulates into the reference instead of being recomputed from scratch.
- **The true solution is a fixed point.** If β already equals the optimal drift then
  `∇g → 0` and `β' = β`. So the scheme has the right stationary point; the open question is
  whether it *reaches* it, which the diagnostics below are designed to answer.

Note this is a genuinely novel scheme — convergence is not guaranteed a priori, and the
plan treats that as something to measure, not assume.

## Decisions taken

- **Two modes, frozen first.** `reference_drift = "SBsquared"` collapses each stage's drift
  to the single field `b_{t=0.5}` — exactly what the existing `oblique` mode does
  ([SB/utils.py:1520](SB/utils.py#L1520)). `"SBsquared_exact"` (Phase B, below) will carry
  the full time-dependent β(t). Frozen first so the *idea* is validated before the solver
  change confounds it.
- **Smoothing behind a flag, default on** — `drift_bootstrap_smooth = true`.
- **Keep the FFD control**: still run the registration purely to populate the `FFD` baseline
  in `errors`, so SB²-runs stay three-way comparable with the existing sweep.

---

## Implementation

### 1. Per-stage drift in the annealing loop — `SB/utils.py`

The loop currently builds β **once** before the anneal ([SB/utils.py:1525](SB/utils.py#L1525))
and passes the same `beta_ipfp` to every stage. For SB² it becomes state carried across
stages:

```
beta_k = None                       # stage 0 → heat
for k, gk in enumerate(schedule):
    drifted = beta_k is not None
    n_steps_k = compute_n_steps_advdiff(mesh, gk, beta_max_k, …) if drifted
                else compute_n_steps(mesh, gk, …)
    f, g, res, res_l2 = (apply_IPFP_2d_anderson_drift(…, beta_k, n_steps_k, …) if drifted
                         else apply_IPFP_2d_anderson(…, n_steps_k, …))
    if k + 1 < len(schedule):                     # recover β for the NEXT stage
        b = (retrieve_b_2d_drift(g, 0.5, gk, mesh, beta_k, n_steps_k) if drifted
             else retrieve_b_2d(g, 0.5, gk, mesh, n_steps_k))
        beta_k = smooth(b) if drift_bootstrap_smooth else b
```

Three things must stay consistent after the loop, or the maps will describe a different
kernel than the one that produced `(f,g)`:

- `beta_ipfp` must end as the β used in the **last** stage, not the newly recovered one.
- `use_drift` for all downstream code (barycentric maps, `rho_seq`, `drift_seq`, CDI
  fallbacks) must reflect whether that last stage was drifted.
- `beta_max` feeds `_n_steps_ipfp`, so it is recomputed per stage.

**Edge case to handle explicitly:** a single-entry `gamma_sb_schedule` never reaches a
drifted stage, so `SBsquared` degenerates exactly to `null`. Detect and log it rather than
silently returning a heat run labelled as SB².

### 2. Drift recovery and smoothing

`retrieve_b_2d*` derives β from `compute_scalar_gradient_LSQ(g_t)`. Your own comment near
[SB/utils.py:565](SB/utils.py#L565) records that this cell-to-cell LSQ noise is precisely why
the barycentric CDI was made gradient-free — it "makes the drift-CDI shear along shocks".
Feeding it forward across ~12 stages is the main way this scheme could fail for a reason
that has nothing to do with the idea.

Smooth each β component with the existing heat semigroup, reusing the `n_steps_smooth`
already computed at [SB/utils.py:1314](SB/utils.py#L1314) from `smooth_gamma`/`smooth_t`
(σ ≈ 0.014 with current values). New config key `drift_bootstrap_smooth` (default `true`)
so the noise question is a one-line A/B rather than a code change.

### 3. FFD control alongside a non-FFD drift

`ffd_maps` is currently populated only when `reference_drift == "ffd"`. Gate it on
`reference_drift == "ffd" or (bootstrap and ffd_control)` instead, with `ffd_control = true`
in Config. `build_ffd_beta`'s output then feeds **only** `ffd_maps`; the kernel uses the
bootstrapped β. Registration cost is mesh-independent (fixed 144² grid, 800 Adam iters).

### 4. Diagnostics — this is what decides whether the scheme works

Per stage, into `ipfp_stages` in `metrics.json`:

- `beta_max`, `pe_max` (the Péclet block at [SB/utils.py:1532-1542](SB/utils.py#L1532-L1542),
  currently computed once, moves inside the loop)
- **`beta_delta` = ‖β_k − β_{k−1}‖ (area-weighted L²)** — the key number. If it decays along
  the ladder the bootstrap is converging to its fixed point; if it grows or plateaus high,
  it is not.
- `drift_source`: `"heat"` | `"bootstrap"`
- first-iteration residual, since warm-starting `(f,g)` across a **changing** kernel is new —
  the potentials are defined relative to a reference measure that now moves each stage.

### 5. Config, analysis, docs

- `SB/Config.toml`: document the two new `reference_drift` values; add
  `drift_bootstrap_smooth` and `ffd_control`.
- `SB/sweep_analysis.py`: add the new modes to `DRIFT_ORDER` and `DRIFT_COLOR` — `_by_drift`
  filters on exact strings, so without this SB² runs load but silently vanish from every plot.
- `CLAUDE.md`: extend the "Reference drift" section.
- Output lands at `…/drift_SBsquared/gmin<γ>/` automatically — `_mach_output_dir` already
  interpolates `reference_drift` verbatim, no change needed.

## Risks

| risk | mitigation / how it shows up |
|---|---|
| Scheme diverges or stalls | `beta_delta` per stage is the direct readout; the true solution is a fixed point, so decay is the expected signature |
| LSQ gradient noise accumulates | `drift_bootstrap_smooth` A/B |
| β grows → advective CFL → `n_steps` blows up | the 0D guard already warns past 20 000 steps; `beta_max` logged per stage |
| Warm-started `(f,g)` invalid across a moving kernel | first-iteration residual per stage |

## Verification

1. **Degeneracy check:** `SBsquared` with a single-γ schedule must reproduce `null` exactly
   (same final residual, same errors) — proves stage 0 is untouched.
2. **Continuity check:** at stage 0, `retrieve_b_2d_drift(g, t, γ, mesh, β=0, n)` must equal
   `retrieve_b_2d(g, t, γ, mesh, n)` to round-off.
3. **CPU smoke run** on h0.025 with a short ladder and `num_iter≈6`, `compute_w2=false`
   (same harness used for the Phase-0 fixes) — confirms every path executes and
   `metrics.json` carries the new per-stage fields.
4. **Head-to-head** at h0.025 against the existing `null` and `ffd` runs at matched γ_min;
   the 108-run baseline is already on disk for both.
5. **Read `beta_delta` down the ladder** — the actual scientific result.

## Phase B — `SBsquared_exact` (after Phase A works)

Carry the full β(t). `body_fn(_, phi)` → `body_fn(i, phi)` in both
[SB/advdiff_solver.py:212](SB/advdiff_solver.py#L212) and `:228`, indexing a `(K,N,2)` stack
at τ = i·dt (~14 MB at 81k cells; the per-substep gather is cheap beside the FVM stencil).

**The trap:** `Q` propagates the backward wave function (SB time = 1−τ) while `Q†` propagates
the forward one (SB time = τ), so the two solvers must index β in **opposite** time
directions. Indexing both the same way yields a wrong but entirely plausible-looking result,
so Phase B needs its own targeted test — e.g. a case with a known asymmetric-in-time drift
where the two directions give visibly different answers.

---

# PART I — Mesh-resolution study of the FFD-guided SB drift  (Phase 0 + 1 done; 2–3 on cluster)

## Context

The FFD-guided reference drift (`reference_drift="ffd"`) has only ever been validated on
one mesh: `diamond_h0.025` (20,190 cells). We want to know whether it holds up as cells
shrink, and whether the SB pipeline is even numerically robust there.

An audit of `SB/` turned up four things that must be settled before any fine-mesh run is
meaningful. Two are correctness blockers, one is a data trap, and one reframes what the
study is actually testing.

**1. The fine meshes are a different airfoil.** Verified by loading every mesh:

| mesh | cells | `metadata["height"]` | half-angle |
|---|---|---|---|
| `diamond_h0.05` … `h0.025` | 5,034 … 20,190 | 0.087489 | **5°** |
| `diamond_h0.0175` … `h0.00625` | 93,895 … 735,071 | 0.176327 | **10°** |

[Euler/diamond.py:215](Euler/diamond.py#L215) sets `alpha = radians(5)` and its loop only
covers `[0.025, 0.035, 0.05]` — the fine meshes were built with an edited, since-reverted
script. Running the ladder as-is would compare two geometries with different shock angles.
Domains agree (4×4, centre (1.5, 2.0)), so only `alpha` needs regenerating.

**2. `[drift.ffd]` in Config.toml never reaches the code.**
[SB/main.py:143](SB/main.py#L143) reads `cmach.get("drift", {}).get("ffd", {})`, i.e.
`run.mach.drift.ffd`, but [SB/Config.toml:101](SB/Config.toml#L101) declares `[drift.ffd]`
at top level. Confirmed with `tomllib`: top-level keys are `['run', 'drift']` and the
lookup returns `{}` every time, so `build_ffd_beta` always falls through to `DEFAULT_CFG`
([SB/ffd_drift.py:63-78](SB/ffd_drift.py#L63-L78)). Invisible today because the values
coincide — but it means `grid_n`/`n_cp` are currently **not tunable**, which matters below.

**3. SB output dirs do not encode the mesh.**
`_mach_output_dir` ([SB/main.py:37-62](SB/main.py#L37-L62)) builds
`outputs/Mach_interpolation/<case>/<sub>/drift_<d>/gmin<γ>` — no `h` anywhere. A refined
run silently overwrites `metrics.json` and every PNG of the coarse run at the same
(density_mode, hmode, drift, γ). There are 108 existing runs at h0.025 that would be
destroyed.

**4. What the study is really asking.** Pulling the existing eig+ffd runs:

```
    gamma    BaryCDI        FFD     Linear   conv
    1e-04     0.0032     0.0023     0.0084   True     <- BaryCDI plateau starts
    1e-03     0.0034     0.0023     0.0084   True
    1e-02     0.0054     0.0023     0.0084   True
    5e-01     0.0253     0.0023     0.0084   True
```

At h=0.025 the raw FFD registration **beats the bridge built on top of it at every γ**,
and BaryCDI saturates at ~0.0032 below γ=1e-4. Per the CLAUDE.md framing, the bridge is
not currently paying for its IPFP cost. Meanwhile the FFD drift is a 200-DOF Bernstein
field on a fixed 144² grid whose spacing is 0.0280 at h=0.025 and 0.0280 at h=0.00625 —
**h-independent by construction** ([SB/ffd_drift.py:245](SB/ffd_drift.py#L245); β is then
evaluated analytically at cell centres at [:252](SB/ffd_drift.py#L252), no grid→mesh
interpolation). So refining the mesh sharpens μ₀/μ₁ but cannot sharpen β. The concrete
hypothesis this study tests: **refinement alone will not close the SB-vs-FFD gap**, and
if so the lever is raising `grid_n`/`n_cp` — which fix 2 above is what unlocks.

## Decisions taken

- Regenerate the fine meshes at **α=5°**, preserving h0.025 and all 108 existing runs as
  the ladder's coarse anchor.
- Ladder: **h ∈ {0.025, 0.0175, 0.0125}** (20k / 94k / 184k cells). Cost scales ≈ 1/h⁴
  (cells × n_steps, since diffusive `n_steps ∝ 1/dx²`): ×1, ×9.5, ×36.
- Fine-mesh runs: `hessian_mode="eig"`, `reference_drift="ffd"` only, γ_min ∈ {1e-2, 1e-3,
  1e-4}. 3 runs per mesh. The `drift_ffd` runs already compute the `Linear` and `FFD`
  controls internally, so the three-method head-to-head survives dropping null/oblique.
- Add **C_D / C_L** as scalar quality metrics alongside L2/L∞/W₂, computed by transporting
  pressure through the same maps and integrating over the wall faces (fix F below).
- Fix correctness and mesh-awareness; add loud guards. Defer pure-performance work (frame
  vmap chunking, duplicate Delaunay hoisting, Anderson GPU-sync) — the h≤0.0125 ladder
  does not need it.

---

## Phase 0 — Code fixes (do first; all cheap)

### A. Move the FFD config block so it is actually read
`SB/Config.toml:101`: `[drift.ffd]` → `[run.mach.drift.ffd]`.

Cache impact is **nil**: the key at
[SB/ffd_drift.py:229-236](SB/ffd_drift.py#L229-L236) hashes the merged
`{**DEFAULT_CFG, **cfg}`, and the Config block's values match `DEFAULT_CFG` key-for-key
(the tuple→list conversion at `:230` makes `n_cp` agree). Existing `ffd_beta_*.npz` stay
valid.

Add a guard in `SB/main.py` alongside the read at `:143`: if a top-level `drift` key
exists in the loaded config, raise with the corrected section name. This catches the
stale `SB/_sweep_configs/*.toml` and any hand-derived config.

### B. Put the mesh in the SB output path
`SB/main.py:_mach_output_dir` — derive an `h` tag from the mesh filename
(`meshes/diamond/diamond_h0.0125.npy` → `h0.0125`) and insert it as a level:

```
outputs/Mach_interpolation/diamond/iso/hessian/eig/h0.0125/drift_ffd/gmin0.0001/
```

`_mach_output_dir` currently receives only `cmach`; the mesh lives in
`case_cfg = cmach[cmach["case"]]`, so pass `case_cfg` (or the mesh path) in. Note
`SB/utils.py` never constructs this path — it takes `output_dir` as an argument
(`:1260`, `:1279`) — so `main.py` is the only edit point.

**Do not migrate the 108 existing h0.025 directories.** They keep their current paths; the
analysis loader (fix E) reads `h` from `metrics.json`, not from the path, so legacy runs
tag themselves correctly. `make_config_for_mesh.py` consequently needs no `output_dir`
handling.

### C. Make the IPFP tolerance configurable and its convergence diagnosable
The stopping criterion is `res = max|f_new − f|`, an unnormalised L∞ over N cells
([SB/resolution.py:243](SB/resolution.py#L243) heat,
[:320](SB/resolution.py#L320) drifted), compared against a tolerance hardcoded at
`IPFP_TOL = 1e-7` in [SB/utils.py:1446](SB/utils.py#L1446). With 9× more cells the max
samples the shock and the wedge corner far more densely, so the same physical state
reports a larger residual.

Rather than guess a new number:
1. Plumb `ipfp_tol` from `Config.toml` through `run_mach_interpolation_case` (default
   1e-7). [SB/sweep_analysis.py:75](SB/sweep_analysis.py#L75) already reads
   `cfg.get("ipfp_tol", 1e-7)` — this closes a loop that was left open.
2. In both Anderson loops, also compute an area-weighted RMS residual
   `sqrt(Σ area·rf² / Σ area)` (mesh is already in scope). Keep **L∞ as the stopping
   test** so the 108 existing `residual_history` arrays stay comparable, but log both and
   write `residual_l2_history` into `metrics.json` next to `residual_history`
   ([SB/utils.py:1693](SB/utils.py#L1693) region).

That way the fine runs *show* whether L∞ stalled while L² converged, instead of us
pre-committing to a tolerance.

### D. Resolution guards in the step-count helpers
`heat_solver.compute_n_steps` ([SB/heat_solver.py:33](SB/heat_solver.py#L33)) and its twin
`advdiff_solver.compute_n_steps_advdiff` ([SB/advdiff_solver.py:54](SB/advdiff_solver.py#L54))
— the two share a byte-identical `dx_ref` block that is worth factoring out. Add warnings,
printed once per (function, gamma) so the γ-anneal loop does not spam:

- `2·gamma_diff < 9·dx_ref²` — heat kernel narrower than ~3 cells, the FVM semigroup no
  longer represents `exp(γΔ)`. At the planned ladder there is headroom (threshold is
  γ≈5e-5 at h=0.025, γ≈1.6e-5 at h=0.0125, vs a floor of γ=1e-4) but nothing currently
  asserts it.
- `n_steps` above a budget (~20,000) — print the projected kernel cost so an
  accidentally-expensive run announces itself in the first seconds of the log rather than
  after hours.

### E. Make `sweep_analysis.py` mesh-aware
- `load_runs` ([SB/sweep_analysis.py:49-101](SB/sweep_analysis.py#L49-L101)): add
  `n_cells=int(cfg["n_cells"])` and `h` parsed from `cfg["mesh"]`, falling back to
  `n_cells` if the filename does not parse. Both fields are already written into
  `metrics.json` at [SB/utils.py:1693-1695](SB/utils.py#L1693-L1695), including for the
  108 legacy runs — so this works retroactively with no re-run.
- Add `h`, `n_cells` to `CSV_COLS` ([:417](SB/sweep_analysis.py#L417)).
- `main()` ([:456](SB/sweep_analysis.py#L456)): group the existing per-hmode plots by
  `(hmode, h)` so fine and coarse curves do not overlay silently.
- Read the new `aero` block (fix F): per method, `cd_err_mean = mean|C_D − C_D_ref|` and
  `cl_abs_max = max|C_L|`. Legacy runs have no `aero` key, so the existing `.get(...)`
  pattern yields NaN and they simply drop out of the aero panels.
- Add `h`, `n_cells`, and `<method>_cd_err_mean` to `CSV_COLS`
  ([:417](SB/sweep_analysis.py#L417)).
- Add **two new figures**:
  - `plot_err_vs_h` — mean-L2 vs h, one panel per γ, three series (BaryCDI / FFD /
    Linear), log-log with a reference slope. The study's primary deliverable.
  - `plot_aero` — C_D vs t per method with the reference overlaid (at fixed h, γ), and
    mean |ΔC_D| vs h as the scalar companion to `plot_err_vs_h`.

### F. Add C_D / C_L as scalar transport-quality metrics

Field norms (L2/L∞/W₂) say how wrong a reconstruction is; C_D says whether it got the
*physics* right. For a diamond in supersonic flow the pressure-only coefficient **is** the
wave drag, which is precisely what shock transport has to place correctly — so it is a
much sharper discriminator between BaryCDI, FFD and Linear than a field norm. Two things
make this nearly free:

- `reconstruct_mach_barycentric_cdi` ([SB/utils.py:900](SB/utils.py#L900)) is
  **field-agnostic** despite its name — `mach0`/`mach1` and `interp0`/`interp1` are just
  "endpoint field + its interpolator". Feeding it pressure reuses the *same* T/S maps at
  zero extra bridge cost.
- The bundles already carry pressure as `primitives[:, 3]`
  (verified: shape (20190, 4), range 0.738–1.318), and SB already loads `prims0`/`prims1`
  at [SB/utils.py:1298-1299](SB/utils.py#L1298-L1299).

**Helper refactor.** `get_drag_coefficient`
([Euler/jax_fvm/src/helper.py:365](Euler/jax_fvm/src/helper.py#L365)) and
`get_lift_coefficient` ([:391](Euler/jax_fvm/src/helper.py#L391)) contain a byte-identical
wall-face gather (`:369-379` ≡ `:395-405`). Extract it to
`_wall_face_data(mesh) -> (cell_ids, normals, ds)` and add

```python
get_force_coefficients_from_pressure(p, mesh, q_inf, L_ref) -> (Cd, Cl)
```

then reimplement both existing functions on top of it via `getPrimitive(W)[:, 3]`. This is
behaviour-preserving — verify against the stored `C_D`/`C_L` in the existing run summaries.
A pressure entry point is what SB needs, since SB reconstructs fields, not the conservative
state `W`.

**Normalisation.** `U_inf = M·sqrt(γ·p_inf/rho_inf)`
([Euler/main.py:130](Euler/main.py#L130)), and `config.toml:5-8` fixes
`rho_inf = p_inf = 1.0`, `gamma = 1.4` across the whole Mach ladder. So
`q_inf = 0.5·γ·p_inf·M² = 0.7·M²` — SB needs no Euler config coupling beyond γ (already
present as `drift_gamma_gas`). `L_ref = mesh.metadata["obstacle_length"]` (= chord = 1.0).

**SB wiring**, in `run_mach_interpolation_case`:
- Extend `_build_cdi_interpolators` ([SB/utils.py:887-897](SB/utils.py#L887-L897)) to
  return its `barycenter_delaunay` (or a factory), so the pressure interpolators are built
  on the **already-computed triangulation**. This matters: the Delaunay is the expensive
  part (~3.6 s / +0.53 GB at 735k cells) and a second Qhull run is pure waste.
- Reconstruct a pressure sequence for each of the three methods, mirroring the existing
  Mach path: BaryCDI with the SB maps, FFD with the raw registration maps `T = x+β`,
  `S = x+α`, Linear as `(1−t)p₀ + t·p₁`. Same ordered `{name: sequence}` shape that
  `compute_interpolation_error` already consumes.
- Reference C_D/C_L come from each reference bundle's own `primitives[:, 3]` through the
  same function — nothing re-derived, so any bias in the wall integration cancels.
- Write an `aero` block into `metrics.json` next to `errors`: per method, per `t_ref`,
  `C_D` and `C_L`, plus `C_D_ref`/`C_L_ref` and the signed deltas.

**Read C_L as a symmetry check, not a discriminator.** At AoA = 0 the diamond is
symmetric, so the reference C_L ≈ 0 and a method's |C_L| is really a measure of how much
asymmetry the transport hallucinated. C_D is the scalar that carries the signal.

### G. Scope `make_config_for_mesh.py`'s rewrites to the active case
Its regexes ([SB/make_config_for_mesh.py:73-83](SB/make_config_for_mesh.py#L73-L83)) are
anchored at column 0 with no section awareness, so `^mesh\s*=` and `^bundle0\s*=` also
rewrite the `[run.mach.bump]` block at `SB/Config.toml:137-139` with diamond paths.
Harmless today (bump is inactive) but it will bite. Track the current `[...]` header while
scanning and only rewrite inside `[run.mach.<case>]`.

---

## Phase 1 — Regenerate the fine meshes at 5°

`Euler/diamond.py:212-220`: keep `alpha = radians(5)`, change the loop to
`[0.0125, 0.0175]`. Leave h0.025 alone — regenerating it would perturb the baseline that
all 108 existing runs sit on.

**Move the existing 10° meshes aside first** (`meshes/diamond/alpha10/`) rather than
letting the generator overwrite them — they are the only copies and nothing else in the
repo can rebuild them.

Verify: reload each and assert `metadata["height"] == 0.087488663525924` and
`metadata["domain"] == {"Lx": 4.0, "Ly": 4.0}`. Cell counts will differ from the 10°
versions (the refinement zone scales with `height`,
[Euler/diamond.py:97](Euler/diamond.py#L97)) — that is expected; record the actual counts,
they feed the cost estimate.

## Phase 2 — Euler snapshots

**Smoke first.** Run only M2.00 and M2.50 at h0.0125 (2 jobs) and confirm the solver
reaches t=4.00 within walltime at ~184k cells. Check the h0.025 wallclock from an existing
run summary before trusting `run_euler_refined.sbatch`'s `--time=12:00:00`
([run_euler_refined.sbatch](run_euler_refined.sbatch)) — cells go ×9.1 and the CFL step
count roughly ×1.4.

Then the full ladders:
```bash
sbatch --export=ALL,H_TAG=h0.0175 run_euler_refined.sbatch   # 11-task array
sbatch --export=ALL,H_TAG=h0.0125 run_euler_refined.sbatch
```
Both write to `results/diamond/<h_tag>/` via `setup_dirs`
([Euler/config.py:110-119](Euler/config.py#L110-L119)); its `format_h` produces exactly the
`h0.0125` spelling `H_TAG` uses, provided `metadata["h"]` matches the filename (assert this
in Phase 1). Expect at least one Mach to stop early on the stationarity check and carry a
non-`t4.00` suffix, as `AOA0.00_M2.35_..._t2.74.npz` does at h0.025 — that is exactly why
`make_config_for_mesh.py` globs.

## Phase 3 — SB runs

Between the smoke Euler pair and the full ladder, do an **SB smoke run**: a hand-made
config pointing at the two h0.0125 endpoints with `ref_bundles = []` (error tables are
skipped when references are absent) and `compute_w2 = false`. This exercises the IPFP, the
FFD registration, the barycentric maps and the reconstruction at 184k cells without
needing the other 9 snapshots — it is the cheapest way to answer the robustness question
outright.

Then, per mesh:
```bash
uv run python SB/make_config_for_mesh.py --h-tag h0.0175   # → SB/Config_h0.0175.toml
uv run python SB/make_config_for_mesh.py --h-tag h0.0125
```
and drive `run_sb_sweep_refined.sbatch` with its `HMODES`/`DRIFTS`/`GAMMAS` narrowed to
`(eig)`, `(ffd)`, and the ladder prefixes ending at `{1e-2, 1e-3, 1e-4}` → `--array=0-2`,
3 runs per mesh. The script's header at
[run_sb_sweep_refined.sbatch:34-36](run_sb_sweep_refined.sbatch#L34-L36) points at a
"FOCUSED SUBSET" that was never written down anywhere — write it into the script's header
so it stops being a dangling reference.

Keep `compute_w2 = true` here: at h0.0125 the ×36 factor is affordable, and W₂ is the
metric that actually distinguishes transport quality. Its cost is roughly comparable to
the bridge solve itself ([SB/utils.py:470](SB/utils.py#L470), 63 Sinkhorn calls per run),
so it is the first thing to drop if walltime bites.

**Re-run the three coarse points too.** The 108 legacy h0.025 runs predate the `aero`
block and the maps are not stored, so C_D cannot be backfilled from them. Re-run the
h0.025 eig/ffd configs at γ_min ∈ {1e-2, 1e-3, 1e-4} — ~3.5 min each at 20k cells — so all
three rungs of the ladder carry aero. Their L2/W₂ numbers should reproduce the stored
values, which doubles as a regression test on the Phase 0 refactors.

## Phase 4 — Analysis

```bash
uv run python SB/sweep_analysis.py --root outputs/Mach_interpolation/diamond/iso/hessian
```
Reads all 108 legacy + 9 new runs, now split by `h`. The questions to answer off
`err_vs_h`:
1. Does BaryCDI's error fall with h, and at what order?
2. Does the FFD control fall too, or flatten? (It should flatten — β is h-independent.)
3. Does the SB-vs-FFD gap close, widen, or hold?
4. Does the γ plateau entry (γ≈1e-4 at h=0.025) move with h?

And off `plot_aero`, the question the field norms cannot answer:

5. Does any method reproduce the reference C_D(t) curve, and does refinement help? A
   method can win on L2 while misplacing the shock feet on the wedge surface — that shows
   up in C_D and nowhere else. This is also the check that decides whether the ranking
   from the L2 tables is physically meaningful at all.

If (3) shows the gap holding, the follow-up is a `grid_n`/`n_cp` sweep at fixed h — which
Phase 0A makes possible for the first time.

---

## Verification

- **0A**: `uv run python -c "import tomllib; c=tomllib.load(open('SB/Config.toml','rb')); print(c['run']['mach']['drift']['ffd'])"` must print the full block, and `'drift' not in c`. Re-run one existing h0.025 eig/ffd config and confirm the log reports a cache **hit** on `ffd_beta_*.npz` (proves the merged config dict is unchanged) and that the resulting L2 errors match the stored `metrics.json` to round-off.
- **0B**: run any Mach case and confirm the printed output dir contains the `h` level; confirm the 108 legacy directories are untouched (`git status outputs/` clean apart from intended writes).
- **0C**: `metrics.json` gains `residual_l2_history` of the same length as `residual_history`; setting `ipfp_tol` in Config.toml changes the reported `config.ipfp_tol` and the iteration count.
- **0D**: force a guard by calling `compute_n_steps(mesh_h0025, 1e-6)` — should warn (threshold γ≈5e-5); `compute_n_steps(mesh_h0025, 5e-3)` should not.
- **0F**: the refactored `get_drag_coefficient`/`get_lift_coefficient` must reproduce the `C_D`/`C_L` already stored in the Euler run summaries under `results/diamond/h0.025/` to round-off. Independently, feeding a reference bundle's own `primitives[:, 3]` through `get_force_coefficients_from_pressure` with `q_inf = 0.7·M²` must give the same numbers as the `W`-based path on that bundle's `conservatives` — that is the check that the pressure entry point and the normalisation are both right. Confirm `t=0` and `t=1` reconstructions return C_D exactly equal to the endpoint bundles' (the CDI formula is exact at the endpoints, so this catches interpolator/Delaunay wiring errors).
- **0E/0G**: `sweep_analysis.py` on the untouched 108-run tree must produce identical numbers to the current CSVs plus the new `h`/`n_cells` columns, all reading `h=0.025`, with the aero columns NaN. `make_config_for_mesh.py --h-tag h0.0125` output diffed against the base config should touch only lines inside `[run.mach.diamond]`.
- **Phase 1**: assert `height == 0.087488663525924` on both regenerated meshes.
- **Phase 2**: 11 bundles per h_tag under `results/diamond/<h_tag>/`, each with `n_cells` matching the mesh.
- **Phase 3**: the smoke run completes at 184k cells without OOM; then 3 `metrics.json` per mesh × 3 meshes, each recording the right `mesh`/`n_cells` and a populated `aero` block. The 3 re-run coarse points must reproduce their stored L2/W₂ values.
- **Phase 4**: `err_vs_h` and the |ΔC_D|-vs-h panel each show three h values per γ; the reference C_L is ≈0 at every t (symmetry sanity check on the wall integration).
