# Warm-starting the Euler solver from interpolated snapshots

> **Measurements during implementation changed two decisions.** Recorded here
> because both invalidate assumptions in the sections below.
>
> **1. The solver has a residual floor; there is no tight steady state to race to.**
> Seeding the solver with its own converged M2.00 corpus bundle and running 10619
> more steps (4 more time units) changed the operator residual by **0.95×**
> (2.538e+01 → 2.660e+01, i.e. not at all) and moved the state by only
> `‖ΔW‖/‖W‖ = 2.6e-04`. The corpus bundles sit at `‖R‖/‖R_freestream‖ ≈ 1.6e-02`
> and stay there. This is a genuine floor (MUSCL limiter chatter is the classic
> cause), not under-convergence. Consequences:
> - The endpoint unit test as written cannot pass at `1e-6`: the *exact* answer
>   does not satisfy that criterion. Threshold `1e-6` is below the reachable range.
> - An operator-residual criterion (`‖R(W)‖/‖R₀‖`) would be **worse**, not better —
>   it plateaus at 1.6e-2 and would never converge. The change-based criterion the
>   solver already uses is the right one.
> - The achievable `rel` floor scales roughly as `1/√(check_every)` because an
>   oscillatory residual cancels over a longer block. Threshold and
>   `stationarity_check_every` are **coupled** and cannot be chosen separately —
>   which is also why the corpus convention pairs `check_every=10000` with `1e-6`.
>
> **2. Therefore: record the convergence curve, never a single threshold.**
> `Euler/main.py` now stacks the per-check `rel` into the scan's ys and
> `--save-convergence` writes `convergence_<snapshot>.npz`. "Iterations to reach
> tolerance X" is read off afterwards for any X.
>
> **This also dissolves the `lax.scan` tail problem** (confound 4), which the
> measurements showed was real and serious: a skipped step costs 3.2 ms against
> 38.4 ms for a real one (**8.4 %**, not the <1 % estimated below), capping any
> early-stopped wall-clock speedup at ~12× regardless of seed quality. The
> benchmark therefore runs **every variant to the same fixed N with
> `--stationarity-threshold 0`** — nothing stops early, so identical trip count,
> identical compile, identical wall time, and no tail confound can enter. The
> saving is `(steps_cold − steps_warm) × per-step cost`, where the per-step cost
> is measured directly as `wall_time/N` and is common to all variants.
> **No restructuring of the validated solver is needed.**

## Context

The project's thesis is that SB interpolation lets you get CFD states without re-running
the PDE solver. The accuracy half of that claim is already measured — and it is negative:
SB does not beat the FFD baseline at any γ or any mesh. What has never been measured is
the **acceleration** half: if you seed the high-resolution Euler solve with an interpolated
field instead of uniform freestream, how many iterations does it save?

That is a separate and possibly stronger claim. A warm start does not have to be *accurate*
to be *useful* — it only has to be closer to the steady state than a uniform freestream, in
the basin of the same attractor. And it gives the three methods a second, physically
meaningful ranking axis that does not depend on the L2-vs-reference tables.

**Goal:** for each of the 9 target Mach numbers M2.05…M2.45 (diamond, h0.025), run the
solver four ways — cold (uniform freestream), Linear-seeded, FFD-seeded, BaryCDI-seeded —
and report iterations and wall time to the *same* steady state.

## The blocker

**SB cannot currently produce a solver-ready state.** `fields.npz` ([SB/utils.py:2498-2512](SB/utils.py#L2498-L2512))
stores `mach` and `press` only — 2 of 4 degrees of freedom. From (M, p) you cannot recover
ρ, and the velocity *direction* is gone entirely. The in-code comment at
[SB/utils.py:2312-2316](SB/utils.py#L2312-L2316) states the design decision plainly:
"SB reconstructs fields, never the conservative state W."

This is plumbing, not mathematics. The transport maps `T`/`S` are `(N,2)` absolute target
positions on cell barycenters — purely geometric, no Mach dependence — and the CDI formula
[`reconstruct_mach_barycentric_cdi`](SB/utils.py#L1365-L1402) takes the field only as an
interpolator payload. Pressure is *already* pushed through the identical machinery at
[SB/utils.py:2317-2341](SB/utils.py#L2317-L2341). That block is the working template.

A second, worse gap: **`T_map`/`S_map` are never persisted** — they die with the process.
Only `ffd_beta_<hash>.npz` survives. So the warm-start state must be produced *inside* the
SB run; it cannot be backfilled from the ~600 finished runs without re-running the IPFP.

## Verified numbers (diamond, h0.025, tf=4.0, CFL=0.6)

| quantity | value |
|---|---|
| cells | 20190 |
| `dt` from uniform freestream @ M2.25 | 3.5223e-04 → **N = 11357 steps** |
| `dt` from the converged M2.25 field | 3.4989e-04 (0.7 % smaller, N = 11433) |
| `stationarity_check_every` (config.toml:21) | 10000 → **one interior check in the whole run** |
| corpus convention (`run_euler_grid_big.sbatch:248`) | `--stationarity-threshold 0` → **stop path is dead** |
| mesh `metadata["solver_mesh"]` | absent → no nested-mesh complication in this scope |

Two consequences that shape the whole design: the seed-dependent `dt` shift is real but
tiny (0.7 %), so pinning `dt` is cheap; and the stationarity machinery as configured today
*cannot resolve a speedup at all*.

---

## Phase 1 — SB exports a solver-ready state

### 1a. Generalize the CDI to vector fields

[`reconstruct_mach_barycentric_cdi`](SB/utils.py#L1365-L1402) works on `(N,)` fields. For
`(N,4)` it breaks at exactly one line:

```python
bad0 = inside_body(W)  | ~np.isfinite(M0)     # (N,) | (N,4)  -> ValueError
```

(Verified: `operands could not be broadcast together with shapes (10,) (10,4)`.)

Fix, and make the fallback **atomic per cell** so components can never disagree:

```python
nan0 = ~np.isfinite(M0)
if nan0.ndim > 1:
    nan0 = nan0.any(axis=-1)          # any bad component condemns the whole cell
bad0 = inside_body(W) | nan0
M0[bad0] = f0[bad0]                   # whole 4-vector reverts together
```

A cell must not take ρ from the transported point and p from the undisplaced one; that
produces a state satisfying neither. `scipy`'s `LinearNDInterpolator` already accepts
`(N,k)` values natively (verified), so this is **one interpolator pair per endpoint, not
four**.

### 1b. Transport the primitives and assemble W

Add a block after the pressure block in `run_mach_interpolation_case`, reusing the existing
`barycenter_delaunay` (the Qhull build is the expensive part and is already done at
[SB/utils.py:2160](SB/utils.py#L2160)):

```python
interp0_W, interp1_W, _, _, _ = build_cdi_interpolators(
    barycenters, prims0, prims1, barycenter_delaunay, mesh)   # (N,4) payloads
```

Then per method, per frame, transport `(ρ,u,v,p)` through that method's `T`/`S` and assemble

```
W = [ρ, ρu, ρv, p/(γ−1) + ½ρ(u²+v²)]      # matches helper.getConserved with M=1.0
```

**Transport primitives, not conservatives** — even though `conservatives` sits unread in
every bundle. Reasons: (i) the CDI blend is a convex combination of *sampled* values, so
ρ>0 and p>0 are inherited from positive endpoints, whereas independently blending ρ and E
can yield `E < ½ρ|V|²` (negative pressure) with positive inputs; (ii) `getConserved` then
makes the energy consistent with the transported ρ, u, v, p by construction.

### 1c. Repair must be visible, never silent

The CDI blend is not conservative and the components are transported independently, so
admissibility is not guaranteed even from primitives. Clamp `ρ ≥ ρ_floor`, `p ≥ p_floor`
(e.g. `1e-6 × freestream`), and **count every clamped cell**, writing the counts into
`metrics.json` under a new `warmstart` key:

```
warmstart: {method: {n_rho_clamped, n_p_clamped, n_fallback_cells, rho_min, p_min}}
```

A seed that needed repair on 3 % of cells is a different object from a clean one, and the
speedup table must be readable alongside that fact.

### 1d. Persist

Write two new files next to `fields.npz`, gated on a new `save_warmstart` key in
`[run.mach]`. Follow the `save_fields` pattern exactly: a trailing keyword arg on
[`run_mach_interpolation_case`](SB/utils.py#L1723-L1749) plumbed from
`cmach.get("save_warmstart", False)` at [SB/main.py:263](SB/main.py#L263). Default **off** —
at h0.025 the file is `3 × 11 × 20190 × 4 × 8 B ≈ 21 MB` before compression, and ~4× that at
h0.0125, so it should be opt-in rather than added to every sweep run.

- **`warmstart.npz`** — `conservatives (n_methods, n_t, N, 4) float64`, `t`, `method_names`,
  `mach_target (n_t,)` = `(1−t)·mach0_inlet + t·mach1_inlet`, `aoa_target`, `mesh_path`,
  `n_cells`, `barycenter (N,2)` (for the load-time consistency check).
  float64 here, unlike `fields.npz` — this is solver input, not a plot.
- **`maps.npz`** — `T_map`, `S_map` per method, `barycenter`. Cheap insurance: it makes any
  *future* field transport free instead of costing another γ-annealed IPFP.

**One SB run produces all three seeds.** `ffd_control` defaults to `true`
([SB/Config.toml:125](SB/Config.toml#L125)), so `ffd_maps` is populated whatever the drift
([SB/utils.py:1900-1910](SB/utils.py#L1900-L1910)) and `methods_dict` carries BaryCDI,
Linear and FFD together. A single `SB/main.py` run on the diamond M2.00→2.50 pair therefore
emits a `warmstart.npz` holding `3 methods × 11 frames` — no separate run per method.

---

## Phase 2 — Euler accepts an initial state from file

### 2a. CLI + loader

`Euler/config.py` ([build_config_from_cli](Euler/config.py#L23-L60)) gains:

| flag | meaning |
|---|---|
| `--init-state PATH` | `warmstart.npz` (or any npz with `conservatives`) |
| `--init-method NAME` | `BaryCDI` \| `FFD` \| `Linear` — selects along axis 0 |
| `--init-t FLOAT` | selects along axis 1 by nearest `t` |
| `--dt FLOAT` | explicit timestep override (see the timing protocol) |

`Euler/main.py` gains `load_initial_state(mesh, cfg)`:

- returns `None` when `--init-state` is absent → caller falls back to `initialize()`
- asserts `N == mesh.area.shape[0]` and that `barycenter` matches `mesh.barycenter`
  to a tight tolerance — the cheap guard against seeding from the wrong mesh
- if `mesh.metadata["solver_mesh"]` is set, scatters to the child mesh with
  `W_solver = W_parent[parent]` (the adjoint of
  [`average_to_parent`](Euler/utils.py#L21-L33)). Not needed for diamond h0.025, but the
  assertion should exist rather than silently producing a shape error.

### 2b. Benchmark output must NOT land in `results/<case>/<h>/`

This is the sharpest hazard in the whole task. [`setup_dirs`](Euler/config.py#L148-L170)
hard-pins `repo_root/results/<case>/<h_dir>`, and its comment says the layout is deliberate:
the resume-glob in `run_euler_grid_big.sbatch` and
`SB/make_config_for_mesh.py:find_bundle` both glob exactly that path. Meanwhile
[`format_snapshot_name`](Euler/config.py#L121-L127) encodes only case/AoA/Mach/flux/
reconstruction/scheme/time — **nothing distinguishes the four seeds**. So writing the
benchmark into the default location would:

- have all four variants of one Mach collide on the same bundle and summary filenames;
- drop half-converged, early-stopped runs into the corpus that `find_bundle` scans, where
  SB could later pick one up as a *reference* bundle;
- risk the sbatch stale-check quarantining good bundles (the failure mode CLAUDE.md warns
  about for mesh-tag variants).

Two changes, both small:

- **`--out-root PATH`** honored by `setup_dirs` (default `repo_root`, preserving today's
  behaviour exactly). The benchmark passes `outputs/warmstart_bench/`, keeping the same
  `<case>/<h>` shape one level down so nothing that globs `results/` sees it.
- **`cfg["snapshot_prefix"]`** wired into the already-present but unused `prefix = ""` hook
  at [config.py:126](Euler/config.py#L126), set to `cold_` / `linear_` / `ffd_` / `barycdi_`
  so the four variants of a Mach cannot overwrite each other.

### 2c. Two things that must NOT change

- **`inlet` always comes from `initialize()`.** It is the farfield Dirichlet value
  (`kw["value"]`, [main.py:56](Euler/main.py#L56)) and the sponge target; it must stay the
  target freestream regardless of the seed.
- **`W_initial` for `get_entropy_creation`** ([main.py:47,168](Euler/main.py#L47)) should
  stay the *freestream* field. Otherwise ΔS silently becomes "entropy relative to the seed"
  and is no longer comparable to the existing corpus.

Call site ([main.py:188-189](Euler/main.py#L188-L189)):

```python
W, inlet = initialize(solver_mesh, CFG)          # inlet + W_initial + dt reference
W_seed = load_initial_state(solver_mesh, CFG)
run(W_seed if W_seed is not None else W, solver_mesh, inlet, CFG, out_dirs,
    export_mesh=mesh, parent=parent, W_freestream=W)
```

`W_freestream` must be added as a **keyword arg defaulting to `None`** (falling back to `W`).
The only external caller is [slurm/run_euler_grid.py:84-85](slurm/run_euler_grid.py#L84-L85),
which calls `run(W, mesh, inlet, cfg, out_dirs)` positionally — a defaulted kwarg leaves it
working unchanged.

---

## Phase 3 — A fair timing protocol

This is where the experiment is won or lost. Six confounds, each with a resolution:

**1. `dt` depends on the seed.** [`get_dt`](Euler/jax_fvm/src/helper.py#L8-L18) takes a
global `jnp.min` over cells, so a non-uniform seed gives a different `dt`, a different
`N`, a different `final_time = stopping_step·dt`, and therefore a different `t{:.2f}` in the
snapshot filename. Measured shift: **0.7 %**. Resolution: the driver computes `dt` once from
the freestream state at the target Mach and passes the *same* `--dt` to all four runs.
Guard against the seed being the stiffer state by taking `dt = min(dt_freestream, dt_seed)`
across all four seeds before the sweep, then pinning that one value.

**2. `stationarity_check_every = 10000` vs `N = 11357`.** As configured there is exactly one
interior check — the stopping step is quantized to a value coarser than the whole run.
Resolution: `--stationarity-check-every 100` (≈114 checks). Each check is two Frobenius
norms over a `(20190,4)` array — negligible beside a MUSCL/HLLC RHS.

**3. The corpus passes `--stationarity-threshold 0`, deliberately.** The comment at
[run_euler_grid_big.sbatch:237-242](run_euler_grid_big.sbatch#L237-L242) explains why:
the residual is normalised by *step count*, not elapsed time, so the criterion is
`(physical rate)·dt` — a finer mesh reports a smaller residual for the same physical state
and stops earlier in physical time, which is fatal for a cross-mesh study.

That caveat is about comparing **across meshes**. This benchmark compares four seeds on
**one mesh at one pinned `dt`**, where the `dt` scaling is a shared constant and cancels —
which is exactly why confound 1 must be resolved by pinning `dt` rather than by letting
each seed pick its own. Resolution: pass a real threshold here (start at `1e-6`, the
`config.toml` default) and leave the sbatch convention alone; the existing corpus depends
on fixed-`tf` runs.

**4. `lax.scan` always runs N trips.** Early stop only flips a flag; the
[`lax.cond`](Euler/main.py#L106) makes the body a cheap copy but the trip count is static,
so `wall_time_s` includes a post-stop tail. Estimated magnitude: ~2.2 ms/step for the real
body (25 s / 11357 steps) vs a 323 KB copy plus scan overhead for the skipped ones —
expected well under 1 % of a solve. **Measure it before touching anything:** time a run
with `--stationarity-threshold 1e30` (stops at the first check) and compare against the
per-step cost. Only if the tail exceeds ~5 % is a chunked outer loop worth the risk of
perturbing a solver that produced ~700 existing bundles.

**5. The residual rewards the wrong thing.** `rel = ‖W_next − W_ref‖ / (block_steps·‖W_ref‖)`
is a *rate of change*, not a distance from the steady state. A seed that is geometrically
close but is not a discrete fixed point can show a large initial transient, and — worse — a
*bad* seed sitting in a slow-moving region can look "stationary" while being far from the
answer. Resolution: record, alongside `stopping_step`, the **distance to the converged
reference** `‖W_final − W_ref‖/‖W_ref‖` using the existing M2.05…M2.45 bundles. The speedup
claim is only valid for runs that land on the same steady state, and Phase 4 gates on that
explicitly.

**6. JIT compile.** Excluded from `wall_time_s` by the warm-up at
[main.py:95](Euler/main.py#L95). All four variants have identical shapes and identical
`kw`, so they compile the same program — the exclusion is fair. Confirm by checking the
four runs report comparable time for the first timed step.

Also: `export.summary` is `false` in [config.toml:44](Euler/config.toml#L44) so nothing is
recorded today. Turn it on for this experiment —
[`build_run_summary`](Euler/utils.py#L162-L182) already emits `wall_time_s`,
`stopping_step`, `converged`, `stationarity_rel`, `cd`, `cl`, and
`Euler/postproc/convergence.py:66-130` already aggregates `summary_*.json` to CSV.

---

## Phase 4 — Benchmark driver

`slurm/run_warmstart_bench.py`, following the subprocess/config pattern of
[slurm/run_euler_grid.py](slurm/run_euler_grid.py) (`parse_range`, `build_case_cfg`,
direct `euler_main.run` calls):

1. For each of the 9 targets M2.05…M2.45: pin `dt`, then run 4 variants
   {cold, Linear, FFD, BaryCDI} → 36 solves.
2. Collect `summary_*.json`; join each run to its reference bundle.
   [`load_summary_rows`](Euler/postproc/convergence.py#L66-L130) is reusable for the
   reading, but note its `SUMMARY_COLUMNS` list
   ([convergence.py:19-32](Euler/postproc/convergence.py#L19-L32)) omits `stopping_step`
   and `converged` even though `build_run_summary` writes them — the driver should emit its
   own CSV rather than extend that list, which belongs to the mesh-convergence study.
3. Emit `warmstart_bench.csv`: `mach, init, dt, N, stopping_step, converged, wall_time_s,
   cd, cl, cd_ref, cl_ref, l2_vs_ref, n_clamped`.
4. Plots: steps-to-stationarity vs Mach per init; speedup factor vs cold; and a
   correlation plot of *interpolation* L2 error (from SB `metrics.json`) against
   *speedup* — the interesting scientific question is whether a more accurate
   interpolant actually buys more solver acceleration, and memory says SB loses on the
   first axis, so the two need not rank the same way.

---

## Verification

Run in this order; each step gates the next.

1. **Endpoint identity (the unit test of the whole pipeline).** Seed at `t=0`, which the
   CDI returns as `prims0` exactly, and run at M=2.00. The state is already a discrete
   fixed point of the scheme, so the run must converge at the first check and reproduce
   `AOA0.00_M2.00_...npz` to round-off. If this fails, nothing downstream means anything —
   the bug is in indexing, mesh ordering, or the prim→cons assembly.
2. **Endpoint identity at `t=1`** (M=2.50), same argument. Catches a T/S swap that `t=0`
   would not.
3. **Admissibility.** For all 9×3 seeds assert `ρ>0`, `p>0`, `E > ½ρ|V|²` before any
   clamping, and report the clamp counts. A method needing widespread repair is a finding,
   not a nuisance.
4. **Same steady state.** For every warm-started run, `|C_D − C_D_cold|/C_D_cold < 1e-3`
   and `‖W_warm − W_cold‖/‖W_cold‖ < 1e-4`. Any run failing this is excluded from the
   speedup table and reported separately — a seed that converges fast to a *different*
   state is a spurious win.
5. **Tail-cost measurement** (confound 4) before trusting any wall-clock number.
6. **Headline.** Mean and per-Mach speedup for each of Linear / FFD / BaryCDI vs cold.

**Environment note:** this machine has no CUDA-enabled jaxlib — JAX falls back to CPU
(`cuInit(0) failed: CUDA_ERROR_NO_DEVICE`). 36 solves × 11357 steps on CPU will be slow and
the timings would not be comparable to the GPU corpus. Run the benchmark on CALI3, from
`/scratch/aaboufadel` per the storage constraint. The SB side (Phase 1) can be developed and
smoke-tested locally at a coarse mesh.

## Out of scope for this pass

Bump case, the full Mach-band sweep, and AoA-axis warm starts — all reachable once Phase 1
emits `warmstart.npz`, but the single diamond pair is the cheapest thing that can answer
the question.
