# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GenerativePDE is a research project combining:
1. **Euler/** — a JAX-based 2D compressible Euler equation solver using the Finite Volume Method (FVM)
2. **SB/** — a Schrödinger Bridge (SB) interpolation module that transports probability densities between two physical flow states (e.g., Mach fields) using the IPFP algorithm

The overall goal is generative PDE solving: use the Euler solver to generate CFD snapshots at various Mach numbers/AoA, then use SB to interpolate between them without re-running the PDE solver.

## Environment & Dependencies

- **Python 3.12** (`requires-python = ">=3.12,<3.13"`), managed with `uv`. (Was 3.11;
  bumped because the `phdtruel` FFD framework used by `reference_drift="ffd"` needs
  `typing.override` + a PEP 695 `type` alias, both 3.12-only.)
- JAX with CUDA (GPU required for production runs)
- Key packages: `jax[cuda12]`, `numpy<2`, `scipy`, `matplotlib`, `meshpy`, `meshio`,
  `scikit-learn`, plus the FFD-drift deps (`chex`, `optax`, `jaxtyping`, `interpax`,
  `pycpd`, `pot`, `hickle`, `h5py`, `pyyaml`) that the vendored `phdtruel/` package needs.
- `phdtruel/` is a vendored JAX framework (imported via `sys.path`, not pip-installed);
  its FFD-registration mappings back `reference_drift="ffd"`. Its runtime deps must be
  **explicit** in `pyproject.toml` (it ships no metadata, so they resolve as orphans).
- `marker-pdf` lives in the optional `docs` dependency group, NOT the default install —
  it pulls `torch → nvidia-cudnn-cu13`, which collides with jax's `nvidia-cudnn-cu12`
  (`undefined symbol: cudnnGetLibConfig`). Install on demand with `uv sync --group docs`.

**CUDA on the cluster**: `jax[cuda12]` wheels bundle their own CUDA + cuDNN and run on
newer drivers (e.g. 13.2) via forward-compat. Do **not** `module load` a system CUDA
toolkit — its older `libcudnn` shadows the pip one and reproduces the `cudnnGetLibConfig`
error. In SLURM scripts: `module purge`, then `export LD_LIBRARY_PATH=""`. After any
dependency change, `rm -rf .venv` on the cluster so `uv` rebuilds cleanly (rsync-based
sync typically excludes `.venv`, so a stale env persists otherwise).

```bash
# Install dependencies
uv sync

# Run any script
uv run python Euler/main.py --help
uv run python SB/main.py
```

## Common Commands

### Euler Solver — Single Run
```bash
uv run python Euler/main.py \
  --case bump \
  --mach 1.3 \
  --mesh-path meshes/bump/bump_h0.02.npy \
  --flux HLLC \
  --time-scheme SRK2 \
  --reconstruction MUSCL
```
CLI options: `--case {bump,diamond}`, `--mach`, `--mesh-path`, `--flux {Rusanov,Tadmor,AUSM,Roe,HLLC}`, `--time-scheme {EE,RK2,RK4,SRK2,SSP_RK2}`, `--reconstruction {constant,MUSCL}`, `--aoa` (diamond only).

### Euler Solver — Grid of Cases
```bash
uv run python slurm/run_euler_grid.py \
  --case diamond \
  --mesh-path meshes/diamond/diamond_h0.05.npy \
  --mach 0.7:1.5:0.1 \
  --aoa 0:10:2
```
`--mach` and `--aoa` accept `start:stop:step`, comma-separated lists, or a single value.

### Mesh Generation
```bash
uv run python Euler/generate_mesh.py --geom diamond -h 0.025 --out-dir meshes/diamond
uv run python Euler/generate_mesh.py --geom naca0012 -h 0.025 --nose-factor 7 --out-dir meshes/naca0012
                         # -> naca0012_h0.025le7{,_solver}.npy: see "Rounded-LE airfoils" below
# without --out-dir, meshes go to data/meshes/; the grid sbatch generates on demand
```

### Schrödinger Bridge Interpolation
```bash
JAX_ENABLE_X64=true uv run python SB/main.py
```
`SB/main.py` takes **no command-line arguments** — every run parameter (which case to run,
1D/2D/Mach settings, density mode, drift mode, etc.) is read from `SB/Config.toml`, or from
the file named by the `SB_CONFIG` env var if set (used to run several configs in one job —
see the SLURM section). Edit `Config.toml`'s `run.case` to select a pipeline (1D synthetic,
2D synthetic, or `2d_mach_interpolation`) and `run.mach.*` to configure the Mach interpolation
(density mode, field source, **reference drift**, γ-schedule, bundle/mesh paths per
`run.mach.case`).

**Important**: `jax_enable_x64=True` must be set before any JAX arrays are created (done at module import in `SB/main.py`). Omitting this causes IPFP probability truncation bugs.

**Reference drift** (`run.mach.reference_drift`) conditions the bridge so it converges at
small γ (see "Reference drift" under Architecture):
- `"null"`  — pure-heat bridge (β≡0). Keep `gamma_sb_schedule` γ_min ≈ **0.005**; the
  heat bridge does not converge much lower near the OT limit (residual stalls, kernel
  collapses → garbage maps).
- `"oblique"` — analytic θ-β-M diamond drift. Anneal down to **1e-4** (requires
  `mach0_inlet`/`mach1_inlet`).
- `"ffd"` — data-driven drift from FFD registration of the two marginals via `phdtruel`.
  Anneal down to **1e-4**. β is cached to the output dir (`ffd_beta_<hash>.npz`).
- `"SBsquared"` — **bootstrapped / self-conditioned** drift: no external β at all. Stage 0
  of the γ-anneal is a pure-heat bridge; every later stage takes its reference drift from
  the *previous* stage's own SB solution, `β_{k+1} = b^(k) = β^(k) + 2γ_k ∇g_t`
  (`retrieve_b_2d_drift` returns that total drift directly), frozen at t=0.5 like
  `"oblique"`. **Requires ≥2 entries in `gamma_sb_schedule`** — with one stage it
  degenerates to `"null"` and logs a `[WARN]`. The true SB solution is a fixed point of
  this map (β optimal ⇒ ∇g→0 ⇒ β'=β), so `drift.beta_delta_history` in `metrics.json`
  decaying down the ladder is the signature that it is converging.
  `drift_bootstrap_smooth` (default true) heat-smooths the recovered β each stage, because
  it comes from `compute_scalar_gradient_LSQ` — the same LSQ noise the barycentric CDI was
  built gradient-free to avoid, which would otherwise compound over a dozen stages.
- `"SBsquared_exact"` — the same bootstrap carrying the full time-dependent β(t) through
  the kernel (a `(K,N,2)` stack at `drift_bootstrap_nt` uniform SB times). `Q` indexes it
  BACKWARD in SB time (1−τ) and `Q†` FORWARD (τ); the mirroring keeps ⟨Qu,v⟩=⟨u,Q†v⟩
  exact (verified 7e-15; an un-mirrored control fails by 183%).

**Bootstrap seeding** (`drift_bootstrap_seed` = `"null"|"ffd"|"oblique"`, with
`drift_bootstrap_start_gamma`): seeds β from the FFD registration (or the analytic
drift) and holds it FIXED until the next stage's γ drops strictly below the switch —
those stages are literally a `drift_ffd` run — then the bootstrap refines from there.
This targets the measured mid-ladder failure of the heat-seeded bootstrap (worst drift
from γ=5e-2 to 2e-3, worse than Linear at 2e-2, because stage 0 recovers a huge
void-dominated 2γ∇g from the heat bridge). Switch `inf` = bootstrap from the start
(original behaviour), `0` = never engage (a seeded run is then exactly `drift_ffd` —
the degeneracy check). Seeded runs write to `drift_<mode>+<seed>/` so they never
collide with unseeded ones. In sweep sbatch scripts, composite `DRIFTS` entries like
`SBsquared_exact+ffd` split into (reference_drift, seed) automatically, with the
switch from `BOOT_SWITCH` (default 1e-3).

`ffd_control` (default true) runs the FFD registration even when it is not driving the
kernel, purely to populate the γ-independent `FFD` control interpolator — so `SBsquared`
and `oblique` runs stay three-way comparable (BaryCDI vs Linear vs FFD) with `ffd` runs.

### Linting & Type Checking
```bash
uv run ruff check .
uv run mypy Euler/ SB/
```

### HPC (SLURM)
```bash
sbatch bump_test.sbatch           # Euler bump case
sbatch run_diamond_test.sbatch    # Euler diamond case
sbatch run_sb.sbatch              # SB Mach interpolation
```
`run_sb.sbatch` runs the **oblique, ffd and null** drift cases in one job. All shared
parameters come from `SB/Config.toml` (single source of truth); the script derives one
config per case with `sed`, overriding only `reference_drift` and `gamma_sb_schedule`
(oblique/ffd → `[0.0003, 0.0002, 0.0001]`, null → `[0.02, 0.01, 0.005]`), and points
`SB_CONFIG` at each. `main.py` routes each to its own output dir
(`…/drift_<mode>/gmin<γ>/`), so the runs never clobber each other. A failing case logs a
`[WARN]` and the others still run.

The script does `module purge` + `export LD_LIBRARY_PATH=""` (no system CUDA module) — see
the CUDA note under Environment.

**Full Mach × AoA survey** (`run_euler_grid_big.sbatch` → `run_sb_grid_big.sbatch`):
sweeps every 0.10-wide Mach band from 0.80 to 3.00 plus four AoA bands, for every drift.

```bash
sbatch run_euler_grid_big.sbatch                     # 481 HR solves, 8/task => 61 tasks
sbatch run_sb_grid_big.sbatch                        # 576 runs, 12/task => 48 tasks
CASE=bump sbatch --array=0-27 run_euler_grid_big.sbatch   # 221 solves => 28 tasks
CASE=bump sbatch --array=0-16 run_sb_grid_big.sbatch      # 198 runs   => 17 tasks
```

Both scripts take `CASE=diamond|bump`. The **bump differs in three ways**, all handled
by the case defaults rather than by separate scripts:
- **No incidence axis.** The bump sits on a channel wall, so AoA is meaningless;
  `Euler/config.py:format_condition_tag` reflects this by omitting the AoA from bump
  snapshot names (`M2.00_…` vs diamond's `AOA0.00_M2.00_…`). `find_bundle` and
  `_bundle_mach` are case-aware for exactly this reason — before the fix,
  `_bundle_mach`'s `_M(...)` anchor missed every bump name and returned 0.0, which
  would have collapsed all 22 bump Mach pairs into one `M0.00-0.00` directory.
- **No `oblique` drift.** It is the analytic θ-β-M wedge solution; there is no wedge.
  The default drift set drops to `null ffd SBsquared_exact+ffd`.
- **No geometry guard.** The α=5° assertion is diamond-specific.

The runs land under `outputs/Mach_interpolation/bump/…`, and since the case is part of
the analysis root path the two cases never pool:

```bash
uv run python SB/sweep_analysis.py --root outputs/Mach_interpolation/bump/iso/hessian
```

**Rounded-LE airfoils need a nose-refined tag** (`H_TAG=h0.025le7`). A rounded
leading edge carries a *detached* bow shock whose standoff is 0.01–0.04 chord; at
plain h0.025 it spans 1.3–5 cells, the captured shock snaps between cell rows as Mach
varies, and the reference C_D/C_L show a sawtooth (naca0012: +7–18% jumps at M1.69,
2.04, 2.81; C_L ≠ 0 at AoA 0). The diamond (sharp LE) and bump are unaffected. The
`le<N>` tag fixes it **without touching SB**:
- `generate_mesh.py --nose-factor N` writes `<case>_h0.025leN.npy` — the standard mesh,
  cell-for-cell, plus an `h_tag` — and `<case>_h0.025leN_solver.npy`, a *nested*
  longest-edge bisection of it refined to h/N within 0.06 chord of the LE (wall
  midpoints projected onto the true contour; child→parent map in its metadata).
- `Euler/main.py` sees `solver_mesh` in the metadata (`config.load_solver_mesh`),
  solves on the fine mesh, and exports the bundle on the standard mesh as the
  area-weighted average of the conserved variables over each cell's children.
  Untouched cells come back unchanged.
- SB therefore runs on the standard mesh at unchanged cost. This is the point: SB's
  explicit kernel sizes dt on the smallest cell, so running it on a nose-refined mesh
  would cost ~21× per run (le10). Euler pays ~7× per solve instead (le7).
- The tag keeps the corpora apart: bundles go to `results/<case>/h0.025leN/`
  (`format_h` returns `h_tag`), SB output to `…/eig/h0.025leN/`, and the analysis labels
  those runs `hmode = eig_leN`, so they never pool with plain `eig` at the same h.
  Analyse with `interval_analysis.py --hmode eig_le7`.
- Never regenerate a variant under a plain tag: the sbatch stale-check would quarantine
  every existing bundle of that tag (different `n_cells`) and re-solve it.

**Chunking.** Both scripts pack several runs per array task (`CHUNK`, default 8 for
Euler and 12 for SB). This is not only for startup amortisation: the cluster enforces
`QOSMaxSubmitJobPerUserLimit`, and a 576-task array trips it outright. That is a
submit-COUNT cap, unrelated to the 1–2 day walltime cap.

The Euler script **chunks** (`CHUNK=8`): a solve is ~25 s at h0.025 but `uv run` +
JAX import + XLA compile is ~40 s, so one task per solve would spend most of the
allocation on startup. A 0.10-wide pair needs its 9 interior references, hence the
0.01 Mach spacing — 221 Mach values per AoA.

Two **physics limits** are enforced rather than discovered at runtime:
- `oblique` has no attached-shock solution below the detachment Mach (~1.25 for the
  5° half-wedge — `beta_le` *raises* at M≤1.20). Those 5 pairs × 2 AoA × 3 γ = 30
  tasks skip in seconds, guarded by calling `beta_le` itself so the check follows
  the geometry.
- Below M=1 there is **no shock at all**, and from 1.0 to ~1.25 it is a detached bow
  shock. The Hessian marginal, the drift and the whole transport premise are built
  for attached oblique shocks, so the two lower bands are exploratory — the analysis
  figure `err_vs_mach_*.png` shades them red/orange against the validated green band.

**Parameter sweep**: `sbatch run_sb_sweep.sbatch` runs the full γ_min × drift × hessian_mode
study (12 γ from 0.5→1e-4 × {null, oblique, ffd} × {eig, det} = 72 combos) as a SLURM
**array job** (`--array=0-71%4`; the throttle self-adapts if the cluster allows fewer
concurrent jobs). Each run's schedule is the ladder **prefix** down to its γ_min (identical
warm-start path + same `num_iter` cap → fair A/B), W₂ on. Test one combo with
`sbatch --array=42 run_sb_sweep.sbatch`. Every Mach run writes a **`metrics.json`**
(config echo, per-stage IPFP stats + residual history, map diagnostics incl. trust-gate
revert counts, L2/L∞/W₂ error tables, timings). Aggregate with
`uv run python SB/sweep_analysis.py` (pure numpy/matplotlib, no GPU — run locally after
rsync-ing `outputs/`): writes error-vs-γ curves, γ-graded error-vs-t, IPFP convergence,
gain heatmap, map diagnostics, **cost-vs-γ**, **cost-vs-accuracy Pareto**, **SB-vs-FFD
head-to-head**, and CSV tables to `<root>/_sweep/`.

**Three interpolation methods** are compared in every run (see `compute_interpolation_error`,
which takes an ordered `{name: field_sequence}` dict):
- `BaryCDI` — the SB barycentric transport interpolation (**γ-dependent**).
- `Linear` — naive `(1−t)M₀ + tM₁` (γ-independent, zero transport).
- `FFD` — the FFD registration used **directly** as an interpolator: the same CDI formula
  but with the raw registration maps `T = x+β`, `S = x+α` instead of the SB maps
  (γ-independent). Only present when `reference_drift="ffd"`. This is the control that
  isolates what the Schrödinger bridge adds *on top of* the registration it is built from —
  if SB never beats FFD, the bridge is not paying for its IPFP cost.

`metrics.json` records the cost split per method: `timings.offline_s` (SB pays the IPFP
anneal, FFD pays the registration, Linear pays nothing) and `timings.recon_s` (the online
query cost). The Pareto plot uses offline+online, so accuracy claims are always priced.

## Architecture

### Euler Solver (`Euler/`)

```
Euler/
├── main.py          — Entry point: initialize + run loop
├── config.py        — CLI arg parsing, config loading from config.toml
├── utils.py         — Snapshot export, run summary export
├── graph.py         — Graph export (.npz) for ML downstream tasks
├── bump.py          — Bump mesh generator (meshpy.triangle)
├── diamond.py       — Diamond mesh generator
└── jax_fvm/src/
    ├── euler_solver.py  — FVM time-stepping (EE, RK2, RK4, SRK2)
    ├── mesh.py          — Mesh class (load/save .npy, BCs, geometry)
    ├── helper.py        — CFL, conserved↔primitive vars, C_D/C_L, entropy
    └── plot.py          — Field plotting on unstructured meshes
```

**Data flow**: `config.toml` → CLI overrides → `Mesh.load_mesh()` → `initialize()` (sets conserved variables `W`) → `run()` (JAX `lax.scan` time loop with stationarity check) → export to `results/` and `figures/`.

**Solver internals**: State vector `W` holds conserved variables `[ρ, ρu, ρv, E]` per cell. The time loop uses `jax.lax.scan` with an early-stopping condition based on a relative stationarity residual. JIT warm-up happens before the main loop to avoid measuring compilation time.

**Boundary conditions** are encoded via `face_markers` in the Mesh:
- `2` = wall (no-slip / no-penetration)
- `3` = inlet (prescribed values)
- `4` = outlet
- Farfield / sponge layers handled in specific cases

**Output**: `results/<case>/<h_tag>/<snapshot>.npz` (conserved vars, primitives, Mach, mesh metadata) and `figures/` (PNG field plots). The `bundle=true` export option saves everything needed for downstream SB interpolation.

### Schrödinger Bridge Module (`SB/`)

```
SB/
├── main.py          — Entry point: reads Config.toml (or $SB_CONFIG), dispatches to 1D / 2D / Mach
├── Config.toml      — Single source of truth for every run parameter (no CLI args)
├── utils.py         — Pipeline runners + physics helpers + barycentric transport maps
├── resolution.py    — IPFP (1D + 2D FVM heat variant + 2D drifted variant) & bridge retrieval
├── heat_solver.py   — FVM heat semigroup ∂φ/∂t = γΔφ (the β≡0 SB kernel)
├── advdiff_solver.py— FVM advection–diffusion semigroups Q / Q† (the drifted SB kernel)
├── drift.py         — Reference drift β: analytic "oblique" builder, flow_map, make_inside_body
├── ffd_drift.py     — Data-driven "ffd" drift: FFD registration of the marginals via phdtruel
│                     (also the γ-independent FFD control interpolator, see ffd_control)
│   note: the "SBsquared" bootstrapped drift lives in the annealing loop of utils.py,
│         not here — it has no builder, it recycles the bridge's own retrieve_b_2d*
├── plot.py          — SB plots (density transport, drift field/sequence, entropy, L2/L∞ error)
└── testcases.py     — 1D/2D synthetic test cases (Gaussian, OU, bimodal)
```

**SB pipeline** (`run_mach_interpolation_case` in `utils.py`):
1. Load two Euler snapshots (bundles `.npz`) at Mach M₀ and M₁
2. Convert to probability densities (the `density_mode` marginal — Ducros sensor mask, screened/TV inpainting, whole-disturbance field, shock iso-curve ridge, or **Alauzet–Loseille Hessian metric** density; see `transform_to_shock_density`)
3. Optionally build a **reference drift β** (`reference_drift` ∈ `null`/`oblique`/`ffd`) that conditions the bridge
4. Run **Anderson-accelerated IPFP** — `apply_IPFP_2d_anderson` (heat) or `apply_IPFP_2d_anderson_drift` (drifted Q/Q†) — with **γ-annealing** (a decreasing `gamma_sb_schedule`, each stage warm-started from the previous `f,g`) to find the potentials `f`, `g`. Requires `Float64`.
5. Compute global boundary-corrected **barycentric transport maps** `T`/`S` — `compute_sb_barycentric_maps` (heat) or `compute_sb_barycentric_maps_drift` (drifted) — and reconstruct intermediate Mach fields via gradient-free **barycentric SB-CDI** (`reconstruct_mach_barycentric_cdi`), vs. a linear baseline
6. If 9 reference bundles are configured, compute L2/L∞ (and optional W₂) error vs. high-resolution references and plot the error curves plus per-method abs-error grids
7. Plot density transport, the drift field (`plot_drift_field`), the **per-timestep drift grid** (`plot_drift_sequence`, drifted cases only), entropy, and interpolated Mach frames

**IPFP algorithm** (`resolution.py`): Iterative Proportional Fitting on log-space potentials, using an FVM semigroup as the Markov kernel (no explicit kernel matrix). β≡0 uses the self-adjoint heat semigroup `apply_logPt_fvm`; a reference drift uses the advection–diffusion semigroups `apply_logQt_fvm` (Q, non-conservative backward) and `apply_logQt_adjoint_fvm` (Q†, conservative forward). Anderson(m) acceleration is done on CPU (numpy least-squares) around the JIT'd kernel solves.

**Reference drift** (see `SB/referrence_drift.tex` eqs. for the math):
- **Barycentric maps** — for β≡0, `T = P₁[id·e^g]/P₁[e^g] − β_bias`, `S` analogously, sharing one Neumann identity-leak bias `β_bias = P₁[id]/P₁[1] − id`; T,S → identity far from the transported feature. Self-adjoint, so numerator/denominator blow-ups cancel in the ratio — no trust gate needed.
- **Drifted maps** (`compute_sb_barycentric_maps_drift`) — T from `g` via Q, S from `f` via Q†. Because Q ≠ Q† the cancellation breaks, so: (a) the leak bias is measured against the drift's own **reflected deterministic flow** Φ_β (`drift.flow_map`, integrated with the same no-flux boundaries + body-retraction the kernel uses via `drift.make_inside_body`), NOT the identity and NOT the unbounded analytic flow; (b) a **trust gate** reverts kernel-collapse outlier cells (displacement > max|Φ−x| + 4σ) to the identity.
- **CDI reconstruction** — query points are clamped to the mesh bbox; points inside the solid body (exact `make_inside_body` predicate) or outside the Delaunay hull fall back to the **undisplaced endpoint value** (local identity), not to the nearest fluid cell (which can be the wrong side of the body).

**Key design constraint**: The SB module uses `sys.path` manipulation to import from `Euler/` (and `phdtruel/` for the FFD drift) — it adds the repo root and `Euler/` to `sys.path` at startup. Do not reorganize imports without accounting for this.

### Mesh Format

Meshes are stored as `.npy` files (numpy structured arrays via `Mesh.save_mesh`/`load_mesh`). The `Mesh` object is a JAX pytree node (registered via `@jax.tree_util.register_pytree_node_class`). Key attributes: `points`, `tris`, `neighbors`, `faces`, `face_markers`, `barycenter`, `area`, `surface`, `normals`, `face_connectivity`, `midedge`.

### Output Structure

```
results/<case>/<h_tag>/      ← .npz snapshots (conserved vars + primitives + Mach)
figures/<case>/<h_tag>/      ← PNG field plots
meshes/bump/                 ← pre-generated bump meshes
meshes/diamond/              ← pre-generated diamond meshes

outputs/                              ← SB interpolation outputs (set via run.output_dir)
├── 1d_test_case/                     ← 1D synthetic cases
├── 2d_test_case/                     ← 2D synthetic cases
└── Mach_interpolation/<case>/        ← Mach field interpolation (<case> = diamond | bump)
    ├── ducros/                       ← density_mode = "mask"
    ├── denoising/{screened,tv}/      ← density_mode = "screened" | "tv"
    ├── field/<field_source>/         ← density_mode = "field" (pert_mach, pert_p, grad_mach, grad_p, combo)
    ├── iso/contours/                 ← density_mode = "iso"
    ├── iso/hessian/{det,eig}/        ← density_mode = "iso_hessian" (Alauzet–Loseille metric)
    └── isocurve/                     ← density_mode = "isocurve"
        └── …/drift_<mode>/gmin<γ>/   ← final split: drift_null|oblique|ffd × smallest annealed γ
```

**Interpolation-pair levels.** Once more than one Mach pair or AoA pair is studied,
the pair itself becomes a directory level — otherwise every pair writes to the same
path. Both levels are **empty for the original pairs**, so the 576 finished runs keep
their exact paths:

```
Mach 2.00→2.50 @ AoA 0  →  …/eig/h0.025/drift_ffd/gmin1e-04/            (legacy, unchanged)
Mach 2.00→2.50 @ AoA 2  →  …/eig/h0.025/aoa2.00/drift_ffd/…             (legacy, unchanged)
Mach 0.80→0.90 @ AoA 0  →  …/eig/h0.025/M0.80-0.90/drift_ffd/…          (new range level)
Mach 0.80→0.90 @ AoA 2  →  …/eig/h0.025/aoa2.00/M0.80-0.90/drift_ffd/…
AoA  0→4       @ M2.00  →  AoA_interpolation/…/h0.025/M2.00/drift_ffd/… (legacy, unchanged)
AoA  0→1       @ M2.00  →  AoA_interpolation/…/h0.025/M2.00/A0.00-1.00/…
```

`density_mode = "iso_hessian"` has two sub-modes (`hessian_mode`): `"det"` (node density
∝ (det|H|)^{p/(2p+2)}) and `"eig"` (|λ_max| feature strength). Use `"eig"` for the
reference-drift A/B — `"det"` collapses on the straight diamond legs, leaving no mass on
the shocks the drift is meant to move. Every Mach run's final directory is
`…/drift_<reference_drift>/gmin<min(gamma_sb_schedule)>/`.

## JAX-Specific Notes

- `jax.lax.scan` is used for the time loop to enable JIT compilation of the full simulation — avoid Python-level loops in hot paths
- `jax.lax.cond` replaces Python `if` inside scanned functions
- `jax_enable_x64=True` is mandatory for the SB module (IPFP needs Float64 precision)
- On SLURM: set `XLA_PYTHON_CLIENT_PREALLOCATE=false` to avoid GPU memory issues; use `JAX_COMPILATION_CACHE_DIR` for repeated runs
- The transonic regime (0.6 < M < 1.1) automatically reduces CFL to 0.4
- **γ-annealing (ε-scaling)**: the IPFP reaches small γ (near the OT limit) by warm-starting each `gamma_sb_schedule` stage from the previous stage's `(f,g)`. Cold-starting at small γ diverges. The drifted kernel step count comes from `advdiff_solver.compute_n_steps_advdiff` (advective + diffusive CFL); the heat kernel from `heat_solver.compute_n_steps`.
- **Anderson IPFP** (`apply_IPFP_2d_anderson[_drift]`) runs the acceleration on CPU/numpy around JIT'd GPU kernel solves; it prints per-`print_every` residuals and stops at `tol` or `num_iter`.
