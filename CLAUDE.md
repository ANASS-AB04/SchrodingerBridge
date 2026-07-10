# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GenerativePDE is a research project combining:
1. **Euler/** — a JAX-based 2D compressible Euler equation solver using the Finite Volume Method (FVM)
2. **SB/** — a Schrödinger Bridge (SB) interpolation module that transports probability densities between two physical flow states (e.g., Mach fields) using the IPFP algorithm

The overall goal is generative PDE solving: use the Euler solver to generate CFD snapshots at various Mach numbers/AoA, then use SB to interpolate between them without re-running the PDE solver.

## Environment & Dependencies

- Python 3.11 (strict), managed with `uv`
- JAX with CUDA (GPU required for production runs)
- Key packages: `jax`, `numpy`, `scipy`, `matplotlib`, `meshpy`, `meshio`, `scikit-learn`

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
uv run python Euler/bump.py      # generates meshes/bump/bump_h*.npy
uv run python Euler/diamond.py   # generates meshes/diamond/diamond_h*.npy
```

### Schrödinger Bridge Interpolation
```bash
JAX_ENABLE_X64=true uv run python SB/main.py
```
`SB/main.py` takes **no command-line arguments** — every run parameter (which case to run,
1D/2D/Mach settings, density mode, etc.) is read from `SB/Config.toml`. Edit `Config.toml`'s
`run.case` to select a pipeline (1D synthetic, 2D synthetic, or `2d_mach_interpolation`) and
`run.mach.*` to configure the Mach interpolation (density mode, field source, bundle/mesh
paths per `run.mach.case`).

**Important**: `jax_enable_x64=True` must be set before any JAX arrays are created (done at module import in `SB/main.py`). Omitting this causes IPFP probability truncation bugs.

### Linting & Type Checking
```bash
uv run ruff check .
uv run mypy Euler/ SB/
```

### HPC (SLURM)
```bash
sbatch bump_test.sbatch           # bump case
sbatch run_diamond_test.sbatch    # diamond case
```

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
├── main.py        — Entry point: reads Config.toml, dispatches to the 1D / 2D / Mach pipeline
├── Config.toml    — Single source of truth for every run parameter (no CLI args)
├── utils.py       — Pipeline runners (run_1d_case, run_2d_case, run_mach_interpolation_case) + physics helpers
├── resolution.py  — IPFP algorithm (1D and 2D FVM variants)
├── heat_solver.py — FVM heat equation solver (used as the SB semigroup)
├── plot.py        — SB-specific plots (density transport, drift field, entropy, L2/L∞ error)
└── testcases.py   — 1D/2D synthetic test cases (Gaussian, OU, bimodal)
```

**SB pipeline** (`run_mach_interpolation_case` in `utils.py`):
1. Load two Euler snapshots (bundles `.npz`) at Mach M₀ and M₁
2. Convert to probability densities (the `density_mode` marginal — Ducros sensor mask, screened/TV inpainting, whole-disturbance field, or shock iso-curve ridge; see `transform_to_shock_density`)
3. Run **Anderson-accelerated IPFP** (`apply_IPFP_2d_anderson`) to find the Schrödinger Bridge potentials `f`, `g` — requires `Float64` for convergence
4. Compute global boundary-corrected **barycentric transport maps** `T`/`S` (`compute_sb_barycentric_maps`) and reconstruct intermediate Mach fields via gradient-free **barycentric SB-CDI** (`reconstruct_mach_barycentric_cdi`), compared against a linear baseline
5. If 9 reference bundles are configured, compute L2/L∞ error vs. high-resolution references and always plot both error curves plus per-method abs-error grids
6. Plot density transport, drift fields, and interpolated Mach frames

**IPFP algorithm** (`resolution.py`): Iterative Proportional Fitting Procedure on log-space potentials. The 2D version uses the FVM heat semigroup (`apply_logPt_fvm`) as the Markov kernel instead of an explicit kernel matrix. The `@partial(jax.jit, static_argnames=['num_iter'])` decoration is critical for performance.

**Key design constraint**: The SB module uses `sys.path` manipulation to import from `Euler/` — it adds both the repo root and `Euler/` to `sys.path` at startup. Do not reorganize imports without accounting for this.

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
└── Mach_interpolation/               ← Mach field interpolation, split by density_mode
    ├── ducros/                       ← density_mode = "mask"
    ├── denoising/{screened,tv}/      ← density_mode = "screened" | "tv"
    ├── iso/<field_source>/           ← density_mode = "field" (pert_mach, pert_p, grad_mach, grad_p, combo)
    └── isocurve/                     ← density_mode = "isocurve"
```

## JAX-Specific Notes

- `jax.lax.scan` is used for the time loop to enable JIT compilation of the full simulation — avoid Python-level loops in hot paths
- `jax.lax.cond` replaces Python `if` inside scanned functions
- `jax_enable_x64=True` is mandatory for the SB module (IPFP needs Float64 precision)
- On SLURM: set `XLA_PYTHON_CLIENT_PREALLOCATE=false` to avoid GPU memory issues; use `JAX_COMPILATION_CACHE_DIR` for repeated runs
- The transonic regime (0.6 < M < 1.1) automatically reduces CFL to 0.4
