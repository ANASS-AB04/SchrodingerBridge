#!/usr/bin/env bash
#SBATCH --job-name=euler-ref
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-10%4
# logs/ root (SLURM does NOT create missing --output dirs)
#SBATCH --output=logs/euler-ref_%A_%a.out
#SBATCH --error=logs/euler-ref_%A_%a.err

# ─────────────────────────────────────────────────────────────────────────────
# BATCH 1 of 2 — Euler solutions on a REFINED mesh.
#
# Produces the 11 diamond snapshots (M2.00 … M2.50 step 0.05) that the SB
# interpolation needs: the two endpoints (M2.00, M2.50) + the 9 references
# (M2.05 … M2.45).  One array task per Mach so they run in parallel.
#
# Results land in results/diamond/<H_TAG>/AOA0.00_M<mach>_HLLC_MUSCL_SRK2_t<T>.npz
# where <T> is whatever time the stationarity check stopped at — which is why
# SB/make_config_for_mesh.py GLOBS these rather than hard-coding names.
#
# PREREQUISITE: the mesh must exist AND be the 5° wedge.  The fine meshes
# shipped before 2026-08 were α=10° (height=0.176) while h0.025 is α=5°
# (height=0.0875), so a refinement study across them compared two different
# airfoils.  They are archived in meshes/diamond/alpha10/; the 5° ladder is
# rebuilt with:
#   uv run python Euler/diamond.py --h 0.0175 0.0125
#
# After this finishes, build the SB config and run BATCH 2:
#   uv run python SB/make_config_for_mesh.py --h-tag h0.0125
#   sbatch --export=ALL,H_TAG=h0.0125 run_sb_sweep_refined.sbatch
#
# NOTE ON COST (5° meshes — the near-body refinement zone scales with the wedge
# height, so these are far smaller than the old 10° ones):
#   h0.025   20190 cells   dx₁₀=3.35e-3    ×1
#   h0.0175  41106 cells   dx₁₀=2.34e-3    ≈×3   (2.0× cells, ~1.4× steps)
#   h0.0125  80889 cells   dx₁₀=1.67e-3    ≈×8   (4.0× cells, ~2.0× steps)
# --time=12:00:00 is generous for both; run one Mach first to calibrate.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs

module purge
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH=""

export MPLBACKEND=Agg
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

JAX_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/jax-cache-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
rm -rf "$JAX_CACHE_ROOT"; mkdir -p "$JAX_CACHE_ROOT"
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_ROOT"

# ── Grid ─────────────────────────────────────────────────────────────────────
H_TAG=${H_TAG:-h0.0125}                 # override: sbatch --export=ALL,H_TAG=h0.0175 …
MACHS=(2.00 2.05 2.10 2.15 2.20 2.25 2.30 2.35 2.40 2.45 2.50)
MACH=${MACHS[$SLURM_ARRAY_TASK_ID]}
MESH="meshes/diamond/diamond_${H_TAG}.npy"

echo "#############################"
echo "Array job/task: ${SLURM_ARRAY_JOB_ID} / ${SLURM_ARRAY_TASK_ID}"
echo "Mach:           ${MACH}"
echo "Mesh:           ${MESH}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "#############################"

uv run python Euler/main.py \
    --case diamond \
    --mach "${MACH}" \
    --aoa 0 \
    --mesh-path "${MESH}" \
    --flux HLLC \
    --time-scheme SRK2 \
    --reconstruction MUSCL

echo "######## euler task ${SLURM_ARRAY_TASK_ID} (M${MACH}, ${H_TAG}) done ########"
