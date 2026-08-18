#!/usr/bin/env bash
#SBATCH --job-name=sb-sweep-full
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-107%4
# logs/ root (SLURM does NOT create missing --output dirs)
#SBATCH --output=logs/sb-sweep-full_%A_%a.out
#SBATCH --error=logs/sb-sweep-full_%A_%a.err

# ─────────────────────────────────────────────────────────────────────────────
# The FULL h0.025 SB sweep in ONE job (replaces run_sb_sweep + run_sb_sweep_low):
#
#   18 γ_min (0.5 → 1e-6)  ×  3 drifts (null, oblique, ffd)  ×  2 hessian modes
#   = 108 combos, one array task each  (--array=0-107%4).
#
# Reads SB/Config.toml (the default h0.025 case); each task overrides only
# reference_drift, the γ-schedule (the full ladder PREFIX down to its γ_min,
# warm-started → fair A/B), hessian_mode, and forces compute_w2=true.  The FFD
# interpolation baseline (errors["FFD"]) is computed NATIVELY per run, so no
# backfill is needed afterwards.
#
# Outputs → outputs/…/iso/hessian/<hmode>/drift_<mode>/gmin<γ>/  (+ metrics.json).
# Aggregate locally:  uv run python SB/sweep_analysis.py
#
# Submit:            sbatch run_sb_sweep_full.sbatch
# Single combo test: sbatch --array=30 run_sb_sweep_full.sbatch
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs

module purge
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH=""

export MPLBACKEND=Agg
export JAX_ENABLE_X64=true
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

JAX_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/jax-cache-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
rm -rf "$JAX_CACHE_ROOT"; mkdir -p "$JAX_CACHE_ROOT"
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_ROOT"

# ── Grid: full 18-γ ladder × 3 drifts × 2 hessian modes = 108 ────────────────
GAMMAS=(0.5 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.0005 0.0002 0.0001 \
        0.00005 0.00002 0.00001 0.000005 0.000002 0.000001)
DRIFTS=(null oblique ffd)
HMODES=(eig det)
NG=${#GAMMAS[@]}                                   # 18
ND=${#DRIFTS[@]}                                   # 3

ID=${SLURM_ARRAY_TASK_ID}
HIDX=$(( ID / (NG * ND) ))                         # 0..1
REM=$((  ID % (NG * ND) ))
DIDX=$(( REM / NG ))                               # 0..2
GIDX=$(( REM % NG ))                               # 0..17
HMODE=${HMODES[$HIDX]}
DRIFT=${DRIFTS[$DIDX]}
GMIN=${GAMMAS[$GIDX]}

SCHED="["                                          # full ladder prefix GAMMAS[0..GIDX]
for ((i = 0; i <= GIDX; i++)); do
    SCHED+="${GAMMAS[$i]}"; (( i < GIDX )) && SCHED+=", "
done
SCHED+="]"

BASE_CFG="SB/Config.toml"
GEN_DIR="${SLURM_TMPDIR:-/tmp}/sb_sweep_full_cfgs"
mkdir -p "$GEN_DIR"
CFG="$GEN_DIR/Config_${HMODE}_${DRIFT}_gmin${GMIN}.toml"
sed -e "s|^reference_drift = .*|reference_drift = \"${DRIFT}\"|" \
    -e "s|^gamma_sb_schedule = .*|gamma_sb_schedule = ${SCHED}|" \
    -e "s|^gamma_sb *= .*|gamma_sb = ${GMIN}|" \
    -e "s|^density_mode *= .*|density_mode = \"iso_hessian\"|" \
    -e "s|^hessian_mode *= .*|hessian_mode = \"${HMODE}\"|" \
    -e "s|^compute_w2 *= .*|compute_w2 = true|" \
    "$BASE_CFG" > "$CFG"

echo "#############################"
echo "Array job/task:  ${SLURM_ARRAY_JOB_ID} / ${SLURM_ARRAY_TASK_ID}"
echo "Combo:           hessian_mode=${HMODE}  drift=${DRIFT}  gamma_min=${GMIN}"
echo "Schedule:        ${SCHED}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "#############################"

SB_CONFIG="$CFG" uv run python SB/main.py

echo "######## task ${ID} (${HMODE}/${DRIFT}/gmin${GMIN}) done ########"
