#!/usr/bin/env bash
#SBATCH --job-name=sb-sweep
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-71%4
# NB: SLURM opens these BEFORE the script runs and does NOT create missing dirs,
# so write into the logs/ root (guaranteed to exist) — a logs/%x/ subdir would
# need to pre-exist or every task fails instantly at I/O setup.
#SBATCH --output=logs/sb-sweep_%A_%a.out
#SBATCH --error=logs/sb-sweep_%A_%a.err

# ─────────────────────────────────────────────────────────────────────────────
# γ_min × drift × hessian_mode sweep for the SB Mach interpolation.
#
#   12 γ_min  ×  3 drifts (null, oblique, ffd)  ×  2 hessian modes (eig, det)
#   = 72 combos, one SLURM array task each  (--array=0-71%4: ≤4 concurrent;
#   if the cluster only allows 1 running job the array simply runs serially).
#
# Every run's annealing schedule is the LADDER PREFIX down to its γ_min, so all
# runs share the same warm-start path and the same num_iter cap → fair A/B.
# Outputs land in outputs/…/iso/hessian/<hmode>/drift_<mode>/gmin<γ>/ (one dir
# per combo, never clobbered) + metrics.json per run for SB/sweep_analysis.py.
#
# Submit:            sbatch run_sb_sweep.sbatch
# Single combo test: sbatch --array=42 run_sb_sweep.sbatch
# After completion:  uv run python SB/sweep_analysis.py   (locally, after rsync)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs                      # log files go to logs/ root (see --output above)

# CALI3: no system CUDA module — jax[cuda12] wheels bundle their own CUDA+cuDNN
# (loading nvidia/cuda/12.4 shadows them → cudnnGetLibConfig error).
module purge
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH=""

export MPLBACKEND=Agg
export JAX_ENABLE_X64=true
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Per-task JAX compilation cache (tasks may share a node — keep caches apart)
JAX_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/jax-cache-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
rm -rf "$JAX_CACHE_ROOT"
mkdir -p "$JAX_CACHE_ROOT"
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_ROOT"

# ── Sweep grid ───────────────────────────────────────────────────────────────
GAMMAS=(0.5 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.0005 0.0002 0.0001)
DRIFTS=(null oblique ffd)
HMODES=(eig det)
NG=${#GAMMAS[@]}                                   # 12
ND=${#DRIFTS[@]}                                   # 3

ID=${SLURM_ARRAY_TASK_ID}
HIDX=$(( ID / (NG * ND) ))                         # 0..1
REM=$((  ID % (NG * ND) ))
DIDX=$(( REM / NG ))                               # 0..2
GIDX=$(( REM % NG ))                               # 0..11
HMODE=${HMODES[$HIDX]}
DRIFT=${DRIFTS[$DIDX]}
GMIN=${GAMMAS[$GIDX]}

# Annealing schedule = ladder prefix [GAMMAS[0] … GAMMAS[GIDX]]
SCHED="["
for ((i = 0; i <= GIDX; i++)); do
    SCHED+="${GAMMAS[$i]}"
    (( i < GIDX )) && SCHED+=", "
done
SCHED+="]"

# ── Derive the per-combo config from the shared Config.toml ──────────────────
BASE_CFG="SB/Config.toml"
GEN_DIR="${SLURM_TMPDIR:-/tmp}/sb_sweep_cfgs"
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
echo "User:            $USER"
echo "Array job/task:  ${SLURM_ARRAY_JOB_ID} / ${SLURM_ARRAY_TASK_ID}"
echo "Node:            ${SLURM_JOB_NODELIST}"
echo "Combo:           hessian_mode=${HMODE}  drift=${DRIFT}  gamma_min=${GMIN}"
echo "Schedule:        ${SCHED}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "#############################"

SB_CONFIG="$CFG" uv run python SB/main.py

echo "######## task ${ID} (${HMODE}/${DRIFT}/gmin${GMIN}) done ########"
