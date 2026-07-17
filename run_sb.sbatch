#!/usr/bin/env bash
#SBATCH --job-name=sb-mach
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err

set -euo pipefail

mkdir -p "logs/${SLURM_JOB_NAME}"

# CALI3 environment: no `cuda`/`uv` modules exist here.  Real CUDA module is
# nvidia/cuda/12.4; uv is installed in ~/.local/bin (curl installer, no root).
module purge
# module load nvidia/cuda/12.4               # removed: shadows jax's cuDNN
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH=""                     # ← add this line


export MPLBACKEND=Agg
export JAX_ENABLE_X64=true
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

JAX_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/jax-cache"
rm -rf "$JAX_CACHE_ROOT"
mkdir -p "$JAX_CACHE_ROOT"
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_ROOT"

echo "#############################"
echo "User:" $USER
echo "SLURM_JOBID:" $SLURM_JOBID
echo "SLURM_SUBMIT_DIR:" $SLURM_SUBMIT_DIR
echo "SLURM_JOB_NODELIST:" $SLURM_JOB_NODELIST
nvidia-smi
echo "#############################"

# ── Run oblique, ffd and null in one job ─────────────────────────────────────
# All shared parameters come from SB/Config.toml (single source of truth); per
# case we override only `reference_drift` and `gamma_sb_schedule`.  oblique/ffd
# anneal to γ_min=1e-4 (the reference drift conditions the bridge that low); null
# stays at γ_min=5e-3 (the pure-heat bridge diverges below that near the OT limit).
# main.py routes each case to its own output dir (…/drift_<mode>/gmin<γ>/), so the
# three runs never clobber each other.
BASE_CFG="SB/Config.toml"
GEN_DIR="${SLURM_TMPDIR:-/tmp}/sb_configs"
mkdir -p "$GEN_DIR"

run_case () {                    # $1 = reference_drift   $2 = gamma_sb_schedule
    local drift="$1" sched="$2"
    local cfg="$GEN_DIR/Config_${drift}.toml"
    sed -e "s|^reference_drift = .*|reference_drift = \"${drift}\"|" \
        -e "s|^gamma_sb_schedule = .*|gamma_sb_schedule = ${sched}|" \
        "$BASE_CFG" > "$cfg"
    echo "################ reference_drift=${drift}  schedule=${sched} ################"
    SB_CONFIG="$cfg" uv run python SB/main.py \
        || echo "[WARN] case reference_drift=${drift} failed — continuing"
}

run_case oblique "[0.0003, 0.0002, 0.0001]"
run_case ffd     "[0.0003, 0.0002, 0.0001]"
run_case null    "[0.02, 0.01, 0.005]"

echo "################ all cases done ################"
