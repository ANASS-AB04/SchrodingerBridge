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
module load nvidia/cuda/12.4
export PATH="$HOME/.local/bin:$PATH"

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

# All run parameters (case, reference_drift, gamma schedule, density_mode …)
# are read from SB/Config.toml — edit that file to change the run.
uv run python SB/main.py
