#!/usr/bin/env bash
#SBATCH --job-name=sb-sweep-low
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-35%4
# logs/ root (SLURM does NOT create missing --output dirs — see run_sb_sweep.sbatch)
#SBATCH --output=logs/sb-sweep-low_%A_%a.out
#SBATCH --error=logs/sb-sweep-low_%A_%a.err

# ─────────────────────────────────────────────────────────────────────────────
# LOW-γ EXTENSION of run_sb_sweep.sbatch: adds 6 smaller γ_min (5e-5 → 1e-6)
# for the 3 drifts × 2 hessian modes = 36 combos, to test whether the drifted SB
# (oblique / ffd) pulls ahead of null + linear deep in the OT limit.
#
# Same fair-A/B contract: the annealing schedule is the FULL ladder prefix from
# 0.5 down to each γ_min (warm-started).  For the drifted kernel n_steps is set by
# the advective CFL, so it stays resolved as γ→0 while the pure-heat (null) kernel
# degenerates (n_steps ∝ γ → 1) — exactly the divergence we want to measure.
#
# Outputs go to the SAME tree (…/drift_<mode>/gmin<γ>/); new γ = new dirs, no
# clobber.  SB/sweep_analysis.py globs all gmin* so the curves auto-extend.
#
# Submit: sbatch run_sb_sweep_lowgamma.sbatch     Single: sbatch --array=12 run_sb_sweep_lowgamma.sbatch
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

# ── Grid: FULL ladder (for the prefix) but only iterate the NEW low-γ tail ────
GAMMAS=(0.5 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.0005 0.0002 0.0001 \
        0.00005 0.00002 0.00001 0.000005 0.000002 0.000001)
DRIFTS=(null oblique ffd)          # drop "null" here if you want to skip its (expected) collapse
HMODES=(eig det)
NEW_OFFSET=12                       # index of the first NEW γ (0.00005) in GAMMAS
NG_NEW=6                            # how many new γ (indices 12..17)
ND=${#DRIFTS[@]}                    # 3

ID=${SLURM_ARRAY_TASK_ID}
HIDX=$(( ID / (NG_NEW * ND) ))                     # 0..1
REM=$((  ID % (NG_NEW * ND) ))
DIDX=$(( REM / NG_NEW ))                           # 0..2
GNEW=$(( REM % NG_NEW ))                           # 0..5
GIDX=$(( NEW_OFFSET + GNEW ))                      # 12..17
HMODE=${HMODES[$HIDX]}
DRIFT=${DRIFTS[$DIDX]}
GMIN=${GAMMAS[$GIDX]}

SCHED="["                                          # full prefix GAMMAS[0..GIDX]
for ((i = 0; i <= GIDX; i++)); do
    SCHED+="${GAMMAS[$i]}"; (( i < GIDX )) && SCHED+=", "
done
SCHED+="]"

BASE_CFG="SB/Config.toml"
GEN_DIR="${SLURM_TMPDIR:-/tmp}/sb_sweep_low_cfgs"
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
echo "#############################"

SB_CONFIG="$CFG" uv run python SB/main.py

echo "######## task ${ID} (${HMODE}/${DRIFT}/gmin${GMIN}) done ########"
