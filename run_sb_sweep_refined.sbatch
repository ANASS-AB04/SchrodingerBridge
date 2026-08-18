#!/usr/bin/env bash
#SBATCH --job-name=sb-sweep-ref
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=5,8,11
# logs/ root (SLURM does NOT create missing --output dirs)
#SBATCH --output=logs/sb-sweep-ref_%A_%a.out
#SBATCH --error=logs/sb-sweep-ref_%A_%a.err

# ─────────────────────────────────────────────────────────────────────────────
# BATCH 2 of 2 — SB sweep on a REFINED mesh (mesh-convergence study).
#
# PURPOSE: does the FFD-guided drift still work as cells shrink, and does
# refinement close the gap between the SB bridge (BaryCDI) and the raw FFD
# registration it is built from?  At h0.025 the FFD control BEATS BaryCDI at
# every γ (0.0023 vs 0.0032 mean L2), and BaryCDI plateaus below γ=1e-4.  Since
# the FFD drift lives on a fixed 144² Bernstein grid that does NOT depend on h,
# refining sharpens the marginals but cannot sharpen β — so the prediction is
# that the gap does NOT close, and the lever is grid_n/n_cp instead.
#
# ── FOCUSED SUBSET (this is the default; ~9 runs total across the ladder) ─────
# HMODES=(eig) and DRIFTS=(ffd) below reduce the grid to the γ axis alone, so
# the array index IS the γ index.  The drift_ffd runs already compute the
# `Linear` and `FFD` controls internally, so the three-method head-to-head
# survives dropping null/oblique.
#
#     --array=5,8,11   →  γ_min = 0.01, 0.001, 0.0001
#
# Each run's schedule is the ladder PREFIX down to its γ_min, so every task
# shares the same warm-start path (fair A/B against the h0.025 sweep).
#
# To recover the FULL 108-combo grid:  DRIFTS="null oblique ffd" HMODES="eig det"
# sbatch --array=0-107%4 run_sb_sweep_refined.sbatch
#
# PREREQUISITES (in order):
#   uv run python Euler/diamond.py --h 0.0175 0.0125    # 5° meshes (done)
#   sbatch --export=ALL,H_TAG=h0.0125 run_euler_refined.sbatch   # BATCH 1
#   uv run python SB/make_config_for_mesh.py --h-tag h0.0125
#   sbatch --export=ALL,H_TAG=h0.0125 run_sb_sweep_refined.sbatch
#
# COST: the 5° refined meshes are much smaller than the old 10° ones (the
# near-body refinement zone scales with the wedge height, which halved):
#   h0.025  20190 cells  dx₁₀=3.35e-3   ×1
#   h0.0175 41106 cells  dx₁₀=2.34e-3   ≈×4    (2.0× cells, 2.0× n_steps)
#   h0.0125 80889 cells  dx₁₀=1.67e-3   ≈×16   (4.0× cells, 4.0× n_steps)
# The whole 108-run h0.025 sweep cost ~3 GPU-h of IPFP, so even the full grid at
# h0.0125 is ~50 GPU-h; the focused 3-run subset is minutes-to-hours.
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

# ── Grid ─────────────────────────────────────────────────────────────────────
H_TAG=${H_TAG:-h0.0125}
GEN_DIR="${SLURM_TMPDIR:-/tmp}/sb_sweep_ref_cfgs"
mkdir -p "$GEN_DIR"

# Base config: derived from the Euler bundles that BATCH 1 just produced.  Each
# array task writes its OWN copy (tasks run concurrently and would otherwise race
# on one file), so this script can be chained straight off the Euler job with
# --dependency and needs no manual make_config_for_mesh.py step.
# Set BASE_CFG explicitly to reuse a hand-edited config instead.
if [[ -n "${BASE_CFG:-}" ]]; then
    if [[ ! -f "$BASE_CFG" ]]; then
        echo "ERROR: BASE_CFG=$BASE_CFG not found."; exit 1
    fi
else
    BASE_CFG="$GEN_DIR/Config_${H_TAG}_base_${SLURM_ARRAY_TASK_ID}.toml"
    uv run python SB/make_config_for_mesh.py --h-tag "${H_TAG}" --out "$BASE_CFG"
fi

GAMMAS=(0.5 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.0005 0.0002 0.0001 \
        0.00005 0.00002 0.00001 0.000005 0.000002 0.000001)
#            idx: 0   1   2    3    4    5     6     7     8      9      10     11
# Focused subset by default; override from the command line to widen the grid,
# e.g.  DRIFTS="null oblique ffd" HMODES="eig det" sbatch --array=0-107%4 ...
read -r -a DRIFTS <<< "${DRIFTS:-ffd}"
read -r -a HMODES <<< "${HMODES:-eig}"
NG=${#GAMMAS[@]}                                   # 18
ND=${#DRIFTS[@]}

ID=${SLURM_ARRAY_TASK_ID}
HIDX=$(( ID / (NG * ND) ))
REM=$((  ID % (NG * ND) ))
DIDX=$(( REM / NG ))
GIDX=$(( REM % NG ))                               # 0..17
HMODE=${HMODES[$HIDX]}
DRIFT=${DRIFTS[$DIDX]}
GMIN=${GAMMAS[$GIDX]}

SCHED="["                                          # full ladder prefix
for ((i = 0; i <= GIDX; i++)); do
    SCHED+="${GAMMAS[$i]}"; (( i < GIDX )) && SCHED+=", "
done
SCHED+="]"

CFG="$GEN_DIR/Config_${H_TAG}_${HMODE}_${DRIFT}_gmin${GMIN}.toml"
sed -e "s|^reference_drift = .*|reference_drift = \"${DRIFT}\"|" \
    -e "s|^gamma_sb_schedule = .*|gamma_sb_schedule = ${SCHED}|" \
    -e "s|^gamma_sb *= .*|gamma_sb = ${GMIN}|" \
    -e "s|^density_mode *= .*|density_mode = \"iso_hessian\"|" \
    -e "s|^hessian_mode *= .*|hessian_mode = \"${HMODE}\"|" \
    -e "s|^compute_w2 *= .*|compute_w2 = true|" \
    "$BASE_CFG" > "$CFG"

echo "#############################"
echo "Array job/task:  ${SLURM_ARRAY_JOB_ID} / ${SLURM_ARRAY_TASK_ID}"
echo "Mesh:            ${H_TAG}   (base config ${BASE_CFG})"
echo "Combo:           hessian_mode=${HMODE}  drift=${DRIFT}  gamma_min=${GMIN}"
echo "Schedule:        ${SCHED}"
echo "#############################"

SB_CONFIG="$CFG" uv run python SB/main.py

echo "######## task ${ID} (${H_TAG}/${HMODE}/${DRIFT}/gmin${GMIN}) done ########"
