#!/usr/bin/env bash
#SBATCH --job-name=sb-aoa
#SBATCH --partition=gpu-h100,gpu-l40,gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-143%12
#SBATCH --output=logs/sb-aoa_%A_%a.out
#SBATCH --error=logs/sb-aoa_%A_%a.err

# ─────────────────────────────────────────────────────────────────────────────
# SB matrix for the angle-of-attack study.
#
#   sbatch run_sb_aoa_sweep.sbatch                    # h0.025, all 144
#   sbatch --array=0 run_sb_aoa_sweep.sbatch          # smoke: one run
#   H_TAG=h0.0175 sbatch run_sb_aoa_sweep.sbatch      # extend to a finer mesh
#
# PREREQUISITE: run_euler_grid_aoa.sbatch must have finished for this H_TAG.
#
#   cases  = aoa0.00 | aoa2.00 | aoa4.00 | axisaoa   (4)   << the interpolation
#   drifts = null | oblique | ffd                    (3)
#   gammas = ladder prefixes, 12 rungs 0.5 → 1e-4   (12)
#                                            144 tasks
#
#   task = ((case * N_DRIFT) + drift) * N_GAMMA + gamma
#
# WHY 0/2/4 deg AND an angle sweep.  Everything before this interpolated Mach at
# ZERO incidence, where the diamond is symmetric — so the reference C_L is pure
# mesh-asymmetry noise (~1e-4, and it flips sign between meshes) and lift cannot
# discriminate at all.  At 2 deg C_L is ~0.08 and at 4 deg ~0.16, roughly 1000x
# that floor.  The angle sweep is a different transport problem again: the shocks
# ROTATE about the leading edge instead of sweeping downstream, and the two
# surfaces move in opposite directions.
#
# ONE GPU PER TASK.  A pooled single-node job caps at the 2 GPUs an h100 node has;
# as an array SLURM spreads tasks over every free L40 and H100.  Each task also
# owns its output directory, so unlike submitting the pooled script several times
# there is no concurrent-write race on metrics.json.
#
# COST at h0.025: ~5-12 min per task (IPFP mean 148 s, max 333 s; W2 roughly
# doubles it; +24 s FFD registration) => ~19 GPU-h, ~2 h wall at 12 concurrent.
# Finer meshes are x4 (h0.0175) and x16 (h0.0125) — extend only once the coarse
# matrix says which corner earns it.
# ─────────────────────────────────────────────────────────────────────────────

set -uo pipefail
# bash printf honours LC_NUMERIC: under fr_FR "%.2f" of 2 gives "2,00", which
# would corrupt every path and glob built from it.
export LC_ALL=C
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH=""
export MPLBACKEND=Agg
export JAX_ENABLE_X64=true
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

JAX_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/jax-cache-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
rm -rf "$JAX_CACHE_ROOT"; mkdir -p "$JAX_CACHE_ROOT"
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_ROOT"

H_TAG=${H_TAG:-h0.025}
HMODE=${HMODE:-eig}
FIXED_MACH=$(LC_ALL=C printf "%.2f" "${FIXED_MACH:-2.00}")

read -r -a CASES  <<< "${CASES:-aoa0.00 aoa2.00 aoa4.00 axisaoa}"
# DRIFTS entries may be composite: <reference_drift>+<seed>, e.g.
#   SBsquared_exact+ffd  = exact time-dependent bootstrap SEEDED by the FFD
#   registration, held fixed until gamma < BOOT_SWITCH, then self-refining.
# Plain names behave exactly as before.
read -r -a DRIFTS <<< "${DRIFTS:-null oblique ffd}"
BOOT_SWITCH=${BOOT_SWITCH:-0.001}
# β(t) endpoint clip for the exact mode — t=1 is singular (Q_0 = identity, so the
# raw log-potential's gradient enters β and sets the CFL/Péclet for every stage).
BOOT_TCLIP=${BOOT_TCLIP:-0.2}
LADDER=(0.5 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.0005 0.0002 0.0001)
read -r -a GIDXS <<< "${GIDXS:-0 1 2 3 4 5 6 7 8 9 10 11}"

NC=${#CASES[@]};  ND=${#DRIFTS[@]};  NG=${#GIDXS[@]}
N_TASKS=$(( NC * ND * NG ))
ID=${SLURM_ARRAY_TASK_ID:-0}
if [[ "$ID" -ge "$N_TASKS" ]]; then
    echo "task $ID >= $N_TASKS — nothing to do"; exit 0
fi
CIDX=$(( ID / (ND * NG) ))
REM=$((  ID % (ND * NG) ))
DIDX=$(( REM / NG ))
GPOS=$(( REM % NG ))
CASE=${CASES[$CIDX]};  DRIFT=${DRIFTS[$DIDX]};  GIDX=${GIDXS[$GPOS]}
GMIN=${LADDER[$GIDX]}

# Split a composite drift into (reference_drift, seed).  Seeded variants hold the
# seed fixed until BOOT_SWITCH then bootstrap; the density mask is switched OFF
# for them (the seed is already void-free, masking a registration-refined drift
# would only clip real transport).  Heat-seeded variants keep Config defaults so
# they stay comparable with the existing SBsquared ladder.
if [[ "$DRIFT" == *"+"* ]]; then
    REF_DRIFT=${DRIFT%%+*};  SEED=${DRIFT#*+}
    SEED_SED=(-e "s|^drift_bootstrap_seed *= .*|drift_bootstrap_seed        = \"${SEED}\"|" \
              -e "s|^drift_bootstrap_start_gamma *= .*|drift_bootstrap_start_gamma = ${BOOT_SWITCH}|" \
              -e "s|^drift_bootstrap_mask *= .*|drift_bootstrap_mask     = false|" \
              -e "s|^drift_bootstrap_t_clip *= .*|drift_bootstrap_t_clip = ${BOOT_TCLIP}|")
else
    REF_DRIFT=$DRIFT;  SEED=null
    SEED_SED=(-e "s|^drift_bootstrap_seed *= .*|drift_bootstrap_seed        = \"null\"|" \
              -e "s|^drift_bootstrap_start_gamma *= .*|drift_bootstrap_start_gamma = inf|" \
              -e "s|^drift_bootstrap_t_clip *= .*|drift_bootstrap_t_clip = ${BOOT_TCLIP}|")
fi

# Ladder PREFIX down to this rung: every task shares the same warm-start path, so
# differences between tasks are the parameter under test and not the anneal route.
SCHED="["
for ((i = 0; i <= GIDX; i++)); do
    SCHED+="${LADDER[$i]}";  (( i < GIDX )) && SCHED+=", "
done
SCHED+="]"

# ── Case → config-generator flags + output path (mirrors SB/main.py) ────────
GEN_DIR="${SLURM_TMPDIR:-/tmp}/sb_aoa_cfgs_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$GEN_DIR"
if [[ "$CASE" == "axisaoa" ]]; then
    # The bridge interpolates in ANGLE at fixed Mach → its own output root.
    CFG_ARGS=(--axis aoa --mach "$FIXED_MACH")
    OUT_ROOT="AoA_interpolation"
    AXIS_LVL="M${FIXED_MACH}/"
else
    AOA_TAG=$(LC_ALL=C printf "%.2f" "${CASE#aoa}")
    CFG_ARGS=(--axis mach --aoa "$AOA_TAG")
    OUT_ROOT="Mach_interpolation"
    # Empty at zero incidence so the already-finished tree keeps its exact paths.
    if [[ "$AOA_TAG" == "0.00" ]]; then AXIS_LVL=""; else AXIS_LVL="aoa${AOA_TAG}/"; fi
fi
OUT="outputs/${OUT_ROOT}/diamond/iso/hessian/${HMODE}/${H_TAG}/${AXIS_LVL}drift_${DRIFT}/gmin${GMIN}"

echo "#############################################"
echo "Array ${SLURM_ARRAY_JOB_ID}/${ID} of ${N_TASKS}"
echo "Mesh:   ${H_TAG}   case=${CASE}   drift=${DRIFT}   gmin=${GMIN}"
echo "Sched:  ${SCHED}"
echo "Out:    ${OUT}"
nvidia-smi --query-gpu=index,name --format=csv,noheader
echo "#############################################"

if [[ -f "${OUT}/metrics.json" ]]; then
    echo "  [skip] already done"; exit 0
fi

BASE_CFG="$GEN_DIR/Config_${H_TAG}_${CASE}_${DRIFT}_gmin${GMIN}_base.toml"
uv run python SB/make_config_for_mesh.py --h-tag "$H_TAG" "${CFG_ARGS[@]}" --out "$BASE_CFG" \
    || { echo "[FAIL] config generation (are the Euler bundles present?)"; exit 1; }

CFG="$GEN_DIR/Config_${H_TAG}_${CASE}_${DRIFT}_gmin${GMIN}.toml"
sed -e "s|^reference_drift = .*|reference_drift = \"${REF_DRIFT}\"|" \
    "${SEED_SED[@]}" \
    -e "s|^gamma_sb_schedule = .*|gamma_sb_schedule = ${SCHED}|" \
    -e "s|^gamma_sb *= .*|gamma_sb = ${GMIN}|" \
    -e "s|^density_mode *= .*|density_mode = \"iso_hessian\"|" \
    -e "s|^hessian_mode *= .*|hessian_mode = \"${HMODE}\"|" \
    -e "s|^compute_w2 *= .*|compute_w2 = true|" \
    "$BASE_CFG" > "$CFG"

SB_CONFIG="$CFG" uv run python SB/main.py

echo "######## ${CASE}/${DRIFT}/gmin${GMIN} (${H_TAG}) done ########"
