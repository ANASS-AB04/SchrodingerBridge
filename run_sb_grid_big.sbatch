#!/usr/bin/env bash
#SBATCH --job-name=sb-big
#SBATCH --partition=gpu-h100,gpu-l40,gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-47%8
#SBATCH --output=logs/sb-big_%A_%a.out
#SBATCH --error=logs/sb-big_%A_%a.err

# ─────────────────────────────────────────────────────────────────────────────
# The FULL SB matrix: every Mach pair, every AoA pair, every drift.
#
#   sbatch run_sb_grid_big.sbatch                        # h0.025, all 576 runs
#   CHUNK=1 sbatch --array=0 run_sb_grid_big.sbatch       # smoke: one run
#   GIDXS="0 1 2 3 4 5 6 7 8 9 10 11" sbatch --array=0-191 run_sb_grid_big.sbatch
#
# PREREQUISITE: run_euler_grid_big.sbatch must have finished for this H_TAG.
#
#   pairs  = 22 Mach pairs (0.80-0.90 … 2.90-3.00) x |MACH_AXIS_AOAS|
#          +  4 AoA  pairs (0-1, 1-2, 2-3, 3-4) @ FIXED_MACH         = 48
#   drifts = null | oblique | ffd | SBsquared_exact+ffd               =  4
#   gammas = ladder prefixes ending at 1e-2, 1e-3, 1e-4               =  3
#                                                               576 runs
#
#   run  = ((pair * N_DRIFT) + drift) * N_GAMMA + gamma
#   task = CHUNK consecutive runs  =>  576/12 = 48 array tasks
#
# WHY CHUNKED.  The cluster enforces QOSMaxSubmitJobPerUserLimit and a 576-task
# array trips it outright (that is a SUBMIT-COUNT cap, unrelated to the 1-2 day
# walltime cap).  Twelve runs per task fits comfortably, and since the walltime
# allowance is generous the only cost is coarser scheduling granularity.  A run
# that fails is counted and the rest of the chunk continues.
#
# WHY ONLY 3 GAMMAS BY DEFAULT.  The gamma-dependence is already mapped in
# detail by the 576-run h0.025 study; what is unknown here is how the METHODS
# rank as the flow regime changes across M=0.8..3.0.  Three rungs place each
# pair on the curve at a quarter of the cost.  Widen with GIDXS once the survey
# says which Mach band deserves the full ladder.
#
# THE OBLIQUE GUARD.  `oblique` is built from the analytic theta-beta-M relation,
# which has NO SOLUTION when the leading-edge shock detaches.  For the 5 deg
# half-wedge that happens around M=1.25: beta_le returns NaN at M=1.20 and below.
# Those (drift, pair) combinations are skipped with a reason rather than left to
# poison a kernel with NaN.  The check calls beta_le itself, so it stays correct
# if the geometry changes.
#
# THE REGIME CAVEAT.  Below M=1 there are no shocks on the diamond and the
# Hessian shock-marginal has nothing to key on; between 1.0 and ~1.25 the shock
# is a detached bow shock.  Those pairs are exploratory — the pipeline was built
# and validated for M >= 1.3.  They are included because they were asked for, and
# because a negative result there is itself informative, but do not read them as
# equal-confidence to the supersonic band.
#
# ONE GPU PER TASK.  A pooled single-node job caps at the 2 GPUs an h100 node
# has; as an array SLURM spreads tasks over every free L40 and H100.  Each task
# owns its output directory, so there is no concurrent-write race on metrics.json.
#
# COST at h0.025: ~5-12 min per RUN => ~77 GPU-h total; ~1.5-2.5 h per 12-run
# task, so 24 h walltime carries a wide margin even for the 12-stage gammas.
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

# Defaulted: under `set -u` a submission WITHOUT --array leaves the array vars
# unbound and the job would die on this line before doing any work.
JAX_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/jax-cache-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}-${SLURM_ARRAY_TASK_ID:-0}"
rm -rf "$JAX_CACHE_ROOT"; mkdir -p "$JAX_CACHE_ROOT"
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_ROOT"

CASE=${CASE:-diamond}                       # diamond | bump
H_TAG=${H_TAG:-h0.025}
HMODE=${HMODE:-eig}
FIXED_MACH=$(LC_ALL=C printf "%.2f" "${FIXED_MACH:-2.00}")

M_START=${M_START:-0.80};  M_STOP=${M_STOP:-3.00};  M_WIDTH=${M_WIDTH:-0.10}
# The bump has NO incidence axis (see the Euler script) and NO oblique drift:
# `oblique` is the analytic theta-beta-M wedge solution, which describes the
# diamond's leading edge and has no counterpart on a channel-wall bump.  Both
# defaults therefore change with the case rather than being silently wrong.
if [[ "$CASE" == "bump" ]]; then
    read -r -a MACH_AXIS_AOAS <<< "${MACH_AXIS_AOAS:-0.00}"
    read -r -a AOA_PAIRS <<< "${AOA_PAIRS:-}"
else
    read -r -a MACH_AXIS_AOAS <<< "${MACH_AXIS_AOAS:-0.00 2.00}"
    read -r -a AOA_PAIRS <<< "${AOA_PAIRS:-0.00-1.00 1.00-2.00 2.00-3.00 3.00-4.00}"
fi

# DRIFTS entries may be composite: <reference_drift>+<seed>, e.g.
#   SBsquared_exact+ffd = exact time-dependent bootstrap SEEDED by the FFD
#   registration, held fixed until gamma < BOOT_SWITCH, then self-refining.
if [[ "$CASE" == "bump" ]]; then
    read -r -a DRIFTS <<< "${DRIFTS:-null ffd SBsquared_exact+ffd}"
else
    read -r -a DRIFTS <<< "${DRIFTS:-null oblique ffd SBsquared_exact+ffd}"
fi
BOOT_SWITCH=${BOOT_SWITCH:-0.001}
# beta(t) endpoint clip for the exact mode — t=1 is singular (Q_0 = identity, so
# the raw log-potential's gradient enters beta and sets the CFL for every stage).
BOOT_TCLIP=${BOOT_TCLIP:-0.2}
# Runs PER TASK.  The cluster caps submitted jobs per user, so 576 one-run
# tasks trip QOSMaxSubmitJobPerUserLimit; 12 per task makes it 48 tasks.
CHUNK=${CHUNK:-12}

LADDER=(0.5 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.0005 0.0002 0.0001)
read -r -a GIDXS <<< "${GIDXS:-5 8 11}"

# ── Build the pair list in a STABLE order ───────────────────────────────────
# Encoding: "M:<aoa>:<m0>:<m1>"  or  "A:<mach>:<a0>:<a1>".
PAIRS=()
# ONE python call emitting "m0:m1" for every Mach pair.  Computing m1 per pair in
# its own subprocess would cost ~44 x 1 s of interpreter startup per task, before
# any GPU work begins — and bash cannot do the float arithmetic itself.
mapfile -t MPAIRS < <(uv run python -c "
import numpy as np
for v in np.arange($M_START, $M_STOP - $M_WIDTH + 1e-9, $M_WIDTH):
    print(f'{v:.2f}:{v + $M_WIDTH:.2f}')" 2>/dev/null)
if [[ ${#MPAIRS[@]} -eq 0 ]]; then echo "[FAIL] could not build the Mach pair list"; exit 1; fi
for a in "${MACH_AXIS_AOAS[@]}"; do
    a=$(printf '%.2f' "$a")
    for mp in "${MPAIRS[@]}"; do PAIRS+=("M:${a}:${mp}"); done   # mp is "m0:m1"
done
for ap in ${AOA_PAIRS[@]+"${AOA_PAIRS[@]}"}; do
    [[ -z "$ap" ]] && continue
    PAIRS+=("A:${FIXED_MACH}:${ap%-*}:${ap#*-}")
done

NP=${#PAIRS[@]};  ND=${#DRIFTS[@]};  NG=${#GIDXS[@]}
N_RUNS=$(( NP * ND * NG ))
N_TASKS=$(( (N_RUNS + CHUNK - 1) / CHUNK ))
ID=${SLURM_ARRAY_TASK_ID:-0}

echo "############# grid: ${NP} pairs x ${ND} drifts x ${NG} gammas = ${N_RUNS} runs"
echo "############# chunk ${CHUNK} => ${N_TASKS} tasks (need --array=0-$((N_TASKS-1)))"

if [[ "$ID" -ge "$N_TASKS" ]]; then
    echo "task $ID >= $N_TASKS — nothing to do"; exit 0
fi

# ── Run one (pair, drift, gamma) combination ────────────────────────────────
# Wrapped in a function so a task can process CHUNK of them back to back.  This
# exists because the cluster caps SUBMITTED jobs per user, and a 576-task array
# trips QOSMaxSubmitJobPerUserLimit; it also amortises the ~40 s of `uv run` +
# JAX import + XLA compile that every task would otherwise pay on its own.
# Returns 0 on success/skip, 1 on failure — a failure must NOT take the rest of
# the chunk down with it, so the caller only counts it.
run_one() {
    local RUN_ID=$1
    local PIDX=$(( RUN_ID / (ND * NG) ))
    local REM=$((  RUN_ID % (ND * NG) ))
    local DIDX=$(( REM / NG ))
    local GPOS=$(( REM % NG ))
    local PAIR=${PAIRS[$PIDX]};  local DRIFT=${DRIFTS[$DIDX]};  local GIDX=${GIDXS[$GPOS]}
    local GMIN=${LADDER[$GIDX]}
    local KIND FIX P0 P1

IFS=':' read -r KIND FIX P0 P1 <<< "$PAIR"

# ── Case → config-generator flags + output path (mirrors SB/main.py) ────────
if [[ "$KIND" == "A" ]]; then
    # The bridge interpolates in ANGLE at fixed Mach → its own output root.
    CFG_ARGS=(--axis aoa --mach "$FIX" --p0 "$P0" --p1 "$P1")
    OUT_ROOT="AoA_interpolation"
    AXIS_LVL="M${FIX}/"
    # Empty for the legacy 0->4 sweep so the finished tree keeps its exact paths.
    if [[ "$P0" == "0.00" && "$P1" == "4.00" ]]; then RANGE=""; else RANGE="A${P0}-${P1}/"; fi
    M_LO=$FIX;  M_HI=$FIX
else
    CFG_ARGS=(--axis mach --aoa "$FIX" --p0 "$P0" --p1 "$P1")
    OUT_ROOT="Mach_interpolation"
    # Empty at zero incidence so the already-finished tree keeps its exact paths.
    if [[ "$FIX" == "0.00" ]]; then AXIS_LVL=""; else AXIS_LVL="aoa${FIX}/"; fi
    if [[ "$P0" == "2.00" && "$P1" == "2.50" ]]; then RANGE=""; else RANGE="M${P0}-${P1}/"; fi
    M_LO=$P0;  M_HI=$P1
fi
OUT="outputs/${OUT_ROOT}/${CASE}/iso/hessian/${HMODE}/${H_TAG}/${AXIS_LVL}${RANGE}drift_${DRIFT}/gmin${GMIN}"

echo "#############################################"
echo "--- run ${RUN_ID}/${N_RUNS}: pair=${PAIR} drift=${DRIFT} gmin=${GMIN}"
echo "    out: ${OUT}"
echo "#############################################"

if [[ -f "${OUT}/metrics.json" ]]; then
    echo "  [skip] already done"; return 0
fi

# ── Oblique-drift domain guard ──────────────────────────────────────────────
# theta-beta-M has no attached-shock solution below the detachment Mach (~1.25
# for this 5 deg half-wedge).  Ask beta_le directly rather than hardcoding a
# threshold, so the guard follows the geometry.
if [[ "$DRIFT" == oblique* && "$CASE" != "diamond" ]]; then
    echo "  [skip] oblique drift is diamond-only (no wedge on a ${CASE})"
    return 0
fi
if [[ "$DRIFT" == oblique* ]]; then
    uv run python - "$M_LO" "$M_HI" <<'PY'
import sys, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "Euler")
from SB import drift as D
alpha = np.arctan(0.087488663525924 / 1.0)          # 5 deg half-wedge
for m in sys.argv[1:3]:
    try:
        b = D.beta_le(float(m), alpha)
    except Exception:                                # noqa: BLE001
        b = float("nan")
    if not np.isfinite(b):
        print(f"  shock DETACHED at M={m} (beta_le undefined) — oblique drift "
              f"is not defined for this pair")
        raise SystemExit(3)
raise SystemExit(0)
PY
    RC=$?                     # capture immediately — any command in between clobbers it
    if [[ $RC -eq 3 ]]; then
        echo "  [skip] oblique drift out of domain for ${PAIR}"; return 0
    elif [[ $RC -ne 0 ]]; then
        echo "[FAIL] oblique domain guard errored (rc=$RC)"; return 1
    fi
fi

# Split a composite drift into (reference_drift, seed).  Seeded variants hold the
# seed fixed until BOOT_SWITCH then bootstrap; the density mask is switched OFF
# for them (the seed is already void-free, masking a registration-refined drift
# would only clip real transport).
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

# A seeded bootstrap needs the FFD registration; so does the FFD control column.
# ffd_control stays on everywhere so BaryCDI/Linear/FFD are three-way comparable
# in EVERY run, not just the ffd ones.

# Ladder PREFIX down to this rung: every task shares the same warm-start path, so
# differences between tasks are the parameter under test and not the anneal route.
SCHED="["
for ((i = 0; i <= GIDX; i++)); do
    SCHED+="${LADDER[$i]}";  (( i < GIDX )) && SCHED+=", "
done
SCHED+="]"
echo "Sched:  ${SCHED}"

GEN_DIR="${SLURM_TMPDIR:-/tmp}/sb_big_cfgs_${SLURM_ARRAY_JOB_ID:-local}"
mkdir -p "$GEN_DIR"
STEM="${H_TAG}_${KIND}${FIX}_${P0}-${P1}_${DRIFT}_gmin${GMIN}"

BASE_CFG="$GEN_DIR/Config_${STEM}_base.toml"
uv run python SB/make_config_for_mesh.py --h-tag "$H_TAG" --case "$CASE" "${CFG_ARGS[@]}" --out "$BASE_CFG" \
    || { echo "[FAIL] config generation — are all 11 Euler bundles present for ${PAIR}?"; return 1; }

CFG="$GEN_DIR/Config_${STEM}.toml"
sed -e "s|^case *= *\"diamond\"|case       = \"${CASE}\"|" \
    -e "s|^reference_drift = .*|reference_drift = \"${REF_DRIFT}\"|" \
    "${SEED_SED[@]}" \
    -e "s|^gamma_sb_schedule = .*|gamma_sb_schedule = ${SCHED}|" \
    -e "s|^gamma_sb *= .*|gamma_sb = ${GMIN}|" \
    -e "s|^density_mode *= .*|density_mode = \"iso_hessian\"|" \
    -e "s|^hessian_mode *= .*|hessian_mode = \"${HMODE}\"|" \
    -e "s|^compute_w2 *= .*|compute_w2 = true|" \
    "$BASE_CFG" > "$CFG"

    if SB_CONFIG="$CFG" uv run python SB/main.py; then
        echo "    [ok] ${PAIR}/${DRIFT}/gmin${GMIN}"
        return 0
    fi
    echo "    [FAIL] ${PAIR}/${DRIFT}/gmin${GMIN}"
    return 1
}

# ── This task's chunk ───────────────────────────────────────────────────────
START=$(( ID * CHUNK ))
END=$(( START + CHUNK - 1 ));  (( END >= N_RUNS )) && END=$(( N_RUNS - 1 ))
echo "Task ${ID}: runs ${START}..${END}   case ${CASE}   mesh ${H_TAG}"
nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null

N_OK=0;  N_BAD=0
for (( RID = START; RID <= END; RID++ )); do
    if run_one "$RID"; then N_OK=$(( N_OK + 1 )); else N_BAD=$(( N_BAD + 1 )); fi
done
echo "######## task ${ID}: ${N_OK} ok, ${N_BAD} failed (runs ${START}..${END}) ########"
exit 0
