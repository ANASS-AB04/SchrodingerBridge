#!/usr/bin/env bash
#SBATCH --job-name=warmstart-bench
#SBATCH --partition=gpu-l40,gpu-rtx6000,gpu-h100,gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# 1h30, not 4h.  Measured: ~5 min SB + ~30 min for 36 solves, plus up to ~10 min
# of Euler solves when a band's snapshots are missing -- so ~50 min worst case.
# The walltime you ASK for is what SLURM backfills against: on a full cluster a
# 90-minute job slots into gaps a 4-hour job waits behind.  Raise it only if you
# widen TVALS or move to a finer mesh.
#SBATCH --time=01:30:00
# Flat, not logs/%x/%j.out: SLURM opens these at SUBMISSION, before any mkdir in
# the script body runs, so a nested path fails the job instantly with no log to
# explain why.  `mkdir -p logs` once is enough for this script.
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# How much solver time does an interpolated snapshot save as a warm start?
#
#   sbatch run_warmstart_bench.sbatch                      # SB + 36 solves
#   SKIP_SB=1 sbatch run_warmstart_bench.sbatch            # reuse warmstart.npz
#   TVALS=0.5 sbatch run_warmstart_bench.sbatch            # smoke: 4 solves
#   CASE=bump sbatch run_warmstart_bench.sbatch
#
# STEP 1 runs SB once with save_warmstart=true.  ffd_control is on by default, so
# a single run emits BaryCDI, Linear AND FFD seeds together -- there is no
# separate run per method.  Skipped when warmstart.npz already exists.
#
# STEP 2 runs the solver from each seed plus a cold baseline, at every
# interpolation time, and writes warmstart_bench.csv.
#
# Every variant runs the SAME fixed N with --stationarity-threshold 0.  Nothing
# stops early, so trip count, compile and wall time are identical across variants
# and the lax.scan early-exit tail (a skipped step still costs ~8% of a real one)
# cannot distort the comparison.  --save-convergence records the whole
# stationarity curve, so "iterations to reach tolerance X" is read off afterwards
# for any X -- which matters because the solver has a residual FLOOR
# (measured: rel plateaus at ~7e-6 at check_every=25, and ||R||/||R_0|| at
# ~1.6e-2), so a threshold picked up front can easily sit below the reachable
# range and every row comes back "never".
# ─────────────────────────────────────────────────────────────────────────────

set -uo pipefail

mkdir -p logs

# CALI3: no cuda/uv modules.  Do NOT load a system CUDA -- its libcudnn shadows
# the one bundled in the jax wheels and reproduces `undefined symbol:
# cudnnGetLibConfig`.  jax[cuda12] runs on the newer driver via forward-compat.
module purge
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH=""

export MPLBACKEND=Agg
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

JAX_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/jax-cache"
rm -rf "$JAX_CACHE_ROOT"; mkdir -p "$JAX_CACHE_ROOT"
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_ROOT"

CASE="${CASE:-diamond}"
H_TAG="${H_TAG:-h0.025}"
AXIS="${AXIS:-mach}"          # mach | aoa  (which quantity SB interpolated along)
BAND="${BAND:-}"              # e.g. M1.80-1.90 or A0.00-1.00; empty = legacy pair
AOATAG="${AOATAG:-}"          # e.g. aoa2.00; empty = AoA 0 (no directory level)
FIXED_MACH="${FIXED_MACH:-}"  # AXIS=aoa only: Mach held fixed
TVALS="${TVALS:-0.1:0.9:0.1}"
METHODS="${METHODS:-}"
CHECK_EVERY="${CHECK_EVERY:-25}"
REPEAT="${REPEAT:-1}"
TOL="${TOL:-}"
SKIP_SB="${SKIP_SB:-0}"
DRIFT="${DRIFT:-ffd}"
# Base SB config to sed on top of.  SB/Config.toml only carries [run.mach.diamond]
# and [run.mach.bump] with hardcoded bundle paths, so any other case or Mach/AoA
# band needs a config generated first:
#   uv run python SB/make_config_for_mesh.py --case naca0012 --h-tag h0.025 \
#       --axis mach --p0 1.80 --p1 1.90 --out SB/Config_naca_M1.80-1.90.toml
# It GLOBS the bundles actually on disk, which matters because the snapshot name
# embeds the stop time and different Mach runs stop at different times.
BASE_CFG="${BASE_CFG:-SB/Config.toml}"
GAMMA_SCHED="${GAMMA_SCHED:-[0.0003, 0.0002, 0.0001]}"

MESH="meshes/${CASE}/${CASE}_${H_TAG}.npy"
# Mirrors SB/main.py:_mach_output_dir (density_mode=iso_hessian/eig).  The axis
# root, the optional aoa<x> level and the optional band level are all part of the
# path -- see the "Interpolation-pair levels" table in CLAUDE.md.  Both optional
# levels are EMPTY for the original pairs, which is why legacy runs keep their
# exact paths.
GMIN=$(python3 -c "import re,sys; s='${GAMMA_SCHED}'; print(f\"{min(float(x) for x in re.findall(r'[0-9.eE+-]+', s)):g}\")")
if [[ "$AXIS" == "aoa" ]]; then ROOT="AoA_interpolation"; else ROOT="Mach_interpolation"; fi
WS_DIR="outputs/${ROOT}/${CASE}/iso/hessian/eig/${H_TAG}"
[[ -n "$AOATAG" ]] && WS_DIR="${WS_DIR}/${AOATAG}"
[[ -n "$BAND"   ]] && WS_DIR="${WS_DIR}/${BAND}"
WS_DIR="${WS_DIR}/drift_${DRIFT}/gmin${GMIN}"
WS="${WS_DIR}/warmstart.npz"
# One tag per (axis, band, drift) so nothing collides in results/<case>/<h>/warmstart/
SUF="${DRIFT}"
[[ -n "$BAND"   ]] && SUF="${SUF}_${BAND}"
[[ -n "$AOATAG" ]] && SUF="${SUF}_${AOATAG}"
[[ "$AXIS" == "aoa" ]] && SUF="aoa_${SUF}"

echo "#############################"
echo "User:            $USER"
echo "SLURM_JOBID:     $SLURM_JOBID"
echo "Case / mesh:     ${CASE} / ${MESH}"
echo "Warm-start file: ${WS}"
echo "t values:        ${TVALS}"
echo "#############################"

# bash printf honours LC_NUMERIC: under fr_FR "%.2f" of 2 yields "2,00", which
# would corrupt every filename and glob built from it.
export LC_ALL=C

# ── STEP 0: mesh, Euler bundles and SB config, built on demand ───────────────
# SB needs 11 snapshots per band (2 endpoints + 9 references at t=k/10), and
# make_config_for_mesh.py GLOBS them rather than constructing names, so every one
# must exist before the config can even be written.  Rather than fail three times
# in a row, build whatever is missing here.  SKIP_EULER=1 to disable.
H_VAL="${H_TAG#h}"
AOA_FIXED="${AOATAG#aoa}"; AOA_FIXED="${AOA_FIXED:-0.00}"

if [[ ! -f "$MESH" ]]; then
    echo "=== [0/3] mesh $MESH missing — generating ==="
    mkdir -p "meshes/${CASE}"
    if ! uv run python Euler/generate_mesh.py --geom "$CASE" -h "$H_VAL" \
            --out-dir "meshes/${CASE}"; then
        echo "[FATAL] mesh generation failed for ${CASE} h=${H_VAL}"; exit 1
    fi
fi
[[ -f "$MESH" ]] || { echo "[FATAL] mesh still not found: $MESH"; exit 1; }

# Band endpoints: "M1.80-1.90" -> 1.80 1.90 ; "A0.00-1.00" -> 0.00 1.00.
# Empty BAND = the legacy pair, whose endpoints live in the base config already.
if [[ -n "$BAND" && "${SKIP_EULER:-0}" != "1" ]]; then
    P0="${BAND:1}"; P1="${P0#*-}"; P0="${P0%%-*}"
    echo "=== [0/3] checking the 11 ${AXIS} snapshots for band ${BAND} (${P0} -> ${P1}) ==="
    MISSING=0
    for k in $(seq 0 10); do
        PARAM=$(python3 -c "print(f'{${P0} + ${k}*(${P1}-${P0})/10:.2f}')")
        if [[ "$AXIS" == "aoa" ]]; then
            M_V="${FIXED_MACH:-2.00}"; A_V="$PARAM"
        else
            M_V="$PARAM"; A_V="$AOA_FIXED"
        fi
        # Case-dependent naming: Euler/config.py prefixes the AoA only for
        # diamond and the airfoils, never for the bump.
        if [[ "$CASE" == "bump" ]]; then STEM="M${M_V}"; else STEM="AOA${A_V}_M${M_V}"; fi
        if compgen -G "results/${CASE}/${H_TAG}/${STEM}_*.npz" > /dev/null; then continue; fi
        echo "  [euler] ${STEM} missing — solving"
        AOA_ARGS=(); [[ "$CASE" != "bump" ]] && AOA_ARGS=(--aoa "$A_V")
        if uv run python Euler/main.py --case "$CASE" --mach "$M_V" \
                ${AOA_ARGS[@]+"${AOA_ARGS[@]}"} --mesh-path "$MESH" \
                --flux HLLC --time-scheme SRK2 --reconstruction MUSCL \
                --stationarity-threshold 0 --quiet; then
            :
        else
            echo "  [WARN] ${STEM} FAILED"; MISSING=$(( MISSING + 1 ))
        fi
    done
    [[ "$MISSING" -gt 0 ]] && { echo "[FATAL] ${MISSING} snapshots could not be produced"; exit 1; }
fi

if [[ ! -f "$BASE_CFG" ]]; then
    echo "=== [0/3] SB config $BASE_CFG missing — generating ==="
    CFG_ARGS=(--case "$CASE" --h-tag "$H_TAG" --axis "$AXIS" --out "$BASE_CFG")
    if [[ -n "$BAND" ]]; then
        P0="${BAND:1}"; P1="${P0#*-}"; P0="${P0%%-*}"
        CFG_ARGS+=(--p0 "$P0" --p1 "$P1")
    fi
    if [[ "$AXIS" == "aoa" ]]; then
        CFG_ARGS+=(--mach "${FIXED_MACH:-2.00}")
    else
        CFG_ARGS+=(--aoa "$AOA_FIXED")
    fi
    if ! uv run python SB/make_config_for_mesh.py "${CFG_ARGS[@]}"; then
        echo "[FATAL] could not generate $BASE_CFG"; exit 1
    fi
fi
[[ -f "$BASE_CFG" ]] || { echo "[FATAL] base SB config not found: $BASE_CFG"; exit 1; }

# ── STEP 1: produce warmstart.npz ────────────────────────────────────────────
if [[ "$SKIP_SB" != "1" && ! -f "$WS" ]]; then
    echo "=== [1/2] SB run -> ${WS} ==="
    CFG="${SLURM_TMPDIR:-/tmp}/sb_warmstart_${SLURM_JOBID}.toml"
    sed -e 's/^save_warmstart = false/save_warmstart = true/' \
        -e "s/^gamma_sb_schedule = .*/gamma_sb_schedule = ${GAMMA_SCHED}/" \
        -e "s/^reference_drift = .*/reference_drift = \"${DRIFT}\"/" \
        "$BASE_CFG" > "$CFG"
    # SB needs float64: without it the IPFP truncates probabilities silently.
    if ! SB_CONFIG="$CFG" JAX_ENABLE_X64=true uv run python SB/main.py; then
        echo "[FATAL] SB run failed — no warm-start states to benchmark"; exit 1
    fi
else
    echo "=== [1/2] skipped (SKIP_SB=${SKIP_SB}, exists=$([[ -f "$WS" ]] && echo yes || echo no)) ==="
fi

if [[ ! -f "$WS" ]]; then
    echo "[FATAL] $WS not produced. Check the SB output dir above."; exit 1
fi

# ── STEP 2: the benchmark ────────────────────────────────────────────────────
# NOTE: no JAX_ENABLE_X64 here.  The solver runs float32, and the loader casts
# the seed to the dtype initialize() produces -- a float64 seed would double
# every buffer and trigger a separate compile, making the warm-started run
# slower for a reason unrelated to seed quality.
echo "=== [2/2] warm-start benchmark ==="
# The drift goes in BOTH the run tag and the csv name: BaryCDI is the SB solution
# CONDITIONED on the reference drift, so an ffd run and an oblique run produce
# different seeds under the same variant name.  Without this they overwrite each
# other and the A/B silently compares a run against itself.
ARGS=(--case "$CASE" --mesh-path "$MESH" --warmstart "$WS" --axis "$AXIS"
      --t "$TVALS" --check-every "$CHECK_EVERY" --repeat "$REPEAT"
      --tag-prefix "ws_${SUF}"
      --out-csv "warmstart_bench_${CASE}_${H_TAG}_${SUF}.csv")
[[ -n "$METHODS"    ]] && ARGS+=(--methods "$METHODS")
[[ -n "$TOL"        ]] && ARGS+=(--tol "$TOL")
[[ -n "$FIXED_MACH" ]] && ARGS+=(--mach "$FIXED_MACH")
# AoA level in the OUTPUT path means the solver must run at that incidence too
[[ -n "$AOATAG"     ]] && ARGS+=(--aoa "${AOATAG#aoa}")

uv run python slurm/run_warmstart_bench.py "${ARGS[@]}"
RC=$?

echo "#############################"
echo "exit code: $RC"
echo "csv:       warmstart_bench_${CASE}_${H_TAG}_${SUF}.csv"
echo "runs:      results/${CASE}/${H_TAG}/warmstart/"
exit $RC
