#!/usr/bin/env bash
#SBATCH --job-name=euler-big
#SBATCH --partition=gpu-l40,gpu-rtx6000,gpu-h100,gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --array=0-60%12
#SBATCH --output=logs/euler-big_%A_%a.out
#SBATCH --error=logs/euler-big_%A_%a.err

# ─────────────────────────────────────────────────────────────────────────────
# HR snapshot generation for the FULL Mach x AoA study.
#
#   sbatch run_euler_grid_big.sbatch                     # h0.025, everything
#   sbatch --array=0 run_euler_grid_big.sbatch           # smoke: one chunk
#   H_TAG=h0.0175 sbatch run_euler_grid_big.sbatch       # a finer mesh
#
# WHAT IT COVERS.  Every SB pair needs its two endpoints AND the 9 interior
# references at t=k/10, so a 0.10-wide Mach pair needs snapshots every 0.01:
#
#   Mach axis : M 0.80 -> 3.00 step 0.01, at each AoA in MACH_AXIS_AOAS   221/AoA
#   AoA  axis : AoA 0.0 -> 4.0 step 0.1 at M = FIXED_MACH                     41
#
# With the default AoA set {0, 2} that is 442 + 41 - 2 shared = 481 solves.
#
# CHUNKED, not one task per solve.  A single solve is ~15-30 s at h0.025 but
# `uv run` + JAX import + XLA compile costs ~40 s, so one-task-per-solve would
# spend more than half the allocation on startup.  Each task runs CHUNK pairs
# back to back instead; at CHUNK=8 the 481 solves become 61 tasks and the fixed
# cost is amortised ~8x.  Raise --array if you widen the grid: the script prints
# the required size and exits non-zero if the array is too small to cover it.
#
# THE TRANSONIC/SUBSONIC WARNING.  Below M=1 there are no shocks on the diamond
# at all, and between 1.0 and ~1.25 the leading-edge shock is DETACHED (a bow
# shock).  The Euler solver will still produce a field, but (a) it may not reach
# a steady state, and (b) the whole SB pipeline downstream keys on a Hessian
# SHOCK marginal.  Those runs are exploratory; the validated regime is M >= 1.3.
# ─────────────────────────────────────────────────────────────────────────────

set -uo pipefail
# bash printf honours LC_NUMERIC: under fr_FR a "%.2f" of 2 yields "2,00", which
# would corrupt every filename and glob built from it.
export LC_ALL=C
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH=""
export MPLBACKEND=Agg
# Unbuffered stdout: python block-buffers when redirected to a file, so a
# long-running task shows nothing in its log for many minutes and cannot be
# monitored -- which matters here because the failure mode we care about
# (NaN residuals) is visible in the first stage line or not at all.
export PYTHONUNBUFFERED=1
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# One compilation cache PER TASK but shared across the chunk's pairs: every pair
# in a chunk has identical mesh and solver settings, so pairs 2..N reuse pair 1's
# compiled kernels and pay only the solve.
# Defaulted: under `set -u` a submission WITHOUT --array leaves the array vars
# unbound and the job would die on this line before doing any work.
JAX_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/jax-cache-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}-${SLURM_ARRAY_TASK_ID:-0}"
rm -rf "$JAX_CACHE_ROOT"; mkdir -p "$JAX_CACHE_ROOT"
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_ROOT"

CASE=${CASE:-diamond}   # diamond | bump | naca0012 | naca2412 | rae2822 | oneraD | oa209
H_TAG=${H_TAG:-h0.025}
MESH="meshes/${CASE}/${CASE}_${H_TAG}.npy"
FIXED_MACH=$(LC_ALL=C printf "%.2f" "${FIXED_MACH:-2.00}")
CHUNK=${CHUNK:-8}

# Mach grid: start:stop:step, at 0.01 so every 0.10-wide pair gets its 9 refs.
M_START=${M_START:-0.80};  M_STOP=${M_STOP:-3.00};  M_STEP=${M_STEP:-0.01}
A_START=${A_START:-0.00};  A_STOP=${A_STOP:-4.00};  A_STEP=${A_STEP:-0.10}
# The BUMP is the only geometry with no incidence: it sits on a channel wall,
# and Euler/config.py:format_condition_tag omits the AoA from its snapshot names
# accordingly.  diamond and every airfoil carry an AoA and get the full grid.
if [[ "$CASE" == "bump" ]]; then
    read -r -a MACH_AXIS_AOAS <<< "${MACH_AXIS_AOAS:-0.00}"
    DO_AOA_AXIS=${DO_AOA_AXIS:-0}
else
    read -r -a MACH_AXIS_AOAS <<< "${MACH_AXIS_AOAS:-0.00 2.00}"
    DO_AOA_AXIS=${DO_AOA_AXIS:-1}
fi

# ── Build the de-duplicated union in a STABLE order ─────────────────────────
# Stability matters: the array index must mean the same work on a resubmission,
# or --array=<subset> would target different pairs than intended.
PAIRS=()
seen=" "
add_pair() {                       # $1 = AoA tag, $2 = Mach tag
    local key="$1_$2"
    case "$seen" in *" $key "*) return;; esac
    seen="$seen$key "
    PAIRS+=("$key")
}
mapfile -t MACHS < <(uv run python -c "
import numpy as np
for v in np.arange($M_START, $M_STOP + 1e-9, $M_STEP): print(f'{v:.2f}')" 2>/dev/null)
mapfile -t AOAS < <(uv run python -c "
import numpy as np
for v in np.arange($A_START, $A_STOP + 1e-9, $A_STEP): print(f'{v:.2f}')" 2>/dev/null)
if [[ ${#MACHS[@]} -eq 0 || ${#AOAS[@]} -eq 0 ]]; then
    echo "[FAIL] could not build the parameter grids"; exit 1
fi

for a in "${MACH_AXIS_AOAS[@]}"; do
    a=$(printf '%.2f' "$a")
    for m in "${MACHS[@]}"; do add_pair "$a" "$m"; done
done
if [[ "$DO_AOA_AXIS" == "1" ]]; then
    for a in "${AOAS[@]}"; do add_pair "$a" "$FIXED_MACH"; done
fi

N_PAIRS=${#PAIRS[@]}
N_TASKS=$(( (N_PAIRS + CHUNK - 1) / CHUNK ))
ID=${SLURM_ARRAY_TASK_ID:-0}

echo "#############################################"
echo "Array ${SLURM_ARRAY_JOB_ID:-local}/${ID}"
echo "Case:   ${CASE}   mesh ${MESH}   (${H_TAG})"
echo "Grid:   ${N_PAIRS} pairs, chunk ${CHUNK} => ${N_TASKS} tasks (need --array=0-$((N_TASKS-1)))"
nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null
echo "#############################################"

if [[ "$ID" -ge "$N_TASKS" ]]; then
    echo "task $ID >= $N_TASKS — nothing to do"; exit 0
fi

# ── Mesh generation (once per task, only if absent) ─────────────────────────
# The airfoils ship no pre-built meshes, and generate_mesh.py writes
# <case>_h<h>.npy -- exactly the name this script expects.  Concurrent array
# tasks may race here, so build into a private temp name and move atomically.
if [[ ! -f "$MESH" ]]; then
    echo "  [mesh] $MESH absent -> generating"
    # Build into a temp dir ON THE SAME FILESYSTEM as the target.  The previous
    # version staged in $SLURM_TMPDIR (node-local) and moved to meshes/ (NFS):
    # a cross-device `mv` is copy-then-unlink, NOT atomic, so a concurrent task
    # could load a half-written .npy and silently solve on a corrupt mesh.
    # Within one filesystem, `mv` is a rename() and is atomic.
    MESH_TMP="meshes/${CASE}/.tmp_${H_TAG}_${SLURM_ARRAY_TASK_ID:-0}_$$"
    mkdir -p "$MESH_TMP"
    # A nose-refined tag h<h>le<N> (rounded-LE airfoils) is the standard mesh plus
    # a nested solver mesh refined to h/N at the leading edge: Euler solves on the
    # latter and writes bundles on the former, so SB and its cost are unchanged.
    H_NUM=${H_TAG#h};  NOSE_ARGS=()
    if [[ "$H_NUM" == *le* ]]; then
        NOSE_ARGS=(--nose-factor "${H_NUM#*le}");  H_NUM=${H_NUM%%le*}
    fi
    if uv run python Euler/generate_mesh.py --geom "$CASE" -h "$H_NUM" \
            ${NOSE_ARGS[@]+"${NOSE_ARGS[@]}"} --out-dir "$MESH_TMP"; then
        # `mv -n` so a task that lost the race does not clobber a good mesh.
        # Solver mesh FIRST: the existence test above keys on $MESH, so a task
        # that sees $MESH must also find the solver mesh it points to.
        mv -n "$MESH_TMP"/*_solver.npy "meshes/${CASE}/" 2>/dev/null
        mv -n "$MESH_TMP"/*.npy "meshes/${CASE}/" 2>/dev/null
        rm -rf "$MESH_TMP"
    else
        echo "[FAIL] mesh generation for ${CASE} at ${H_TAG}"; exit 1
    fi
    [[ -f "$MESH" ]] || { echo "[FAIL] ${MESH} still missing after generation"; exit 1; }
fi

# ── Geometry guard (once per task, diamond only) ──────────────────────────────────────────
# The fine meshes shipped before 2026-08 were the alpha=10 deg wedge while
# h0.025 is alpha=5 deg; a study spanning both would compare two airfoils.
# The bump has no equivalent invariant to assert, so it skips this entirely.
if [[ "$CASE" == "diamond" ]]; then
uv run python - "$MESH" <<'PY' || { echo "[FAIL] geometry guard"; exit 1; }
import sys
sys.path.insert(0, "."); sys.path.insert(0, "Euler")
from Euler.jax_fvm.src.mesh import Mesh
m = Mesh(); m.load_mesh(sys.argv[1])
h = m.metadata["height"]
ok = abs(h - 0.087488663525924) < 1e-12
print(f"  geometry: N={len(m.tris)}  height={h:.9f}  5deg={ok}")
raise SystemExit(0 if ok else 1)
PY
fi

# ── Run this task's chunk ───────────────────────────────────────────────────
START=$(( ID * CHUNK ))
END=$(( START + CHUNK - 1 ));  (( END >= N_PAIRS )) && END=$(( N_PAIRS - 1 ))
echo "  chunk covers pairs ${START}..${END}"

N_OK=0;  N_SKIP=0;  N_FAIL=0
for (( i = START; i <= END; i++ )); do
    PAIR=${PAIRS[$i]}
    R_AOA=${PAIR%_*}
    R_MACH=${PAIR#*_}
    # Only bump omits the AoA field; diamond and all airfoils carry it.
    # Mirrors Euler/config.py:format_condition_tag -- keep the two in step.
    if [[ "$CASE" == "bump" ]]; then STEM="M${R_MACH}"
    else                            STEM="AOA${R_AOA}_M${R_MACH}"; fi

    # Stale-bundle check: a bundle may exist but have been computed on a
    # DIFFERENT mesh at the same h_tag (exactly what the 10deg/5deg swap did).
    # SB then loads the new mesh against old fields and dies with "length of
    # facecolors must match the number of triangles".  Quarantine, never delete.
    uv run python - "$H_TAG" "$MESH" "$STEM" "$CASE" <<'PY'
import sys, os, glob, shutil, time
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "Euler")
from Euler.jax_fvm.src.mesh import Mesh
h_tag, mesh_path, stem, case = sys.argv[1:5]
m = Mesh(); m.load_mesh(mesh_path)
n_mesh = len(m.tris)
res_dir = os.path.join("results", case, h_tag)
stale_dir = os.path.join(res_dir, f"stale_{time.strftime('%Y%m%d')}")
for f in sorted(glob.glob(os.path.join(res_dir, f"{stem}_*.npz"))):
    try:
        d = np.load(f, allow_pickle=True)
        n = int(np.asarray(d["n_cells"]).ravel()[0]); d.close()
    except Exception as exc:                                    # noqa: BLE001
        n, why = -1, f"unreadable ({exc})"
    else:
        why = None if n == n_mesh else f"n_cells={n} != mesh {n_mesh} (different mesh!)"
    if why:
        os.makedirs(stale_dir, exist_ok=True)
        shutil.move(f, os.path.join(stale_dir, os.path.basename(f)))
        print(f"  [stale] {os.path.basename(f)}: {why}")
PY

    if compgen -G "results/${CASE}/${H_TAG}/${STEM}_*.npz" > /dev/null; then
        echo "  [skip ${i}] ${STEM} already present"
        N_SKIP=$(( N_SKIP + 1 ));  continue
    fi

    echo "  [run  ${i}] ${STEM}"
    # --stationarity-threshold 0 runs every case to tf.  Euler/main.py normalises
    # the stationarity residual by STEP COUNT, not elapsed time, so the criterion
    # is (physical rate)*dt and dt comes from the CFL — a finer mesh reports a
    # smaller residual for the same physical state and stops EARLIER in physical
    # time.  For a study comparing snapshots across meshes that is fatal.
    # --aoa is a diamond-only flag; it is meaningless for the bump
    AOA_ARGS=()
    [[ "$CASE" != "bump" ]] && AOA_ARGS=(--aoa "${R_AOA}")
    if uv run python Euler/main.py \
            --case "$CASE" --mach "${R_MACH}" ${AOA_ARGS[@]+"${AOA_ARGS[@]}"} \
            --mesh-path "${MESH}" \
            --flux HLLC --time-scheme SRK2 --reconstruction MUSCL \
            --stationarity-threshold 0; then
        N_OK=$(( N_OK + 1 ))
    else
        # Transonic/subsonic cases may genuinely fail to converge.  One bad Mach
        # must not take the other seven in the chunk down with it.
        echo "  [WARN] ${STEM} FAILED — continuing"
        N_FAIL=$(( N_FAIL + 1 ))
    fi
done

echo "######## chunk ${ID}: ${N_OK} solved, ${N_SKIP} skipped, ${N_FAIL} failed ########"
exit 0
