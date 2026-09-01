#!/usr/bin/env bash
#SBATCH --job-name=euler-aoa
#SBATCH --partition=gpu-h100,gpu-l40,gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-40%12
#SBATCH --output=logs/euler-aoa_%A_%a.out
#SBATCH --error=logs/euler-aoa_%A_%a.err

# ─────────────────────────────────────────────────────────────────────────────
# HR (Euler) snapshots for the angle-of-attack study.
#
#   sbatch run_euler_grid_aoa.sbatch                       # h0.025, all 41 pairs
#   H_TAG=h0.0175 sbatch run_euler_grid_aoa.sbatch
#   sbatch --array=0 run_euler_grid_aoa.sbatch             # smoke: one pair
#
# Covers the UNION of the two studies in one grid, so the three shared corners
# are solved once rather than twice:
#
#   Mach axis : AoA in {0, 2, 4} deg  x  M in 2.00 … 2.50 step 0.05   = 33
#   AoA  axis : AoA in 0 … 4 deg step 0.4  @  M = FIXED_MACH          = 11
#   shared    : (0,Mfix) (2,Mfix) (4,Mfix)                            = -3
#                                                              UNION  = 41
#
# ONE GPU PER TASK, by design.  A pooled single-node job caps at the 2 GPUs an
# h100 node has; as an array each task is independent and SLURM spreads them over
# every free L40 and H100 GPU on the cluster.  Small --mem and a short --time make
# them schedulable onto partially-occupied nodes instead of queueing for idle ones.
#
# Already-present bundles are skipped, so at h0.025 (13 of 41 on disk) this costs
# 28 solves of ~15-30 s each.
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
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

JAX_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/jax-cache-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
rm -rf "$JAX_CACHE_ROOT"; mkdir -p "$JAX_CACHE_ROOT"
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_ROOT"

H_TAG=${H_TAG:-h0.025}
MESH="meshes/diamond/diamond_${H_TAG}.npy"
FIXED_MACH=$(LC_ALL=C printf "%.2f" "${FIXED_MACH:-2.00}")

read -r -a MACH_AXIS_AOAS <<< "${MACH_AXIS_AOAS:-0.00 2.00 4.00}"
read -r -a MACHS      <<< "${MACHS:-2.00 2.05 2.10 2.15 2.20 2.25 2.30 2.35 2.40 2.45 2.50}"
read -r -a AOA_LADDER <<< "${AOA_LADDER:-0.00 0.40 0.80 1.20 1.60 2.00 2.40 2.80 3.20 3.60 4.00}"

# ── Build the union, de-duplicated and in a STABLE order ────────────────────
# Stability matters: the array index must mean the same (AoA, Mach) pair on a
# resubmission, or --array=<subset> would target different work than intended.
PAIRS=()
seen=" "
add_pair() {                     # $1 = AoA tag, $2 = Mach tag
    local key="$1_$2"
    case "$seen" in *" $key "*) return;; esac
    seen="$seen$key "
    PAIRS+=("$key")
}
for a in "${MACH_AXIS_AOAS[@]}"; do
    for m in "${MACHS[@]}"; do add_pair "$(printf '%.2f' "$a")" "$(printf '%.2f' "$m")"; done
done
for a in "${AOA_LADDER[@]}"; do
    add_pair "$(printf '%.2f' "$a")" "$FIXED_MACH"
done

N_PAIRS=${#PAIRS[@]}
ID=${SLURM_ARRAY_TASK_ID:-0}
if [[ "$ID" -ge "$N_PAIRS" ]]; then
    echo "task $ID >= ${N_PAIRS} pairs — nothing to do"; exit 0
fi
PAIR=${PAIRS[$ID]}
R_AOA=${PAIR%_*}
R_MACH=${PAIR#*_}

echo "#############################################"
echo "Array ${SLURM_ARRAY_JOB_ID}/${ID}   pair ${ID}/${N_PAIRS}"
echo "Mesh:  ${MESH}   AoA ${R_AOA} deg   Mach ${R_MACH}"
nvidia-smi --query-gpu=index,name --format=csv,noheader
echo "#############################################"

# ── Geometry guard ──────────────────────────────────────────────────────────
# The fine meshes shipped before 2026-08 were the alpha=10 deg wedge while h0.025
# is alpha=5 deg; a study spanning both would compare two different airfoils.
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

# ── Stale-bundle check ──────────────────────────────────────────────────────
# A bundle for this (AoA, Mach) may exist but have been computed on a DIFFERENT
# mesh at the same h_tag — exactly what happened when the 10 deg meshes were
# replaced.  SB then loads the new mesh against old fields and dies with
# "length of facecolors must match the number of triangles".  Compare the cell
# count recorded in the bundle against the mesh; quarantine, never delete.
uv run python - "$H_TAG" "$MESH" "$R_AOA" "$R_MACH" <<'PY'
import sys, os, glob, shutil, time
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "Euler")
from Euler.jax_fvm.src.mesh import Mesh
h_tag, mesh_path, aoa, mach = sys.argv[1:5]
m = Mesh(); m.load_mesh(mesh_path)
n_mesh = len(m.tris)
res_dir = os.path.join("results", "diamond", h_tag)
stale_dir = os.path.join(res_dir, f"stale_{time.strftime('%Y%m%d')}")
for f in sorted(glob.glob(os.path.join(res_dir, f"AOA{aoa}_M{mach}_*.npz"))):
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

if compgen -G "results/diamond/${H_TAG}/AOA${R_AOA}_M${R_MACH}_*.npz" > /dev/null; then
    echo "  [skip] AOA${R_AOA} M${R_MACH} already present"
    exit 0
fi

# --stationarity-threshold 0 runs every case to tf.  Euler/main.py normalises the
# stationarity residual by STEP COUNT, not elapsed time, so the criterion is
# (physical rate)*dt and dt comes from the CFL — a finer mesh reports a smaller
# residual for the same physical state and stops EARLIER in physical time.  For a
# study comparing snapshots across meshes and parameters that is fatal.
uv run python Euler/main.py \
    --case diamond --mach "${R_MACH}" --aoa "${R_AOA}" \
    --mesh-path "${MESH}" \
    --flux HLLC --time-scheme SRK2 --reconstruction MUSCL \
    --stationarity-threshold 0

echo "######## AOA${R_AOA} M${R_MACH} (${H_TAG}) done ########"
