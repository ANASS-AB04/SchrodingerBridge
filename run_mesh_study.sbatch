#!/usr/bin/env bash
#SBATCH --job-name=mesh-study
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mesh-study_%j.out
#SBATCH --error=logs/mesh-study_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Diamond mesh-resolution study — ONE job, multi-GPU.
#
#   sbatch run_mesh_study.sbatch                               # h0.0175 + h0.0125
#   sbatch --export=ALL,H_TAGS=h0.0125 run_mesh_study.sbatch   # one mesh only
#
# Per mesh:  11 Euler snapshots (M2.00…M2.50)  →  SB config  →  3 SB runs
#            (γ_min = 1e-2, 1e-3, 1e-4, eig marginal, ffd drift)
# Then:      sweep_analysis.py over every mesh, including the existing h0.025.
#
# PARALLELISM: the 11 Euler cases are independent, as are the 3 SB runs, so both
# stages run as a pool of one process per GPU (pinned with CUDA_VISIBLE_DEVICES).
# The stages themselves are ordered — SB needs all 11 snapshots.  NGPU is read
# from the allocation, so changing --gres is all it takes to widen the pool.
#
# ── PARTITION CHOICE ────────────────────────────────────────────────────────
#   gpu-h100   2 GPU/node   ← default: SB forces float64 (SB/main.py sets
#                             JAX_ENABLE_X64) and the L40 runs FP64 at 1/64 rate
#                             (~1.4 TFLOPS vs ~34 on H100), plus 2-4× bandwidth.
#                             Worth far more than the 2× GPU-count deficit.
#   gpu-l40    4 GPU/node   ← the Euler stage is float32, where the L40 is fine
#                             and the extra 2 GPUs make it ~2× faster.
#
# So if the Euler stage turns out to dominate the wall time, split the stages —
# 4×L40 for the float32 solver, then 2×H100 for the float64 bridge:
#   sbatch --partition=gpu-l40  --gres=gpu:4 --export=ALL,STAGES=euler run_mesh_study.sbatch
#   sbatch --partition=gpu-h100 --gres=gpu:2 --export=ALL,STAGES="sb analysis" run_mesh_study.sbatch
# (STAGES defaults to "euler sb analysis" — the whole pipeline in one go.)
#
# Concurrent processes share one JAX compilation cache on purpose: every Mach has
# identical array shapes, so only the first pays compilation.
#
# RESUMABLE: finished Euler bundles and SB metrics.json are skipped, so if this
# hits the walltime just resubmit and it continues.
# ─────────────────────────────────────────────────────────────────────────────

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH=""

export MPLBACKEND=Agg
export JAX_ENABLE_X64=true
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false     # required: several procs per node

# Shared compilation cache — identical shapes across Machs, so wave 2+ skip it.
JAX_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/jax-cache-${SLURM_JOB_ID}"
rm -rf "$JAX_CACHE_ROOT"; mkdir -p "$JAX_CACHE_ROOT"
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_ROOT"

# ── Resources ────────────────────────────────────────────────────────────────
NGPU=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
[[ "$NGPU" -lt 1 ]] && NGPU=1
# Split the CPU allocation across the concurrent processes, min 1 thread each.
NTHREAD=$(( ${SLURM_CPUS_PER_TASK:-8} / NGPU ));  [[ "$NTHREAD" -lt 1 ]] && NTHREAD=1
export OMP_NUM_THREADS=$NTHREAD
export MKL_NUM_THREADS=$NTHREAD

H_TAGS=${H_TAGS:-"h0.0175 h0.0125"}
STAGES=${STAGES:-"euler sb analysis"}
MACHS=(2.00 2.05 2.10 2.15 2.20 2.25 2.30 2.35 2.40 2.45 2.50)

# γ ladder; each SB run anneals down the PREFIX ending at its γ_min, so all runs
# share the same warm-start path as the existing h0.025 sweep.
LADDER=(0.5 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.0005 0.0002 0.0001)
GIDXS=(5 8 11)                         # → γ_min = 0.01, 0.001, 0.0001

GEN_DIR="${SLURM_TMPDIR:-/tmp}/mesh_study_cfgs_${SLURM_JOB_ID}"
mkdir -p "$GEN_DIR"

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

echo "#############################################"
echo "Job ${SLURM_JOB_ID}   partition ${SLURM_JOB_PARTITION:-?}"
echo "Meshes:  ${H_TAGS}"
echo "Stages:  ${STAGES}"
echo "GPUs:    ${NGPU}   (${NTHREAD} CPU threads per process)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "#############################################"

# Block until fewer than NGPU background jobs are running.
throttle() { while [[ $(jobs -rp | wc -l) -ge "$NGPU" ]]; do sleep 5; done; }

for H_TAG in $H_TAGS; do

    MESH="meshes/diamond/diamond_${H_TAG}.npy"

    # ── Guard: every mesh in the ladder must be the SAME airfoil ─────────────
    # The fine meshes shipped before 2026-08 were α=10° (height=0.176) while
    # h0.025 is α=5° (height=0.0875); mixing them compares two geometries.
    uv run python - "$MESH" <<'PY' || { echo "[FAIL] geometry guard, skipping $H_TAG"; continue; }
import sys
sys.path.insert(0, "."); sys.path.insert(0, "Euler")
from Euler.jax_fvm.src.mesh import Mesh
m = Mesh(); m.load_mesh(sys.argv[1])
h = m.metadata["height"]
ok = abs(h - 0.087488663525924) < 1e-12
print(f"  geometry: N={len(m.tris)}  height={h:.9f}  5deg={ok}")
raise SystemExit(0 if ok else 1)
PY

    # ── 1. Euler snapshots — pool of NGPU concurrent runs ────────────────────
    if has_stage euler; then
        echo; echo "===== [$H_TAG] STAGE 1: Euler snapshots (${NGPU}-way) ====="
        SLOT=0
        for MACH in "${MACHS[@]}"; do
            if compgen -G "results/diamond/${H_TAG}/AOA0.00_M${MACH}_*.npz" > /dev/null; then
                echo "  [skip] M${MACH} already present"
                continue
            fi
            throttle
            GPU=$(( SLOT % NGPU ));  SLOT=$(( SLOT + 1 ))
            LOG="logs/euler_${SLURM_JOB_ID}_${H_TAG}_M${MACH}.log"
            echo "  [gpu ${GPU}] M${MACH}  → ${LOG}"
            CUDA_VISIBLE_DEVICES=$GPU uv run python Euler/main.py \
                --case diamond --mach "${MACH}" --aoa 0 \
                --mesh-path "${MESH}" \
                --flux HLLC --time-scheme SRK2 --reconstruction MUSCL \
                > "$LOG" 2>&1 &
        done
        wait
        echo "  Euler stage done for ${H_TAG}"
    fi

    has_stage sb || continue

    N_BUNDLES=$(compgen -G "results/diamond/${H_TAG}/AOA0.00_M*.npz" | wc -l)
    if [[ "$N_BUNDLES" -lt 11 ]]; then
        echo "[FAIL] $H_TAG has only ${N_BUNDLES}/11 bundles — skipping SB (see logs/euler_${SLURM_JOB_ID}_${H_TAG}_*.log)"
        continue
    fi

    # ── 2. SB config derived from whatever bundles Euler actually wrote ──────
    # (globbed, because the stationarity check stops different Machs at
    #  different final times, so the filenames are not predictable)
    BASE_CFG="$GEN_DIR/Config_${H_TAG}.toml"
    uv run python SB/make_config_for_mesh.py --h-tag "${H_TAG}" --out "$BASE_CFG" \
        || { echo "[FAIL] config generation for $H_TAG"; continue; }

    # ── 3. SB runs — pool of NGPU concurrent runs ────────────────────────────
    echo; echo "===== [$H_TAG] STAGE 2: SB runs (${NGPU}-way) ====="
    SLOT=0
    for GIDX in "${GIDXS[@]}"; do
        GMIN=${LADDER[$GIDX]}
        SCHED="["
        for ((i = 0; i <= GIDX; i++)); do
            SCHED+="${LADDER[$i]}"; (( i < GIDX )) && SCHED+=", "
        done
        SCHED+="]"

        OUT="outputs/Mach_interpolation/diamond/iso/hessian/eig/${H_TAG}/drift_ffd/gmin${GMIN}"
        if [[ -f "${OUT}/metrics.json" ]]; then
            echo "  [skip] gmin=${GMIN} already done"
            continue
        fi

        CFG="$GEN_DIR/Config_${H_TAG}_gmin${GMIN}.toml"
        sed -e "s|^reference_drift = .*|reference_drift = \"ffd\"|" \
            -e "s|^gamma_sb_schedule = .*|gamma_sb_schedule = ${SCHED}|" \
            -e "s|^gamma_sb *= .*|gamma_sb = ${GMIN}|" \
            -e "s|^density_mode *= .*|density_mode = \"iso_hessian\"|" \
            -e "s|^hessian_mode *= .*|hessian_mode = \"eig\"|" \
            -e "s|^compute_w2 *= .*|compute_w2 = true|" \
            "$BASE_CFG" > "$CFG"

        throttle
        GPU=$(( SLOT % NGPU ));  SLOT=$(( SLOT + 1 ))
        LOG="logs/sb_${SLURM_JOB_ID}_${H_TAG}_gmin${GMIN}.log"
        echo "  [gpu ${GPU}] gmin=${GMIN}  schedule=${SCHED}  → ${LOG}"
        CUDA_VISIBLE_DEVICES=$GPU SB_CONFIG="$CFG" uv run python SB/main.py \
            > "$LOG" 2>&1 &
    done
    wait
    echo "  SB stage done for ${H_TAG}"
done

# ── 4. Analysis across every mesh present (incl. the existing h0.025) ────────
if has_stage analysis; then
    echo; echo "===== ANALYSIS ====="
    uv run python SB/sweep_analysis.py \
        --root outputs/Mach_interpolation/diamond/iso/hessian \
        || echo "[WARN] sweep_analysis failed"
fi

echo; echo "######## mesh study done ########"
echo "Per-run logs:  logs/{euler,sb}_${SLURM_JOB_ID}_*.log"
echo "Pull back:     rsync -avz cali3:~/GenerativePDE/{outputs,results}/ ./"
