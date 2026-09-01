#!/usr/bin/env bash
#SBATCH --job-name=mesh-study
#SBATCH --partition=gpu-h100,gpu-l40,gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mesh-study_%j.out
#SBATCH --error=logs/mesh-study_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Diamond mesh-resolution study — ONE job, multi-GPU.
#
#   sbatch run_mesh_study.sbatch                    # h0.025 + h0.0175 + h0.0125, ffd
#   H_TAGS=h0.0125 sbatch run_mesh_study.sbatch     # one mesh only
#   DRIFTS="ffd SBsquared" sbatch run_mesh_study.sbatch    # both drifts, head-to-head
#   AOA=2 sbatch run_mesh_study.sbatch              # 2 deg incidence: lift becomes
#                                                   # a real signal, see AoA note below
#
# Pass H_TAGS/DRIFTS/STAGES as shell exports rather than --export=ALL,VAR=...:
# their values contain spaces, which sbatch's comma-separated --export mangles.
#
# Per mesh, per drift:  11 Euler snapshots (M2.00…M2.50)  →  SB config  →  3 SB runs
#                       (γ_min = 1e-2, 1e-3, 1e-4, eig marginal)
# Then:                 sweep_analysis.py over every mesh present.
# Euler snapshots already on disk are skipped, so a drift-only rerun goes almost
# straight to the SB stage.
#
# PARALLELISM: the 11 Euler cases are independent, as are the 3 SB runs, so both
# stages run as a pool of one process per GPU (pinned with CUDA_VISIBLE_DEVICES).
# The stages themselves are ordered — SB needs all 11 snapshots.  NGPU is read
# from the allocation, so changing --gres is all it takes to widen the pool.
#
# ── PARTITION CHOICE ────────────────────────────────────────────────────────
# gpu-h100 + gpu-l40 only: the two best cards, and SLURM takes whichever frees
# first.  Inventory (sinfo):
#   gpu-h100     4 nodes  gpu:2  384 GB   ~34 TF FP64 (1/2 rate) — best for SB
#   gpu-l40      3 nodes  gpu:4  384 GB   ~1.4 TF FP64, but strong FP32 for Euler
#   gpu-a40 / gpu-rtx6000                 weaker FP64; add them if the wait bites
#
# TRADE-OFF: these two are the most contended on the cluster (both were 100%
# allocated with a ~40-deep queue when last checked, and --test-only priced the
# H100 wait at ~46 h).  Widen the list if you would rather start now:
#   sbatch --partition=gpu-h100,gpu-l40,gpu-a40,gpu-rtx6000 run_mesh_study.sbatch
# An RTX6000 measured ~3x slower than H100 on this workload — much less than the
# FP64 peak gap suggests, because these FVM kernels are bandwidth-bound.
#
# --gres=gpu:2 is capped by the SMALLEST GPU count in the list (h100 has 2).
#
# ── MEMORY: ask for what you use, not what fits ─────────────────────────────
# Measured host RSS at the LARGEST mesh here (h0.0125, 80889 cells): 514 MB for
# mesh+Delaunay+interpolators, plus the JAX/CUDA runtime (~1-1.5 GB/process),
# times NGPU processes ⇒ ~4-6 GB.  16G leaves ~3x headroom, and a small request
# is what gets the job scheduled onto partially-occupied nodes instead of
# queueing for a fully idle one.  Confirm on a finished job with:
#   sacct -j <jobid> --format=JobID,MaxRSS,MaxVMSize,ReqMem,Elapsed,State
# RAISE IT if you extend the ladder: at h0.00875/h0.00625 (375k/735k cells) the
# Delaunay alone reaches ~0.5 GB and the 11-frame vmap ~2.5 GB transient.
# Under-requesting is OOM-killed, not swapped.
#
# So if the Euler stage dominates the wall time, split the stages — L40 for the
# float32 solver, H100 for the float64 bridge:
#   STAGES=euler          sbatch --partition=gpu-l40  --gres=gpu:4 run_mesh_study.sbatch
#   STAGES="sb analysis"  sbatch --partition=gpu-h100 --gres=gpu:2 run_mesh_study.sbatch
# (STAGES defaults to "euler sb analysis" — the whole pipeline in one go.)
#
# Concurrent processes share one JAX compilation cache on purpose: every Mach has
# identical array shapes, so only the first pays compilation.
#
# RESUMABLE: finished Euler bundles and SB metrics.json are skipped, so if this
# hits the walltime just resubmit and it continues.
# ─────────────────────────────────────────────────────────────────────────────

set -uo pipefail
# Force the C locale: bash printf honours LC_NUMERIC, so under fr_FR a
# "%.2f" of 2 yields "2,00" — which would silently corrupt every bundle glob
# and every output path built from it.
export LC_ALL=C
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

# h0.025 is the ladder ANCHOR — leaving it out of the default silently produced a
# 2-point err_vs_h and made SBsquared skip the one mesh with a full legacy baseline.
H_TAGS=${H_TAGS:-"h0.025 h0.0175 h0.0125"}

# ── Angle of attack ─────────────────────────────────────────────────────────
# AOA=0 is the symmetric case: C_L is then pure mesh-asymmetry noise (the
# reference C_L is ~1e-4 and even FLIPS SIGN between meshes), so lift cannot
# discriminate between interpolators there.  At AOA=2 the lift is ~1000x that
# floor and becomes the sharper of the two force coefficients.
# Bundles are named AOA<tag>_M<mach>_..., and SB/main.py gives a non-zero AoA its
# own output subtree, so AoA cases never overwrite each other.
AOA=${AOA:-0}
AOA_TAG=$(LC_ALL=C printf "%.2f" "$AOA")

# ── Interpolation axis ──────────────────────────────────────────────────────
#   AXIS=mach  (default)  AoA fixed at $AOA, Mach sweeps 2.00 → 2.50
#   AXIS=aoa              Mach fixed at $FIXED_MACH, AoA sweeps 0 → 4 deg
# The AoA axis is the one on which LIFT carries real signal: at AoA=0 the
# reference C_L is mesh-asymmetry noise (~1e-4, sign-flipping between meshes),
# whereas across 0→4 deg C_L spans ~0 → 0.16.
AXIS=${AXIS:-mach}
FIXED_MACH=${FIXED_MACH:-2.00}
FIXED_MACH_TAG=$(LC_ALL=C printf "%.2f" "$FIXED_MACH")
read -r -a AOA_LADDER <<< "${AOA_LADDER:-0.00 0.40 0.80 1.20 1.60 2.00 2.40 2.80 3.20 3.60 4.00}"
STAGES=${STAGES:-"euler sb analysis"}
MACHS=(2.00 2.05 2.10 2.15 2.20 2.25 2.30 2.35 2.40 2.45 2.50)

# ── Stationarity: run every case to tf instead of stopping early ─────────────
# Euler/main.py:83 normalises the stationarity residual by STEP COUNT, not by
# elapsed physical time:  rel = ‖ΔW‖/(block_steps·‖W_ref‖) = (physical rate)·dt.
# dt comes from the CFL condition, so dt ∝ h and a finer mesh reports a SMALLER
# residual for the same physical state — the fixed threshold silently gets more
# lenient as you refine (h0.025 stopped 1 of 11 runs early, h0.0175 stopped 4,
# two of them before t=2).  For a mesh-convergence study that is fatal: the
# reference snapshots would be converged to different degrees at different h,
# so err_vs_h would measure reference quality rather than discretisation.
# FULL_TIME=1 passes --stationarity-threshold 0 so nothing stops early; every
# snapshot is at the same physical time on every mesh.  It costs little — the
# full t=4 runs took 26-30 s at h0.0175.
FULL_TIME=${FULL_TIME:-1}
# tf has no CLI flag — it is read from Euler/config.toml.  TF here is only used
# by the stale-bundle check below to recognise an early-stopped snapshot, so it
# must mirror that file's value.
TF=${TF:-$(grep -E '^\s*tf\s*=' Euler/config.toml | head -1 | sed 's/.*=\s*//' | tr -d ' ')}
TF=${TF:-4.0}
if [[ "$FULL_TIME" == "1" ]]; then
    STAT_ARGS=(--stationarity-threshold 0)
else
    STAT_ARGS=()
fi

# γ ladder; each SB run anneals down the PREFIX ending at its γ_min, so all runs
# share the same warm-start path as the existing h0.025 sweep.
LADDER=(0.5 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.0005 0.0002 0.0001)
# Every rung by default, not just 3.  The resume-skip means already-finished
# points cost nothing, so a rerun fills in the gaps from earlier submissions.
#   GIDXS="5 8 11" sbatch ...     # the old focused subset
read -r -a GIDXS <<< "${GIDXS:-0 1 2 3 4 5 6 7 8 9 10 11}"

# Which reference drift(s) to run per mesh.  "SBsquared" is the bootstrapped
# self-conditioned drift — it needs no external β, and because its γ ladder IS
# the bootstrap it must keep the full prefix schedule (which GIDXS already gives).
#   DRIFTS="ffd SBsquared" sbatch ...        # head-to-head in one job
read -r -a DRIFTS <<< "${DRIFTS:-ffd}"

GEN_DIR="${SLURM_TMPDIR:-/tmp}/mesh_study_cfgs_${SLURM_JOB_ID}"
mkdir -p "$GEN_DIR"

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

echo "#############################################"
echo "Job ${SLURM_JOB_ID}   partition ${SLURM_JOB_PARTITION:-?}"
echo "Meshes:  ${H_TAGS}"
echo "Axis:    ${AXIS}   (AoA ${AOA_TAG} deg / fixed M ${FIXED_MACH_TAG})"
echo "Gammas:  ${GIDXS[*]}"
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

    # ── 0. Quarantine stale bundles ─────────────────────────────────────────
    # The resume-skip below trusts "a bundle for this Mach exists".  That is not
    # enough: results/diamond/<h_tag>/ can hold snapshots computed on a DIFFERENT
    # mesh at the same h_tag — which is exactly what happened when the α=10°
    # meshes were replaced by α=5° ones.  SB then loads the new mesh against old
    # fields and dies with "length of facecolors must match the number of
    # triangles".  Compare the cell count recorded in each bundle against the
    # mesh and move any mismatch aside so it gets regenerated.
    # With FULL_TIME=1 a bundle stopped before tf is stale too (see below).
    uv run python - "$H_TAG" "$MESH" "${FULL_TIME}" "${TF}" <<'PY'
import sys, os, glob, shutil, time
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "Euler")
from Euler.jax_fvm.src.mesh import Mesh

h_tag, mesh_path, full_time, tf = sys.argv[1], sys.argv[2], sys.argv[3] == "1", float(sys.argv[4])
m = Mesh(); m.load_mesh(mesh_path)
n_mesh = len(m.tris)

res_dir = os.path.join("results", "diamond", h_tag)
stale_dir = os.path.join(res_dir, f"stale_{time.strftime('%Y%m%d-%H%M%S')}")
moved = 0
for f in sorted(glob.glob(os.path.join(res_dir, "*.npz"))):
    try:
        d = np.load(f, allow_pickle=True)
        n = int(np.asarray(d["n_cells"]).ravel()[0])
        t = float(np.asarray(d["time"]).ravel()[0])
        d.close()
    except Exception as exc:
        n, t, why = -1, -1.0, f"unreadable ({exc})"
    else:
        why = None
        if n != n_mesh:
            why = f"n_cells={n} != mesh {n_mesh} (different mesh!)"
        elif full_time and t < 0.999 * tf:
            why = f"stopped early at t={t:.3f} < tf={tf}"
    if why:
        os.makedirs(stale_dir, exist_ok=True)
        shutil.move(f, os.path.join(stale_dir, os.path.basename(f)))
        print(f"  [stale] {os.path.basename(f)}: {why}")
        moved += 1
print(f"  bundle check: {moved} stale moved to {stale_dir if moved else '(none)'}, "
      f"{len(glob.glob(os.path.join(res_dir, '*.npz')))} kept")
PY

    # ── 1. Euler snapshots — pool of NGPU concurrent runs ────────────────────
    if has_stage euler; then
        echo; echo "===== [$H_TAG] STAGE 1: Euler snapshots (${NGPU}-way) ====="
        SLOT=0
        # One (aoa, mach) pair per ladder rung; which one varies is set by AXIS.
        if [[ "$AXIS" == "aoa" ]]; then
            RUNG=("${AOA_LADDER[@]}")
        else
            RUNG=("${MACHS[@]}")
        fi
        for R in "${RUNG[@]}"; do
            if [[ "$AXIS" == "aoa" ]]; then
                R_AOA=$R;   R_MACH=$FIXED_MACH_TAG
            else
                R_AOA=$AOA; R_MACH=$R
            fi
            R_AOA_TAG=$(LC_ALL=C printf "%.2f" "$R_AOA")
            if compgen -G "results/diamond/${H_TAG}/AOA${R_AOA_TAG}_M${R_MACH}_*.npz" > /dev/null; then
                echo "  [skip] AOA${R_AOA_TAG} M${R_MACH} already present"
                continue
            fi
            throttle
            GPU=$(( SLOT % NGPU ));  SLOT=$(( SLOT + 1 ))
            LOG="logs/euler_${SLURM_JOB_ID}_${H_TAG}_AOA${R_AOA_TAG}_M${R_MACH}.log"
            echo "  [gpu ${GPU}] AOA${R_AOA_TAG} M${R_MACH}  → ${LOG}"
            CUDA_VISIBLE_DEVICES=$GPU uv run python Euler/main.py \
                --case diamond --mach "${R_MACH}" --aoa "${R_AOA}" \
                --mesh-path "${MESH}" \
                --flux HLLC --time-scheme SRK2 --reconstruction MUSCL \
                ${STAT_ARGS[@]+"${STAT_ARGS[@]}"} \
                > "$LOG" 2>&1 &
        done
        wait
        echo "  Euler stage done for ${H_TAG}"
    fi

    has_stage sb || continue

    if [[ "$AXIS" == "aoa" ]]; then
        N_BUNDLES=$(compgen -G "results/diamond/${H_TAG}/AOA*_M${FIXED_MACH_TAG}_*.npz" | wc -l)
    else
        N_BUNDLES=$(compgen -G "results/diamond/${H_TAG}/AOA${AOA_TAG}_M*.npz" | wc -l)
    fi
    if [[ "$N_BUNDLES" -lt 11 ]]; then
        echo "[FAIL] $H_TAG has only ${N_BUNDLES}/11 bundles — skipping SB (see logs/euler_${SLURM_JOB_ID}_${H_TAG}_*.log)"
        continue
    fi

    # ── 2. SB config derived from whatever bundles Euler actually wrote ──────
    # (globbed, because the stationarity check stops different Machs at
    #  different final times, so the filenames are not predictable)
    if [[ "$AXIS" == "aoa" ]]; then
        AXIS_TAG="axAOA_M${FIXED_MACH_TAG}"
        CFG_ARGS=(--axis aoa --mach "${FIXED_MACH_TAG}")
    else
        AXIS_TAG="aoa${AOA_TAG}"
        CFG_ARGS=(--axis mach --aoa "${AOA_TAG}")
    fi
    BASE_CFG="$GEN_DIR/Config_${H_TAG}_${AXIS_TAG}.toml"
    uv run python SB/make_config_for_mesh.py --h-tag "${H_TAG}" "${CFG_ARGS[@]}" --out "$BASE_CFG" \
        || { echo "[FAIL] config generation for $H_TAG"; continue; }

    # ── 3. SB runs — pool of NGPU concurrent runs ────────────────────────────
    echo; echo "===== [$H_TAG] STAGE 2: SB runs (${NGPU}-way) ====="
    SLOT=0
    for DRIFT in "${DRIFTS[@]}"; do
    for GIDX in "${GIDXS[@]}"; do
        GMIN=${LADDER[$GIDX]}
        SCHED="["
        for ((i = 0; i <= GIDX; i++)); do
            SCHED+="${LADDER[$i]}"; (( i < GIDX )) && SCHED+=", "
        done
        SCHED+="]"

        # Mirror SB/main.py::_aoa_tag exactly:
        #   Mach axis, AoA 0   →  (no level)      keeps the finished tree in place
        #   Mach axis, AoA a   →  aoaA/
        #   AoA  axis, 0 → 4   →  aoa0.00-4.00/   a different experiment entirely
        if [[ "$AXIS" == "aoa" ]]; then
            AOA_LVL="aoa${AOA_LADDER[0]}-${AOA_LADDER[${#AOA_LADDER[@]}-1]}/"
        elif [[ "$AOA_TAG" != "0.00" ]]; then
            AOA_LVL="aoa${AOA_TAG}/"
        else
            AOA_LVL=""
        fi
        OUT="outputs/Mach_interpolation/diamond/iso/hessian/eig/${H_TAG}/${AOA_LVL}drift_${DRIFT}/gmin${GMIN}"
        if [[ -f "${OUT}/metrics.json" ]]; then
            echo "  [skip] ${DRIFT} gmin=${GMIN} already done"
            continue
        fi

        CFG="$GEN_DIR/Config_${H_TAG}_${AXIS_TAG}_${DRIFT}_gmin${GMIN}.toml"
        sed -e "s|^reference_drift = .*|reference_drift = \"${DRIFT}\"|" \
            -e "s|^gamma_sb_schedule = .*|gamma_sb_schedule = ${SCHED}|" \
            -e "s|^gamma_sb *= .*|gamma_sb = ${GMIN}|" \
            -e "s|^density_mode *= .*|density_mode = \"iso_hessian\"|" \
            -e "s|^hessian_mode *= .*|hessian_mode = \"eig\"|" \
            -e "s|^compute_w2 *= .*|compute_w2 = true|" \
            "$BASE_CFG" > "$CFG"

        throttle
        GPU=$(( SLOT % NGPU ));  SLOT=$(( SLOT + 1 ))
        LOG="logs/sb_${SLURM_JOB_ID}_${H_TAG}_${AXIS_TAG}_${DRIFT}_gmin${GMIN}.log"
        echo "  [gpu ${GPU}] ${DRIFT} gmin=${GMIN}  schedule=${SCHED}  → ${LOG}"
        CUDA_VISIBLE_DEVICES=$GPU SB_CONFIG="$CFG" uv run python SB/main.py \
            > "$LOG" 2>&1 &
    done
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
