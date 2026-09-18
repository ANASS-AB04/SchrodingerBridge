#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Submit the FULL study (HR snapshots -> SB matrix) for one or more geometries,
# chaining them so the SB array only starts once its Euler array has finished.
#
#   ./run_all_geoms.sh                       # all five new airfoils
#   ./run_all_geoms.sh naca0012              # just one
#   ./run_all_geoms.sh --dry-run             # print what would be submitted
#   H_TAG=h0.0175 ./run_all_geoms.sh         # a finer mesh
#
# Each geometry gets, in order:
#   1. an Euler array  — meshes are generated on demand by the array itself, so
#      there is no separate meshing step; it lands on L40/RTX/H100/A40.
#   2. an SB array     — submitted with --dependency=afterany on the Euler job,
#      restricted to L40/H100 (the bridge runs are long and the rtx6000s are
#      better spent on HR generation).
#
# afterany, NOT afterok: the Euler array tolerates individual failures by design
# (one bad Mach must not kill its chunk), so it can finish "failed" while having
# produced almost everything.  The SB script skips pairs whose bundles are
# missing, so it degrades gracefully; afterok would strand the whole study on a
# single unconverged transonic solve.
#
# COST per geometry at h0.025:  ~480 Euler solves (~4 GPU-h, 61 tasks)
#                               ~432 SB runs      (~60 GPU-h, 36 tasks)
# Five airfoils is therefore ~320 GPU-h.  Submit one first and read its results
# before committing the rest.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
export LC_ALL=C
cd "$(dirname "$0")"

DRY=0
[[ "${1:-}" == "--dry-run" ]] && { DRY=1; shift; }

GEOMS=("$@")
if [[ ${#GEOMS[@]} -eq 0 ]]; then
    GEOMS=(naca0012 naca2412 rae2822 oneraD oa209)
fi

H_TAG=${H_TAG:-h0.025}
EULER_ARRAY=${EULER_ARRAY:-0-60%12}
SB_ARRAY=${SB_ARRAY:-0-35%8}

echo "geometries : ${GEOMS[*]}"
echo "mesh       : ${H_TAG}"
echo

submit() {   # echo + run (or just echo under --dry-run); prints the job id
    if [[ "$DRY" -eq 1 ]]; then echo "    would run: $*" >&2; echo "DRYRUN"; return; fi
    local out;  out=$("$@" 2>&1) || { echo "    SUBMIT FAILED: $out" >&2; echo ""; return; }
    echo "$out" | grep -oE '[0-9]+$'
}

for g in "${GEOMS[@]}"; do
    echo "── $g ──────────────────────────────────────────────"

    # 1. HR snapshots.  The array generates the mesh on first use, so nothing
    #    needs to exist beforehand except the geometry being known to
    #    Euler/generate_mesh.py.
    jid_e=$(submit sbatch --parsable --job-name="eul-$g" \
                --array="$EULER_ARRAY" \
                --export=ALL,CASE="$g",H_TAG="$H_TAG" \
                run_euler_grid_big.sbatch)
    if [[ -z "$jid_e" ]]; then echo "    [skip] Euler submission failed"; continue; fi
    echo "    Euler : job ${jid_e}   (array ${EULER_ARRAY})"

    # 2. SB matrix, held until the Euler array is done.
    jid_s=$(submit sbatch --parsable --job-name="sb-$g" \
                --array="$SB_ARRAY" \
                --dependency=afterany:"$jid_e" \
                --export=ALL,CASE="$g",H_TAG="$H_TAG" \
                run_sb_grid_big.sbatch)
    if [[ -z "$jid_s" ]]; then echo "    [skip] SB submission failed"; continue; fi
    echo "    SB    : job ${jid_s}   (after ${jid_e}, array ${SB_ARRAY})"
    echo
done

cat <<'EOF'
── after they finish ────────────────────────────────────────────
  # what actually completed, per geometry:
  for g in naca0012 naca2412 rae2822 oneraD oa209; do
      echo "$g: $(find outputs -path "*/$g/*" -name metrics.json | wc -l) SB runs, \
$(ls results/$g/h0.025/*.npz 2>/dev/null | wc -l) snapshots"
  done

  # analysis (per geometry — the case is part of the root path, so they never pool):
  for g in naca0012 naca2412 rae2822 oneraD oa209; do
      uv run python SB/sweep_analysis.py   --root outputs/Mach_interpolation/$g/iso/hessian
      uv run python SB/interval_analysis.py --root outputs/Mach_interpolation/$g/iso/hessian
  done
EOF
