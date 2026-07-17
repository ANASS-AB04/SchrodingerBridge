#!/usr/bin/env bash
# Submit the 4 diamond SB runs as separate SLURM jobs.
#   1. no drift, no annealing, hessian_mode = det
#   2. no drift, no annealing, hessian_mode = eig
#   3. oblique drift,          hessian_mode = det
#   4. oblique drift,          hessian_mode = eig
#
# Each variant gets its own Config copy so nothing is overwritten, and the
# output-dir logic (drift_<...>/gmin<γ>) keeps every run's outputs separate.
#
# Usage:  bash submit_sb_sweep.sh
set -euo pipefail

BASE_CFG="SB/Config.toml"
GEN_DIR="SB/_sweep_configs"
mkdir -p "$GEN_DIR" logs

# No annealing anywhere (gamma_sb_schedule = []), so each run uses its single
# gamma_sb.  No-drift baselines at γ=0.005 (heat kernel converges there);
# oblique-drift runs at γ=0.001 (the M-matrix kernel handles the small γ).
#
# variant name | reference_drift | hessian_mode | gamma_sb
VARIANTS=(
  "nodrift_det|null|det|0.005"
  "nodrift_eig|null|eig|0.005"
  "drift_det|oblique|det|0.0001"
  "drift_eig|oblique|eig|0.0001"
  "drift_eig|ffd|eig|0.003"
)

for v in "${VARIANTS[@]}"; do
  IFS='|' read -r name drift hmode gamma <<< "$v"
  cfg="$GEN_DIR/Config_${name}.toml"

  # Copy base config, then override the four keys for this variant.
  # gamma_sb_schedule = [] → no annealing → the single gamma_sb below is used.
  sed -E \
    -e "s|^reference_drift *=.*|reference_drift = \"${drift}\"|" \
    -e "s|^hessian_mode *=.*|hessian_mode = \"${hmode}\"|" \
    -e "s|^gamma_sb_schedule *=.*|gamma_sb_schedule = []|" \
    -e "s|^gamma_sb *=.*|gamma_sb   = ${gamma}|" \
    "$BASE_CFG" > "$cfg"

  # SLURM cannot create the --output directory itself; make it before submit.
  mkdir -p "logs/sb-${name}"
  echo "Submitting ${name}: drift=${drift} hessian=${hmode} gamma_sb=${gamma}"
  sbatch --job-name="sb-${name}" \
         --export=ALL,SB_CONFIG="$cfg" \
         run_sb.sbatch
done

echo "Submitted ${#VARIANTS[@]} jobs.  Watch with:  squeue -u \$USER"
