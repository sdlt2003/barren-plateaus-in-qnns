#!/bin/bash
set -euo pipefail

# Submit Phase 2 Slurm arrays for selected architectures.
#
# Usage:
#   bash src/phase2/sbatches/submit_phase2.sh
#   bash src/phase2/sbatches/submit_phase2.sh baseline_hea
#   bash src/phase2/sbatches/submit_phase2.sh qcnn
#   bash src/phase2/sbatches/submit_phase2.sh resqnet
#   bash src/phase2/sbatches/submit_phase2.sh baseline_hea qcnn resqnet

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if (( $# == 0 )); then
  ARCHITECTURES=("baseline_hea" "qcnn" "resqnet")
else
  ARCHITECTURES=("$@")
fi

AUTO_BACKUP_ON_DONE="${AUTO_BACKUP_ON_DONE:-1}"
ENABLE_REAL_HW="${ENABLE_REAL_HW:-0}"

submit_with_finalize() {
  local arch="$1"
  local scenario="$2"
  local sbatch_file="$3"
  local array_job_id

  array_job_id="$(
    ARCHITECTURE="$arch" sbatch "$sbatch_file" | awk '{print $4}'
  )"
  echo "  -> submitted ${scenario} array: ${array_job_id}"

  if [[ "$AUTO_BACKUP_ON_DONE" != "1" ]]; then
    return 0
  fi

  local finalize_job_id
  finalize_job_id="$(
    sbatch \
      --dependency="afterany:${array_job_id}" \
      --job-name="p2_done_${scenario}_${arch}" \
      --output="outputs/logs/p2_done_${scenario}_${arch}_%j.out" \
      --error="outputs/logs/p2_done_${scenario}_${arch}_%j.err" \
      --wrap "cd '$REPO_ROOT' && bash src/phase2/sbatches/finalize_run_backup.sh '${array_job_id}' '${scenario}' '${arch}'" \
      | awk '{print $4}'
  )"
  echo "  -> finalize/backup job: ${finalize_job_id} (afterany:${array_job_id})"
}

for ARCH in "${ARCHITECTURES[@]}"; do
  case "$ARCH" in
    baseline_hea|qcnn|resqnet) ;;
    *)
      echo "Unsupported architecture: $ARCH"
      echo "Allowed values: baseline_hea qcnn resqnet"
      exit 1
      ;;
  esac

  echo "== Submitting Phase 2 for architecture: $ARCH =="
  submit_with_finalize "$ARCH" "ideal" "src/phase2/sbatches/exp-ideal.sbatch"
  submit_with_finalize "$ARCH" "shot_noise" "src/phase2/sbatches/exp-shot-noise.sbatch"
  if [[ "$ENABLE_REAL_HW" == "1" ]]; then
    submit_with_finalize "$ARCH" "real_hw" "src/phase2/sbatches/exp-real.sbatch"
  else
    echo "  -> skip real_hw (disabled, set ENABLE_REAL_HW=1 to enable)"
  fi
done

echo "Done. Check queue with: squeue -u $USER"
if [[ "$AUTO_BACKUP_ON_DONE" == "1" ]]; then
  echo "Auto-backup-on-done is enabled (AUTO_BACKUP_ON_DONE=1)."
fi
if [[ "$ENABLE_REAL_HW" != "1" ]]; then
  echo "Real-hardware submission is disabled by default (ENABLE_REAL_HW=0)."
fi
