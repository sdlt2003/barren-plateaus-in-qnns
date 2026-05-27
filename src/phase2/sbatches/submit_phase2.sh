#!/bin/bash
set -euo pipefail

# Submit Phase 2 Slurm arrays for selected architectures.
#
# Usage:
#   bash src/phase2/sbatches/submit_phase2.sh
#   bash src/phase2/sbatches/submit_phase2.sh baseline_hea
#   bash src/phase2/sbatches/submit_phase2.sh qcnn
#   bash src/phase2/sbatches/submit_phase2.sh baseline_hea qcnn

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if (( $# == 0 )); then
  ARCHITECTURES=("baseline_hea" "qcnn")
else
  ARCHITECTURES=("$@")
fi

for ARCH in "${ARCHITECTURES[@]}"; do
  case "$ARCH" in
    baseline_hea|qcnn) ;;
    *)
      echo "Unsupported architecture: $ARCH"
      echo "Allowed values: baseline_hea qcnn"
      exit 1
      ;;
  esac

  echo "== Submitting Phase 2 for architecture: $ARCH =="
  ARCHITECTURE="$ARCH" sbatch src/phase2/sbatches/exp-ideal.sbatch
  ARCHITECTURE="$ARCH" sbatch src/phase2/sbatches/exp-shot-noise.sbatch
  ARCHITECTURE="$ARCH" sbatch src/phase2/sbatches/exp-real.sbatch
done

echo "Done. Check queue with: squeue -u $USER"
