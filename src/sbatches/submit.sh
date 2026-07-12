#!/bin/bash
set -euo pipefail

# Submit simulation Slurm arrays (ideal + shot-noise) for selected architectures.
#
# Usage:
#   bash src/sbatches/submit.sh                       # all three architectures
#   bash src/sbatches/submit.sh baseline
#   bash src/sbatches/submit.sh qcnn resqnet
#
# Environment variables (passed through to the sbatch scripts):
#   OPTIMIZERS       e.g. "nft" or "cobyla,qnspsa,nft" (default in sbatch)
#   QUBITS_OVERRIDE  e.g. "4,8,12,16" (applies to every architecture submitted;
#                    do NOT use with qcnn/resqnet unless all sizes are powers of 2)
#   TRACK_PARAMS     "1" to store the parameter trajectory
#   ENABLE_REAL_HW   "1" to also submit the real-hardware array

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if (( $# == 0 )); then
  ARCHITECTURES=("baseline" "qcnn" "resqnet")
else
  ARCHITECTURES=("$@")
fi

ENABLE_REAL_HW="${ENABLE_REAL_HW:-0}"

for ARCH in "${ARCHITECTURES[@]}"; do
  case "$ARCH" in
    baseline|qcnn|resqnet) ;;
    *)
      echo "Unsupported architecture: $ARCH"
      echo "Allowed values: baseline qcnn resqnet"
      exit 1
      ;;
  esac

  echo "== Submitting simulation arrays for architecture: $ARCH =="
  ARCHITECTURE="$ARCH" sbatch src/sbatches/exp-ideal.sbatch
  ARCHITECTURE="$ARCH" sbatch src/sbatches/exp-shot-noise.sbatch
  if [[ "$ENABLE_REAL_HW" == "1" ]]; then
    ARCHITECTURE="$ARCH" sbatch src/sbatches/exp-real.sbatch
  else
    echo "  -> skip real_hw (disabled, set ENABLE_REAL_HW=1 to enable)"
  fi
done

echo "Done. Check queue with: squeue -u $USER"
