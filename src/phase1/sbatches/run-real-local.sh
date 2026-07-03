#!/bin/bash
# Run baseline real-hardware grid locally (no Slurm).
# Uses project-relative outdirs so bnd run can write results under /work.
set -euo pipefail

cd "$(dirname "$0")/../../.."

RUN_ROOT="outputs/$(date +%Y%m%d-%H%M%S)_phase1_real-hw_baseline"
SEED="${SEED:-10}"
BACKEND="${BACKEND:-ibm_basquecountry}"
TIMEOUT="${RUNTIME_RESULT_TIMEOUT:-600}"
LOG="${RUN_ROOT}/run.log"

mkdir -p "$RUN_ROOT"

mkdir -p "${RUN_ROOT}/seed_${SEED}"
{
  echo ""
  echo "========== $(date -Iseconds) | seed=$SEED qubits=4,8,12,16,20 (one session) =========="
} | tee -a "$LOG"
bnd run python src/phase2/exp-real.py \
  --architecture baseline_hea \
  --seed "$SEED" \
  --run-root "$RUN_ROOT" \
  --qubit-sizes "4,8,12,16,20" \
  --shots 1024 \
  --window 10 \
  --tolerance 1e-3 \
  --backend "$BACKEND" \
  --runtime-result-timeout "$TIMEOUT" 2>&1 | tee -a "$LOG"
echo "========== $(date -Iseconds) | DONE seed=$SEED ==========" | tee -a "$LOG"

echo "ALL COMPLETE $(date -Iseconds) | RUN_ROOT=$RUN_ROOT" | tee -a "$LOG"
