#!/bin/bash
# Run baseline real-hardware grid locally (no Slurm).
# Uses project-relative outdirs so bnd run can write results under /work.
set -euo pipefail

cd "$(dirname "$0")/../../.."

RUN_ROOT="outputs/$(date +%Y%m%d-%H%M%S)_phase1_baseline-real-hw"
SEED="${SEED:-10}"
BACKEND="${BACKEND:-ibm_basquecountry}"
TIMEOUT="${RUNTIME_RESULT_TIMEOUT:-120}"
LOG="${RUN_ROOT}/run.log"

mkdir -p "$RUN_ROOT"

for Q in 4 8 12 16 20; do
  BUDGET=$((32 * (2 ** ((Q - 4) / 4))))
  RUN_DIR="${RUN_ROOT}/seed_${SEED}/qubits_${Q}"
  mkdir -p "$RUN_DIR"
  {
    echo ""
    echo "========== $(date -Iseconds) | seed=$SEED qubits=$Q budget=$BUDGET =========="
  } | tee -a "$LOG"
  bnd run python src/phase1/exp-real.py \
    --n_qubits "$Q" \
    --seed "$SEED" \
    --budget-evals "$BUDGET" \
    --shots 1024 \
    --window 10 \
    --tolerance 1e-3 \
    --backend "$BACKEND" \
    --runtime-result-timeout "$TIMEOUT" \
    --outdir "$RUN_DIR" 2>&1 | tee -a "$LOG"
  echo "========== $(date -Iseconds) | DONE seed=$SEED qubits=$Q ==========" | tee -a "$LOG"
done

echo "ALL COMPLETE $(date -Iseconds) | RUN_ROOT=$RUN_ROOT" | tee -a "$LOG"
