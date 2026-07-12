#!/bin/bash
set -euo pipefail

# Submit the Phase 3 gradient-metrics array and a merge job that runs after all tasks finish.
#
# Usage:
#   bash src/sbatches/submit_grad_metrics.sh
#
# Environment variables (optional overrides, same as exp-gradient-metrics.sbatch):
#   NOISE_MODES, NORM_SEEDS, NORM_SIZES, ARCHITECTURES, N_SAMPLES, ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

NOISE_MODES="${NOISE_MODES:-ideal,shot-noise}"
NORM_SEEDS="${NORM_SEEDS:-10,20,30,40,50}"
NORM_SIZES="${NORM_SIZES:-4,8,12,16}"

IFS=',' read -r -a NOISE_ARR <<< "$NOISE_MODES"
IFS=',' read -r -a SEED_ARR <<< "$NORM_SEEDS"
IFS=',' read -r -a SIZE_ARR <<< "$NORM_SIZES"
NUM_NOISE=${#NOISE_ARR[@]}
TASKS_PER_MODE=$((1 + ${#SEED_ARR[@]} * ${#SIZE_ARR[@]}))
TOTAL=$((NUM_NOISE * TASKS_PER_MODE))

echo "Submitting grad-metrics array: $TOTAL tasks ($NUM_NOISE modes x $TASKS_PER_MODE per mode)"
ARRAY_JOB=$(sbatch --parsable --array="1-${TOTAL}" src/sbatches/exp-gradient-metrics.sbatch)
echo "  array job id: $ARRAY_JOB"

MERGE_JOB=$(sbatch --parsable --dependency="afterok:${ARRAY_JOB}" \
	--export=ALL,ARRAY_JOB_ID="${ARRAY_JOB}" \
	src/sbatches/merge_grad_metrics.sbatch)
echo "  merge job id: $MERGE_JOB (runs after array completes)"

echo "Run root will be under outputs/.runmeta/${ARRAY_JOB}/run_name.txt"
echo "Check queue: squeue -u $USER"
