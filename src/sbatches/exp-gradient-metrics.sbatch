#!/bin/bash
#SBATCH --job-name=barren_grad_metrics_p3
#SBATCH --output=outputs/logs/p3_grad_%A_%a.out
#SBATCH --error=outputs/logs/p3_grad_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=CPU
#SBATCH --time=1-00:00:00
#SBATCH --array=1-42

set -euo pipefail
cd /home/sdelostoyos/pfs/projects/barren-plateaus-qnns

# Gradient diagnostics (barren-plateau certification), parallelized as a Slurm array.
# Each array task runs ONE slice:
#   - 2 variance tasks (ideal + shot-noise): Var(grad) vs qubits, all architectures.
#   - 40 norm-decay tasks (2 modes x 5 seeds x 4 sizes): one plot per (seed, qubits).
#
# After all tasks finish, run merge (automatic if submitted via submit_grad_metrics.sh):
#   bnd run --cpu python src/experiments/exp-gradient-metrics.py --aggregate-only --outdir <run_root>

ARCHITECTURES="${ARCHITECTURES:-baseline,qcnn,resqnet}"
QUBIT_SIZES="${QUBIT_SIZES:-4,8,12,16}"
N_SAMPLES="${N_SAMPLES:-300}"
INIT_DIST="${INIT_DIST:-uniform}"
NORM_SIZES="${NORM_SIZES:-4,8,12,16}"
NORM_SEEDS="${NORM_SEEDS:-10,20,30,40,50}"
NORM_OPTIMIZERS="${NORM_OPTIMIZERS:-cobyla,qnspsa,nft}"
NORM_BUDGET="${NORM_BUDGET:-0}"
NORM_BUDGET_K="${NORM_BUDGET_K:-37.5}"
NORM_STRIDE="${NORM_STRIDE:-10}"
NOISE_MODES="${NOISE_MODES:-ideal,shot-noise}"
TARGET_PRECISION="${TARGET_PRECISION:-0.1}"
GRAD_METHOD="${GRAD_METHOD:-auto}"
SEED="${SEED:-10}"

IFS=',' read -r -a NOISE_MODE_ARR <<< "$NOISE_MODES"
IFS=',' read -r -a NORM_SEED_ARR <<< "$NORM_SEEDS"
IFS=',' read -r -a NORM_SIZE_ARR <<< "$NORM_SIZES"
NUM_NOISE=${#NOISE_MODE_ARR[@]}
NUM_NORM_SEEDS=${#NORM_SEED_ARR[@]}
NUM_NORM_SIZES=${#NORM_SIZE_ARR[@]}
TASKS_PER_MODE=$((1 + NUM_NORM_SEEDS * NUM_NORM_SIZES))
TOTAL_TASKS=$((NUM_NOISE * TASKS_PER_MODE))

if (( SLURM_ARRAY_TASK_ID > TOTAL_TASKS )); then
	echo "Skipping task $SLURM_ARRAY_TASK_ID (TOTAL_TASKS=$TOTAL_TASKS)"
	exit 0
fi

RUN_META_DIR="outputs/.runmeta/${SLURM_ARRAY_JOB_ID}"
mkdir -p "$RUN_META_DIR"

{
	flock 200
	if [[ ! -f "$RUN_META_DIR/run_name.txt" ]]; then
		printf '%s_%s\n' "$(date +%Y%m%d-%H%M%S)" "phase3_grad-metrics" > "$RUN_META_DIR/run_name.txt"
	fi
} 200>"$RUN_META_DIR/lock"

RUN_ROOT="outputs/$(<"$RUN_META_DIR/run_name.txt")"
mkdir -p "$RUN_ROOT"

IDX=$((SLURM_ARRAY_TASK_ID - 1))
MODE_IDX=$((IDX / TASKS_PER_MODE))
LOCAL=$((IDX % TASKS_PER_MODE))
NOISE_MODE="${NOISE_MODE_ARR[$MODE_IDX]}"

if (( LOCAL == 0 )); then
	TASK_KIND="variance"
	TASK_JOB_NAME="p3_grad_var_${NOISE_MODE}"
	EXTRA_ARGS=(--noise-mode "$NOISE_MODE" --skip-norm)
else
	TASK_KIND="norm"
	NORM_IDX=$((LOCAL - 1))
	SEED_IDX=$((NORM_IDX / NUM_NORM_SIZES))
	Q_IDX=$((NORM_IDX % NUM_NORM_SIZES))
	NORM_SEED="${NORM_SEED_ARR[$SEED_IDX]}"
	NORM_Q="${NORM_SIZE_ARR[$Q_IDX]}"
	TASK_JOB_NAME="p3_grad_norm_${NOISE_MODE}_s${NORM_SEED}_q${NORM_Q}"
	EXTRA_ARGS=(
		--noise-mode "$NOISE_MODE"
		--skip-variance
		--norm-seed "$NORM_SEED"
		--norm-qubits "$NORM_Q"
	)
fi

scontrol update JobId="${SLURM_JOB_ID}" JobName="${TASK_JOB_NAME}" >/dev/null 2>&1 || true

echo "Phase3 grad-metrics | task=$SLURM_ARRAY_TASK_ID/$TOTAL_TASKS | kind=$TASK_KIND | noise=$NOISE_MODE"
echo "Slurm job name: $TASK_JOB_NAME"
echo "Run root: $RUN_ROOT"

bnd run --cpu python src/experiments/exp-gradient-metrics.py \
	--architectures "$ARCHITECTURES" \
	--qubit-sizes "$QUBIT_SIZES" \
	--n-samples "$N_SAMPLES" \
	--init-dist "$INIT_DIST" \
	--norm-sizes "$NORM_SIZES" \
	--norm-seeds "$NORM_SEEDS" \
	--norm-optimizers "$NORM_OPTIMIZERS" \
	--norm-budget "$NORM_BUDGET" \
	--norm-budget-k "$NORM_BUDGET_K" \
	--norm-stride "$NORM_STRIDE" \
	--target-precision "$TARGET_PRECISION" \
	--grad-method "$GRAD_METHOD" \
	--seed "$SEED" \
	--outdir "$RUN_ROOT" \
	"${EXTRA_ARGS[@]}"

echo "Completed $TASK_KIND ($NOISE_MODE) -> $RUN_ROOT"
