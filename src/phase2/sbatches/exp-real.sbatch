#!/bin/bash
#SBATCH --job-name=barren_opt_real_hw_p2
#SBATCH --output=outputs/logs/p2_real_hw_M_%A_%a.out
#SBATCH --error=outputs/logs/p2_real_hw_M_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=CPU-LARGE
#SBATCH --time=7-00:00:00
#SBATCH --array=1-25%8

set -euo pipefail
cd /home/sdelostoyos/pfs/projects/barren-plateaus-qnns

BACKEND="${BACKEND:-ibm_basquecountry}"
EXECUTION_MODE="${EXECUTION_MODE:-batch}"
ARRAY_PARALLEL="${ARRAY_PARALLEL:-8}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
# Empty/unset = wait without client-side timeout (IBM-recommended).
RUNTIME_RESULT_TIMEOUT="${RUNTIME_RESULT_TIMEOUT:-}"
# Empty/unset = Batch()/Session() without max_time (IBM plan default, typically ~8h Premium).
RUNTIME_MAX_TIME="${RUNTIME_MAX_TIME:-${SESSION_MAX_TIME:-}}"

ARCHITECTURE="${ARCHITECTURE:-baseline_hea}"
case "$ARCHITECTURE" in
	baseline_hea) ARCH_LABEL="baseline" ;;
	qcnn) ARCH_LABEL="qcnn" ;;
	resqnet) ARCH_LABEL="resqnet" ;;
	*) ARCH_LABEL="$ARCHITECTURE" ;;
esac
RUN_LABEL="phase2_real-hw_${ARCH_LABEL}"
if [[ -n "${RUN_ROOT:-}" ]]; then
	mkdir -p "$RUN_ROOT"
	echo "Using existing run root: $RUN_ROOT"
else
	RUN_META_DIR="outputs/.runmeta/${SLURM_ARRAY_JOB_ID}"
	mkdir -p "$RUN_META_DIR"

	{
		flock 200
		if [[ ! -f "$RUN_META_DIR/run_name.txt" ]]; then
			printf '%s_%s\n' "$(date +%Y%m%d-%H%M%S)" "$RUN_LABEL" > "$RUN_META_DIR/run_name.txt"
		fi
	} 200>"$RUN_META_DIR/lock"

	RUN_ROOT="outputs/$(<"$RUN_META_DIR/run_name.txt")"
	mkdir -p "$RUN_ROOT"
fi

SEEDS=(10 20 30 40 50)
if [[ -n "${QUBITS_OVERRIDE:-}" ]]; then
	IFS=',' read -r -a QUBITS <<< "$QUBITS_OVERRIDE"
else
	if [[ "$ARCHITECTURE" == "qcnn" || "$ARCHITECTURE" == "resqnet" ]]; then
		QUBITS=(4 8 16)
	else
		QUBITS=(4 8 12 16 20)
	fi
fi

# For Q=32 we split optimizers into separate Slurm tasks to reduce TTL risk.
# For all other Q we keep the legacy single-task "both optimizers" behavior.
QOPS_Q=()
QOPS_OPT=()
for q in "${QUBITS[@]}"; do
	if [[ "$q" == "32" ]]; then
		QOPS_Q+=(32 32)
		QOPS_OPT+=(cobyla qnspsa)
	else
		QOPS_Q+=("$q")
		QOPS_OPT+=(both)
	fi
done

NUM_SEEDS=${#SEEDS[@]}
NUM_QOPS=${#QOPS_Q[@]}
TOTAL_TASKS=$((NUM_SEEDS * NUM_QOPS))

if (( SLURM_ARRAY_TASK_ID > TOTAL_TASKS )); then
	echo "Skipping task $SLURM_ARRAY_TASK_ID (TOTAL_TASKS=$TOTAL_TASKS for arch=$ARCHITECTURE)"
	exit 0
fi

# qubit-major: tasks 1..5 -> first q-op all seeds, ...
IDX=$((SLURM_ARRAY_TASK_ID - 1))
QOP_IDX=$((IDX / NUM_SEEDS))
SEED_IDX=$((IDX % NUM_SEEDS))
SEED=${SEEDS[$SEED_IDX]}
Q=${QOPS_Q[$QOP_IDX]}
OPTIMIZER=${QOPS_OPT[$QOP_IDX]}

TASK_JOB_NAME="p2_real_${ARCH_LABEL}_s${SEED}_q${Q}"
if [[ "$OPTIMIZER" != "both" ]]; then
	TASK_JOB_NAME="${TASK_JOB_NAME}_${OPTIMIZER}"
fi
scontrol update JobId="${SLURM_JOB_ID}" JobName="${TASK_JOB_NAME}" >/dev/null 2>&1 || true

RUN_DIR="${RUN_ROOT}/seed_${SEED}/qubits_${Q}"
if [[ "$OPTIMIZER" != "both" ]]; then
	RUN_DIR="${RUN_DIR}/opt_${OPTIMIZER}"
fi
mkdir -p "$RUN_DIR"

echo "Phase2 real-hw | arch=$ARCHITECTURE | seed=$SEED | qubits=$Q | task=$SLURM_ARRAY_TASK_ID/$TOTAL_TASKS"
echo "Slurm job name: $TASK_JOB_NAME"
echo "Run root: $RUN_ROOT"
echo "Backend: $BACKEND"
echo "Execution mode: $EXECUTION_MODE (1 IBM container per Slurm task)"
echo "Runtime max_time: ${RUNTIME_MAX_TIME:-IBM plan default}"
echo "Runtime result timeout: ${RUNTIME_RESULT_TIMEOUT:-disabled}"
echo "Skip completed: $SKIP_COMPLETED"
echo "Optimizer: $OPTIMIZER"

TIMEOUT_ARGS=()
if [[ -n "$RUNTIME_RESULT_TIMEOUT" ]]; then
	TIMEOUT_ARGS=(--runtime-result-timeout "$RUNTIME_RESULT_TIMEOUT")
fi

MAX_TIME_ARGS=()
if [[ -n "$RUNTIME_MAX_TIME" ]]; then
	MAX_TIME_ARGS=(--runtime-max-time "$RUNTIME_MAX_TIME")
fi

SKIP_ARGS=()
if [[ "$SKIP_COMPLETED" == "1" ]]; then
	SKIP_ARGS=(--skip-completed)
fi

BUDGET_ARGS=()
if [[ -n "${BUDGET_K:-}" ]]; then
	BUDGET_ARGS=(--budget-k "$BUDGET_K")
fi
if [[ -n "${BUDGET_EVALS:-}" ]]; then
	BUDGET_ARGS+=(--budget-evals "$BUDGET_EVALS")
fi

bnd run --cpu python src/phase2/exp-real.py \
	--architecture "$ARCHITECTURE" \
	--seed "$SEED" \
	--n_qubits "$Q" \
	--outdir "$RUN_DIR" \
	--optimizer "$OPTIMIZER" \
	--execution-mode "$EXECUTION_MODE" \
	--shots 1024 \
	--window 10 \
	--tolerance 1e-3 \
	--backend "$BACKEND" \
	"${MAX_TIME_ARGS[@]}" \
	"${TIMEOUT_ARGS[@]}" \
	"${SKIP_ARGS[@]}" \
	"${BUDGET_ARGS[@]}"

echo "Completed seed=$SEED qubits=$Q"
