#!/bin/bash
#SBATCH --job-name=barren_opt_real_hw_p2
#SBATCH --output=outputs/logs/p2_real_hw_M_%A_%a.out
#SBATCH --error=outputs/logs/p2_real_hw_M_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=CPU-LARGE
#SBATCH --time=7-00:00:00
#SBATCH --array=1-5%1

set -euo pipefail
cd /home/sdelostoyos/pfs/projects/barren-plateaus-qnns

BACKEND="${BACKEND:-ibm_basquecountry}"
# Empty/unset = wait without client-side timeout (IBM-recommended).
RUNTIME_RESULT_TIMEOUT="${RUNTIME_RESULT_TIMEOUT:-}"
# Empty/unset = Session() without max_time (IBM plan default, typically ~8h Premium).
SESSION_MAX_TIME="${SESSION_MAX_TIME:-}"

ARCHITECTURE="${ARCHITECTURE:-baseline_hea}"
case "$ARCHITECTURE" in
	baseline_hea) ARCH_LABEL="baseline" ;;
	qcnn) ARCH_LABEL="qcnn" ;;
	resqnet) ARCH_LABEL="resqnet" ;;
	*) ARCH_LABEL="$ARCHITECTURE" ;;
esac
RUN_LABEL="phase2_${ARCH_LABEL}-real-hw"
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

SEEDS=(10 20 30 40 50)
if [[ "$ARCHITECTURE" == "qcnn" || "$ARCHITECTURE" == "resqnet" ]]; then
	QUBIT_SIZES="4,8,16"
else
	QUBIT_SIZES="4,8,12,16,20"
fi

SEED_IDX=$((SLURM_ARRAY_TASK_ID - 1))
if (( SEED_IDX < 0 || SEED_IDX >= ${#SEEDS[@]} )); then
	echo "Skipping task $SLURM_ARRAY_TASK_ID (invalid seed index)"
	exit 0
fi

SEED=${SEEDS[$SEED_IDX]}
TASK_JOB_NAME="p2_real_${ARCH_LABEL}_s${SEED}"
scontrol update JobId="${SLURM_JOB_ID}" JobName="${TASK_JOB_NAME}" >/dev/null 2>&1 || true

mkdir -p "${RUN_ROOT}/seed_${SEED}"

echo "Phase2 real-hw | arch=$ARCHITECTURE | seed=$SEED | qubits=$QUBIT_SIZES | task=$SLURM_ARRAY_TASK_ID"
echo "Slurm job name: $TASK_JOB_NAME"
echo "Run root: $RUN_ROOT"
echo "Backend: $BACKEND"
echo "Session max_time: ${SESSION_MAX_TIME:-IBM plan default}"
echo "Runtime result timeout: ${RUNTIME_RESULT_TIMEOUT:-disabled}"
echo "IBM session mode: one session per seed"

TIMEOUT_ARGS=()
if [[ -n "$RUNTIME_RESULT_TIMEOUT" ]]; then
	TIMEOUT_ARGS=(--runtime-result-timeout "$RUNTIME_RESULT_TIMEOUT")
fi

SESSION_ARGS=()
if [[ -n "$SESSION_MAX_TIME" ]]; then
	SESSION_ARGS=(--session-max-time "$SESSION_MAX_TIME")
fi

bnd run --cpu python src/phase2/exp-real.py \
	--architecture "$ARCHITECTURE" \
	--seed "$SEED" \
	--run-root "$RUN_ROOT" \
	--qubit-sizes "$QUBIT_SIZES" \
	--shots 1024 \
	--window 10 \
	--tolerance 1e-3 \
	--backend "$BACKEND" \
	"${SESSION_ARGS[@]}" \
	"${TIMEOUT_ARGS[@]}"

echo "Completed seed=$SEED across qubits=$QUBIT_SIZES"
