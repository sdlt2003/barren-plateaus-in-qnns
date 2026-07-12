#!/bin/bash
#SBATCH --job-name=barren_opt_grid_p2
#SBATCH --output=outputs/logs/p2_M_%A_%a.out
#SBATCH --error=outputs/logs/p2_M_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=CPU
#SBATCH --time=1-00:00:00
#SBATCH --array=1-25

cd /home/sdelostoyos/pfs/projects/barren-plateaus-qnns

ARCHITECTURE="${ARCHITECTURE:-baseline}"
case "$ARCHITECTURE" in
	baseline) ARCH_LABEL="baseline" ;;
	qcnn) ARCH_LABEL="qcnn" ;;
	resqnet) ARCH_LABEL="resqnet" ;;
	*) ARCH_LABEL="$ARCHITECTURE" ;;
esac
RUN_PHASE="${RUN_PHASE:-phase3}"
RUN_LABEL="${RUN_PHASE}_ideal_${ARCH_LABEL}"
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

SEEDS=(10 20 30 40 50)  # 5 semillas
if [[ -n "${QUBITS_OVERRIDE:-}" ]]; then
	IFS=',' read -r -a QUBITS <<< "$QUBITS_OVERRIDE"
elif [[ "$ARCHITECTURE" == "qcnn" || "$ARCHITECTURE" == "resqnet" ]]; then
	QUBITS=(4 8 16)      # Solo potencias de 2 en el rango original
else
	QUBITS=(4 8 12 16 20)
fi
NUM_SEEDS=${#SEEDS[@]}
NUM_QUBITS=${#QUBITS[@]}
TOTAL_TASKS=$((NUM_SEEDS * NUM_QUBITS))

if (( SLURM_ARRAY_TASK_ID > TOTAL_TASKS )); then
	echo "Skipping task $SLURM_ARRAY_TASK_ID (TOTAL_TASKS=$TOTAL_TASKS for arch=$ARCHITECTURE)"
	exit 0
fi

# Índices 2D en orden qubit-major.
IDX=$((SLURM_ARRAY_TASK_ID - 1))
Q_IDX=$((IDX / NUM_SEEDS))
SEED_IDX=$((IDX % NUM_SEEDS))

SEED=${SEEDS[$SEED_IDX]}
Q=${QUBITS[$Q_IDX]}
TASK_JOB_NAME="p2_ideal_${ARCH_LABEL}_s${SEED}_q${Q}"
scontrol update JobId="${SLURM_JOB_ID}" JobName="${TASK_JOB_NAME}" >/dev/null 2>&1 || true

SEED_DIR="${RUN_ROOT}/seed_${SEED}"
RUN_DIR="${SEED_DIR}/qubits_${Q}"
mkdir -p "$RUN_DIR"

echo "Phase2 ideal | arch=$ARCHITECTURE | seed=$SEED | qubits=$Q | task=$SLURM_ARRAY_TASK_ID"
echo "Slurm job name: $TASK_JOB_NAME"
echo "Run root: $RUN_ROOT"

OPTIMIZERS="${OPTIMIZERS:-cobyla,qnspsa,nft}"
TRACK_PARAMS="${TRACK_PARAMS:-0}"
GRAD_STRIDE="${GRAD_STRIDE:-10}"
TRACK_ARGS=()
if [[ "$TRACK_PARAMS" == "1" ]]; then
	TRACK_ARGS+=(--track-params)
fi
if [[ "${TRACK_GRAD:-0}" == "1" ]]; then
	TRACK_ARGS+=(--track-grad)
elif [[ "${TRACK_GRAD_NORM:-0}" == "1" ]]; then
	TRACK_ARGS+=(--track-grad-norm)
fi
if [[ "${TRACK_GRAD_VAR:-0}" == "1" ]]; then
	TRACK_ARGS+=(--track-grad-var)
fi
if [[ ${#TRACK_ARGS[@]} -gt 0 ]]; then
	TRACK_ARGS+=(--grad-stride "$GRAD_STRIDE")
fi

bnd run --cpu python src/experiments/exp-ideal.py \
	--architecture "$ARCHITECTURE" \
	--n_qubits "$Q" \
	--seed "$SEED" \
	--budget-k 37.5 \
	--optimizers "$OPTIMIZERS" \
	"${TRACK_ARGS[@]}" \
	--outdir "$RUN_DIR"
