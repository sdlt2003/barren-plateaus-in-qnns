#!/bin/bash
#SBATCH --job-name=barren_opt_real_hw_p2
#SBATCH --output=outputs/logs/p2_real_hw_M_%A_%a.out
#SBATCH --error=outputs/logs/p2_real_hw_M_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=CPU
#SBATCH --array=1-25

set -euo pipefail
cd /home/sdelostoyos/pfs/projects/barren-plateaus-qnns

ARCHITECTURE="${ARCHITECTURE:-baseline_hea}"
case "$ARCHITECTURE" in
	baseline_hea) ARCH_LABEL="baseline" ;;
	qcnn) ARCH_LABEL="qcnn" ;;
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

SEEDS=(10 20 30 40 50)  # 5 semillas
if [[ "$ARCHITECTURE" == "qcnn" ]]; then
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
TASK_JOB_NAME="p2_real_${ARCH_LABEL}_s${SEED}_q${Q}"
scontrol update JobId="${SLURM_JOB_ID}" JobName="${TASK_JOB_NAME}" >/dev/null 2>&1 || true

SEED_DIR="${RUN_ROOT}/seed_${SEED}"
RUN_DIR="${SEED_DIR}/qubits_${Q}"
mkdir -p "$RUN_DIR"

# Reduced but increasing exponentially with qubits.
BUDGET_EVALS=$((32 * (2 ** ((Q - 4) / 4))))

echo "Phase2 real-hw | arch=$ARCHITECTURE | seed=$SEED | qubits=$Q | task=$SLURM_ARRAY_TASK_ID"
echo "Slurm job name: $TASK_JOB_NAME"
echo "Run root: $RUN_ROOT"
echo "Budget evals: $BUDGET_EVALS"

bnd run --cpu python src/phase2/exp-real.py \
	--architecture "$ARCHITECTURE" \
	--n_qubits "$Q" \
	--seed "$SEED" \
	--budget-evals "$BUDGET_EVALS" \
	--shots 1024 \
	--window 10 \
	--tolerance 1e-3 \
	--outdir "$RUN_DIR"

echo "Completed: seed=$SEED, n_qubits=$Q"
