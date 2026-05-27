#!/bin/bash
#SBATCH --job-name=barren_opt_grid
#SBATCH --output=outputs/logs/M_%A_%a.out
#SBATCH --error=outputs/logs/M_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=CPU
#SBATCH --array=1-25

cd /home/sdelostoyos/pfs/projects/barren-plateaus-qnns

RUN_LABEL="multijob"
RUN_META_DIR="outputs/.runmeta/${SLURM_ARRAY_JOB_ID}"
mkdir -p "$RUN_META_DIR"

# Create one shared timestamped root for the whole array submission.
{
	flock 200
	if [[ ! -f "$RUN_META_DIR/run_name.txt" ]]; then
		printf '%s_%s\n' "$(date +%Y%m%d-%H%M%S)" "$RUN_LABEL" > "$RUN_META_DIR/run_name.txt"
	fi
} 200>"$RUN_META_DIR/lock"

RUN_ROOT="outputs/$(<"$RUN_META_DIR/run_name.txt")"
mkdir -p "$RUN_ROOT"

# Define arrays
SEEDS=(10 20 30 40 50)           # 5 semillas
QUBITS=(4 8 12 16 20)            # 5 tamaños de qubits
NUM_SEEDS=${#SEEDS[@]}
NUM_QUBITS=${#QUBITS[@]}

# Calcula índices (aritmética 2D, qubit-major)
IDX=$((SLURM_ARRAY_TASK_ID - 1))
Q_IDX=$((IDX / NUM_SEEDS))
SEED_IDX=$((IDX % NUM_SEEDS))

SEED=${SEEDS[$SEED_IDX]}
Q=${QUBITS[$Q_IDX]}
CIRCUIT_NAME="baseline_hea"
TASK_JOB_NAME="p1_ideal_${CIRCUIT_NAME}_s${SEED}_q${Q}"
scontrol update JobId="${SLURM_JOB_ID}" JobName="${TASK_JOB_NAME}" >/dev/null 2>&1 || true

SEED_DIR="${RUN_ROOT}/seed_${SEED}"
RUN_DIR="${SEED_DIR}/qubits_${Q}"

mkdir -p "$RUN_DIR"

echo "Running seed=$SEED, n_qubits=$Q (array task $SLURM_ARRAY_TASK_ID)"
echo "Slurm job name: $TASK_JOB_NAME"
echo "Run root: $RUN_ROOT"

bnd run --cpu python src/phase1/exp-ideal.py \
	--n_qubits $Q \
	--seed $SEED \
	--budget-k 37.5 \
	--outdir "$RUN_DIR"
