#!/bin/bash
#SBATCH --job-name=barren_opt_real_hw
#SBATCH --output=outputs/logs/real_hw_M_%A_%a.out
#SBATCH --error=outputs/logs/real_hw_M_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=CPU
#SBATCH --array=1-25

# Fail the job if any command fails.
set -euo pipefail

cd /home/sdelostoyos/pfs/projects/barren-plateaus-qnns

RUN_LABEL="multijob-real-hw"
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

# Define arrays (fewer for real hardware due to cost/queue)
SEEDS=(10 20 30 40 50)           # 5 semillas
QUBITS=(4 8 12 16 20)            # 5 tamaños de qubits
NUM_QUBITS=${#QUBITS[@]}

# Calcula índices (aritmética 2D)
IDX=$((SLURM_ARRAY_TASK_ID - 1))
SEED_IDX=$((IDX / NUM_QUBITS))
Q_IDX=$((IDX % NUM_QUBITS))

SEED=${SEEDS[$SEED_IDX]}
Q=${QUBITS[$Q_IDX]}

SEED_DIR="${RUN_ROOT}/seed_${SEED}"
RUN_DIR="${SEED_DIR}/qubits_${Q}"

mkdir -p "$RUN_DIR"

# Reduced but still increasing exponentially with qubits.
# For Q in {4,8,12,16,20} -> budget in {32,64,128,256,512}.
BUDGET_EVALS=$((32 * (2 ** ((Q - 4) / 4))))

echo "Running real-hw seed=$SEED, n_qubits=$Q (array task $SLURM_ARRAY_TASK_ID)"
echo "Run root: $RUN_ROOT"
echo "Budget evals: $BUDGET_EVALS"

bnd run --cpu python src/exp-real.py \
	--n_qubits $Q \
	--seed $SEED \
	--budget-evals "$BUDGET_EVALS" \
	--shots 1024 \
	--window 10 \
	--tolerance 1e-3 \
	--outdir "$RUN_DIR"

echo "Completed: seed=$SEED, n_qubits=$Q"
