#!/bin/bash
#SBATCH --job-name=barren_opt_real_hw
#SBATCH --output=outputs/logs/real_hw_M_%A_%a.out
#SBATCH --error=outputs/logs/real_hw_M_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=CPU-LARGE
#SBATCH --time=7-00:00:00
#SBATCH --array=1-25%1

# Fail the job if any command fails.
set -euo pipefail

cd /home/sdelostoyos/pfs/projects/barren-plateaus-qnns

BACKEND="${BACKEND:-ibm_basquecountry}"
RUNTIME_RESULT_TIMEOUT="${RUNTIME_RESULT_TIMEOUT:-600}"

RUN_LABEL="phase1_real-hw_baseline"
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
NUM_SEEDS=${#SEEDS[@]}
NUM_QUBITS=${#QUBITS[@]}

# Calcula índices (aritmética 2D, qubit-major)
IDX=$((SLURM_ARRAY_TASK_ID - 1))
Q_IDX=$((IDX / NUM_SEEDS))
SEED_IDX=$((IDX % NUM_SEEDS))

SEED=${SEEDS[$SEED_IDX]}
Q=${QUBITS[$Q_IDX]}
CIRCUIT_NAME="baseline_hea"
TASK_JOB_NAME="p1_real_${CIRCUIT_NAME}_s${SEED}_q${Q}"
scontrol update JobId="${SLURM_JOB_ID}" JobName="${TASK_JOB_NAME}" >/dev/null 2>&1 || true

SEED_DIR="${RUN_ROOT}/seed_${SEED}"
RUN_DIR="${SEED_DIR}/qubits_${Q}"

mkdir -p "$RUN_DIR"

# Reduced but still increasing exponentially with qubits.
# For Q in {4,8,12,16,20} -> budget in {32,64,128,256,512}.
BUDGET_EVALS=$((32 * (2 ** ((Q - 4) / 4))))

echo "Running real-hw seed=$SEED, n_qubits=$Q (array task $SLURM_ARRAY_TASK_ID)"
echo "Slurm job name: $TASK_JOB_NAME"
echo "Run root: $RUN_ROOT"
echo "Budget evals: $BUDGET_EVALS"
echo "Backend: $BACKEND"
echo "Runtime result timeout: $RUNTIME_RESULT_TIMEOUT"

bnd run --cpu python src/phase1/exp-real.py \
	--n_qubits $Q \
	--seed $SEED \
	--budget-evals "$BUDGET_EVALS" \
	--shots 1024 \
	--window 10 \
	--tolerance 1e-3 \
	--backend "$BACKEND" \
	--runtime-result-timeout "$RUNTIME_RESULT_TIMEOUT" \
	--outdir "$RUN_DIR"

echo "Completed: seed=$SEED, n_qubits=$Q"
