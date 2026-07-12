#!/bin/bash
#SBATCH --job-name=barren_grad_merge_p3
#SBATCH --output=outputs/logs/p3_grad_merge_%j.out
#SBATCH --error=outputs/logs/p3_grad_merge_%j.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=CPU
#SBATCH --time=00:10:00

set -euo pipefail
cd /home/sdelostoyos/pfs/projects/barren-plateaus-qnns

ARRAY_JOB_ID="${ARRAY_JOB_ID:?Set ARRAY_JOB_ID to the Slurm array job id}"
RUN_META_DIR="outputs/.runmeta/${ARRAY_JOB_ID}"
RUN_ROOT="outputs/$(<"$RUN_META_DIR/run_name.txt")"

echo "Merging partial grad-metrics under $RUN_ROOT"
bnd run --cpu python src/experiments/exp-gradient-metrics.py \
	--aggregate-only \
	--outdir "$RUN_ROOT"
echo "Done -> $RUN_ROOT/grad_metrics.json"
