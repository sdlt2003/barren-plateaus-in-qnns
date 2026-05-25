# Barren Plateaus QNNs

Experiments to compare COBYLA vs QNSPSA for a simple ansatz under ideal and shot-noise settings. Outputs are written under `outputs/` with per-run histories and plots.

Budgeting:
- `exp-ideal.py`, `exp-shot-noise.py`, and `exp-real.py` use dynamic budget by default: `budget_evals = k * p` where `p` is ansatz parameter count.
- Default `k`: 37.5 for ideal and shot-noise, 10 for real hardware.
- Use `--budget-k` to tune dynamic scaling.
- Use `--budget-evals` to force a fixed budget.

## Setup

- Python environment with the packages in `requirements.txt`.
- If you use `bnd`, run commands as shown below (the sbatch scripts do this).

## Run Experiments

Ideal (statevector):

```bash
bnd run python src/exp-ideal.py --n_qubits 8 --outdir outputs/opt_run --seed 7

# Dynamic budget with custom k
bnd run python src/exp-ideal.py --n_qubits 8 --budget-k 37.5

# Fixed budget override
bnd run python src/exp-ideal.py --n_qubits 8 --budget-evals 150
```

Shot-noise (Monte Carlo):

```bash
bnd run python src/exp-shot-noise.py --n_qubits 12 --outdir outputs/shot_noise_run --seed 7

# Dynamic budget with custom k
bnd run python src/exp-shot-noise.py --n_qubits 12 --budget-k 37.5

# Fixed budget override
bnd run python src/exp-shot-noise.py --n_qubits 12 --budget-evals 300
```

Each run writes:
- `optimizer_history.npz`
- `optimizer_compare.png`

## Analyze Outputs

Analyze a specific run directory (full path or run name):

```bash
bnd run python src/analyze_outputs.py outputs/20260509-225612_multijob
bnd run python src/analyze_outputs.py 20260509-225612_multijob
```

The analysis writes to `<run_dir>/analysis/`:
- `summary_per_run.csv`
- `summary_aggregated.csv`
- `final_cost_vs_qubits.png`

## Real Hardware (IBM Quantum)

Run on actual IBM quantum hardware—**doesn't use cluster resources**:

```bash
# Needs credentials from .env (API_KEY, CRN_KEY)
bnd run python src/exp-real.py --n_qubits 4 --outdir outputs/real_run --seed 7

# With specific backend (defaults to least_busy)
bnd run python src/exp-real.py --n_qubits 4 --backend ibm_oslo --shots 1024

# Dynamic budget on hardware (default k=10)
bnd run python src/exp-real.py --n_qubits 4 --budget-k 10

# Fixed budget override
bnd run python src/exp-real.py --n_qubits 4 --budget-evals 30
```

**Important:** Computation runs directly on IBM Quantum hardware, not locally. Each objective evaluation submits a job to the backend. Execution time and cost depend on IBM's queue.

Grid run (multiple seeds × qubits):

```bash
sbatch multijobs-real-hardware.sbatch
```

Runs 6 combinations: seeds {42, 123} × qubits {4, 8, 12}. Organizes results in timestamped folders with per-seed/per-qubit subdirectories, then analyze with:

```bash
bnd run python src/analyze_outputs.py <run_name>
```

## Slurm Helpers

The sbatch files launch grid runs locally and organize results under timestamped folders:

```bash
sbatch multijobs.sbatch
sbatch multijobs-shot-noise.sbatch
```
sbatch multijobs-real-hardware.sbatch  # Real hardware grid (fewer qubits due to cost)
```

## Output Layout

Typical structure for timestamped runs:

outputs/<timestamp>_<label>/analysis/summary_per_run.csv
outputs/<timestamp>_<label>/analysis/summary_aggregated.csv
```
