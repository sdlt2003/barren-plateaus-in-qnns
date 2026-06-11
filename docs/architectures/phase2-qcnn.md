# Phase 2 - QCNN Architecture Benchmark

## Goal

Phase 2 compares:

- `baseline_hea + current modifier pipeline`
- `qcnn + current modifier pipeline`

The modifier pipeline is intentionally unchanged from phase 1:

- same objective observable style (single-qubit `Z` readout),
- same optimizers (`COBYLA`, `QNSPSA`),
- same budget logic and early stopping strategy.

Only the circuit architecture is varied.

## Implemented architectures

- `baseline_hea`
  - Hardware-efficient ansatz (`efficient_su2`) with logarithmic-depth reps rule.
  - Readout qubit: last qubit (`n_qubits - 1`).

- `qcnn`
  - Hierarchical conv+pool style QCNN topology.
  - Two convolution sublayers per level on alternating neighboring pairs.
  - Pooling module follows the paper-inspired direction: source qubit `A` controls unitaries on neighbor `B`, and `A` is inactive after pooling.
  - Logarithmic number of levels.
  - Readout qubit: final survivor `B` from the hierarchical pooling tree.
  - Constraint: `n_qubits` must be a power of 2 in this first implementation.

## Phase 2 experiment scripts

- `src/phase2/exp-ideal.py`
- `src/phase2/exp-shot-noise.py`
- `src/phase2/exp-real.py`

All scripts expose:

- `--architecture {baseline_hea,qcnn}`
- `--n_qubits`
- budget controls (`--budget-evals` or dynamic `--budget-k`)

## Local run examples

Ideal:

```bash
bnd run --cpu python src/phase2/exp-ideal.py --architecture baseline_hea --n_qubits 8 --outdir outputs/p2_ideal_baseline_q8
bnd run --cpu python src/phase2/exp-ideal.py --architecture qcnn --n_qubits 8 --outdir outputs/p2_ideal_qcnn_q8
```

Shot-noise:

```bash
bnd run --cpu python src/phase2/exp-shot-noise.py --architecture baseline_hea --n_qubits 8 --target-precision 0.1 --outdir outputs/p2_shot_baseline_q8
bnd run --cpu python src/phase2/exp-shot-noise.py --architecture qcnn --n_qubits 8 --target-precision 0.1 --outdir outputs/p2_shot_qcnn_q8
```

Real hardware:

```bash
bnd run --cpu python src/phase2/exp-real.py --architecture baseline_hea --n_qubits 4 --budget-evals 32 --outdir outputs/p2_real_baseline_q4
bnd run --cpu python src/phase2/exp-real.py --architecture qcnn --n_qubits 4 --budget-evals 32 --outdir outputs/p2_real_qcnn_q4
```

## Slurm launchers

- `src/phase2/sbatches/exp-ideal.sbatch`
- `src/phase2/sbatches/exp-shot-noise.sbatch`
- `src/phase2/sbatches/exp-real.sbatch`

The launchers keep qubit-major indexing to obtain low-qubit coverage across seeds earlier.
Set architecture at submit time:

```bash
ARCHITECTURE=qcnn sbatch src/phase2/sbatches/exp-ideal.sbatch
```

## Architecture visualization script

`src/visualize_architectures.py` renders architecture diagrams using the same factory consumed by experiments.

Example:

```bash
bnd run --cpu python src/visualize_architectures.py --architectures baseline_hea qcnn --n-qubits 8 --output-dir outputs/architecture_viz_q8
```

Outputs:

- text circuit diagram (`*.txt`) for each architecture,
- optional image diagram (`*.png`) when matplotlib backend is available,
- summary CSV with `num_params`, `depth`, `size`, and `readout_qubit`.
