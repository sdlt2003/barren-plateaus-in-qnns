# Budget Eval Logic

Este documento resume los valores de `budget_evals` por combinacion de qubits para cada script.

## Regla usada

- Modo dinamico: `budget_evals = round(k * p)`.
- `p` es `ansatz.num_parameters`.
- Para este ansatz (`efficient_su2` con `reps = round(log2(n_qubits))` y entanglement lineal), aqui se usa:
  - `p = 2 * n_qubits * (reps + 1)`.
- `MIN_BUDGET_EVALS = 1` (no afecta estas combinaciones porque todos los valores quedan muy por encima de 1).

## Tabla 1 - `src/exp-ideal.py` (k por defecto = 37.5)

| n_qubits | reps | p (num_params) | budget_evals por defecto |
|---|---:|---:|---:|
| 4 | 2 | 24 | 900 |
| 8 | 3 | 64 | 2400 |
| 12 | 4 | 120 | 4500 |
| 16 | 4 | 160 | 6000 |
| 20 | 4 | 200 | 7500 |

## Tabla 2 - `src/exp-shot-noise.py` (k por defecto = 37.5)

| n_qubits | reps | p (num_params) | budget_evals por defecto |
|---|---:|---:|---:|
| 4 | 2 | 24 | 900 |
| 8 | 3 | 64 | 2400 |
| 12 | 4 | 120 | 4500 |
| 16 | 4 | 160 | 6000 |
| 20 | 4 | 200 | 7500 |

## Tabla 3 - `src/exp-real.py` (k por defecto = 10, ahora dinamico por defecto)

| n_qubits | reps | p (num_params) | budget_evals por defecto |
|---|---:|---:|---:|
| 4 | 2 | 24 | 240 |
| 8 | 3 | 64 | 640 |
| 12 | 4 | 120 | 1200 |
| 16 | 4 | 160 | 1600 |
| 20 | 4 | 200 | 2000 |

## Perfil de ejecucion real-hw en Slurm (carga reducida, exponencial)

Para evitar timeouts en grid de hardware real, el launcher `src/sbatches/exp-real.sbatch`
usa un budget fijo por task que crece exponencialmente con qubits:

- Formula: `budget_evals = 32 * 2^((Q - 4) / 4)` para `Q in {4, 8, 12, 16, 20}`.
- Todos los budgets quedan por encima de 30 y son mucho menores que los dinamicos por defecto.

| Qubits (Q) | Budget evals en sbatch real-hw |
|---:|---:|
| 4 | 32 |
| 8 | 64 |
| 12 | 128 |
| 16 | 256 |
| 20 | 512 |

Esto solo afecta al launcher sbatch de real hardware. El script `src/exp-real.py` sigue
permitiendo modo dinamico por defecto cuando se ejecuta directamente.

## Overrides por CLI

- En los 3 scripts, si pasas `--budget-evals N`, se usa modo fijo con ese `N`.
- Si no pasas `--budget-evals`, se usa modo dinamico con `--budget-k` (o su default del script).

## Nota importante

- En `QNSPSA`, la funcion de fidelidad tambien consume presupuesto del contador de `qnspsa`.
- Por eso, con el mismo `budget_evals` nominal, `QNSPSA` puede agotar presupuesto efectivo antes que `COBYLA`.
