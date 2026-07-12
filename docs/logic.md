# Budget Eval Logic

Este documento resume los valores de `budget_evals` por combinacion de qubits para cada script.

## Regla usada

- Modo dinamico: `budget_evals = round(k * p)`.
- `p` es `ansatz.num_parameters`.
- Para este ansatz (`efficient_su2` con `reps = round(log2(n_qubits))` y entanglement lineal), aqui se usa:
  - `p = 2 * n_qubits * (reps + 1)`.
- `resolve_budget_evals()` aplica un piso minimo de 1 evaluacion (irrelevante en la practica).

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

## Tabla 3 - `src/experiments/exp-real.py` (k por defecto = 20, dinamico)

Default en `hyperparams.py`: `REAL_HW_DEFAULT_BUDGET_K = 20` (menor que simulacion
por limites de tiempo y coste en IBM Runtime).

Arquitectura `baseline` (`efficient_su2`):

| n_qubits | reps | p (num_params) | budget_evals por defecto |
|---|---:|---:|---:|
| 4 | 2 | 24 | 480 |
| 8 | 3 | 64 | 1280 |
| 12 | 4 | 120 | 2400 |
| 16 | 4 | 160 | 3200 |
| 20 | 4 | 200 | 4000 |

## Nuevas arquitecturas (Phase 2): parametros y budget dinamico

Esta seccion resume el numero de parametros `p` para `qcnn` y `resqnet`, y el
`budget_evals` dinamico por defecto:

- Simulacion (`exp-ideal.py` / `exp-shot-noise.py`): `k = 37.5`
- Real hardware (`exp-real.py`): `k = 20`

### QCNN (`architecture = qcnn`)

Valido para qubits potencia de 2 (`4, 8, 16, ...`).

En esta implementacion:

- `levels = log2(n_qubits)`
- `conv_blocks = 2*n_qubits - 2 - log2(n_qubits)`
- `pool_blocks = n_qubits - 1`
- `p = 3*conv_blocks + 2*pool_blocks = 8*n_qubits - 8 - 3*log2(n_qubits)`

| n_qubits | levels | p (num_params) | budget sim (k=37.5) | budget real-hw (k=20) |
|---:|---:|---:|---:|---:|
| 4  | 2 | 18  | 675  | 360  |
| 8  | 3 | 47  | 1762 | 940  |
| 16 | 4 | 108 | 4050 | 2160 |
| 32 | 5 | 233 | 8738 | 4660 |

Nota: para `1762.5` se aplica `round(...)`, dando `1762`.

### ResQNet (`architecture = resqnet`)

Con `depth_split = (D1, D2)` y modo residual estructural:

- `shared_params = 2*n_qubits`
- `qn1_params = 2*n_qubits*D1`
- `qn2_params = 2*n_qubits*D2`
- `p = 2*n_qubits*(1 + D1 + D2)`

Para el default actual `depth_split = (5,1)`:

- `p = 14*n_qubits`

| n_qubits | D1,D2 | p (num_params) | budget sim (k=37.5) | budget real-hw (k=20) |
|---:|:---:|---:|---:|---:|
| 4  | 5,1 | 56  | 2100 | 1120 |
| 8  | 5,1 | 112 | 4200 | 2240 |
| 16 | 5,1 | 224 | 8400 | 4480 |
| 32 | 5,1 | 448 | 16800 | 8960 |

Importante:

- Si `D1 + D2` se mantiene constante, `p` tambien se mantiene.
- Ejemplo: `(5,1)`, `(4,2)` y `(3,3)` tienen el mismo numero de parametros.

## Perfil de ejecucion real-hw en Slurm

Launcher: `src/sbatches/exp-real.sbatch` → `src/experiments/exp-real.py`.

**Budget por defecto:** dinamico `round(k * p)` con el mismo `k = 20` del script
(no hay formula fija exponencial en el sbatch). Overrides opcionales al lanzar:

- `BUDGET_K=<float>` → pasa `--budget-k` (p. ej. cap mas bajo en pilotos q=32).
- `BUDGET_EVALS=<int>` → pasa `--budget-evals` (modo fijo).

**Grids por arquitectura** (si no se define `QUBITS_OVERRIDE`):

| Arquitectura | Qubits en array |
|--------------|-----------------|
| `baseline` | 4, 8, 12, 16, 20 |
| `qcnn`, `resqnet` | 4, 8, 16 |

**Modo IBM Runtime:** `EXECUTION_MODE=batch` por defecto (1 Batch IBM por tarea
Slurm = 1 punto `(seed, qubits)`). `SKIP_COMPLETED=1` reanuda sin repetir puntos
con `run_status.json` completado.

**q=32 (opcional):** el sbatch puede incluir q=32 via `QUBITS_OVERRIDE`; en ese
caso reparte COBYLA y QNSPSA en tareas Slurm separadas para reducir riesgo de TTL.
Los budgets de la tabla anterior para q=32 son orientativos; en pilotos HW suele
acotarse con `BUDGET_EVALS` o `BUDGET_K` hasta validar tiempos.

## Overrides por CLI

- En los 3 scripts, si pasas `--budget-evals N`, se usa modo fijo con ese `N`.
- Si no pasas `--budget-evals`, se usa modo dinamico con `--budget-k` (o su default del script).

## Nota importante

- En `QNSPSA`, la funcion de fidelidad tambien consume presupuesto del contador de `qnspsa`.
- Por eso, con el mismo `budget_evals` nominal, `QNSPSA` puede agotar presupuesto efectivo antes que `COBYLA`.

## Fase 3 - Optimizadores y dinamica de entrenamiento

Los optimizadores estan centralizados en `src/optimizers.py` (`OPTIMIZER_REGISTRY`).
El codigo de experimentos y de analisis es generico para N optimizadores.

- Optimizadores disponibles: `cobyla`, `qnspsa`, `nft`.
- `nft` (Nakanishi-Fujii-Todo) es el equivalente en Qiskit de Rotosolve:
  minimiza cada parametro explotando la dependencia sinusoidal del coste, sin
  calcular el vector gradiente.
- Solo `qnspsa` necesita callback de fidelidad (`needs_fidelity=True`).

### Seleccion de optimizadores

- Simulacion (`exp-ideal.py`, `exp-shot-noise.py`): `--optimizers cobyla,qnspsa,nft`
  (por defecto los tres). En los sbatch, variable `OPTIMIZERS`.
- Hardware real (`exp-real.py`): `--optimizer both|cobyla|qnspsa|nft`. `both` sigue
  siendo `cobyla+qnspsa`; `nft` es opt-in para no cambiar el coste de los grids.

### Layer-wise Learning

- Dinamica de entrenamiento (no un optimizador): entrena las capas de forma
  secuencial congelando las anteriores. `src/layerwise.py`.
- Activar con `--training-mode layerwise` y `--layerwise-inner cobyla|nft`
  (el optimizador interno no puede requerir fidelidad).
- Los grupos de capas por arquitectura estan en `ArchitectureSpec.layer_param_indices`
  (baseline: bloques de rotacion `2n`; qcnn: por nivel conv+pool; resqnet: bloque
  compartido + capas de QN1 + capas de QN2).
- El presupuesto total se reparte por igual entre capas; el historial se guarda
  bajo una unica clave `layerwise_<inner>`.

## Fase 3 - Metricas de gradiente (certificacion de BPs)

Modulo reutilizable `src/gradients.py` (regla parameter-shift, exacto en
statevector):

- `gradient_variance_at_init(...)`: `Var(dC/dtheta_i)` sobre N vectores aleatorios
  en la inicializacion. Un barren plateau se manifiesta como decaimiento
  exponencial de esta varianza con el numero de qubits.
- `gradient_norm_trajectory(...)`: `||grad C||_2` a lo largo de la trayectoria de
  optimizacion (usa los parametros guardados con `--track-params`).
- Experimento: `src/experiments/exp-gradient-metrics.py` (+ sbatch en
  `src/sbatches/`). Genera `grad_var_vs_qubits.png` y `grad_norm_decay.png`.
