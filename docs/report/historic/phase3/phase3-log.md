# Phase 3 implementation log

Bitacora del proceso de implementacion de la **Fase 3** (nuevos optimizadores +
dinamica Layer-wise) y de la **infraestructura de metricas de gradiente**
(certificacion de barren plateaus).

## Objetivo y alcance de esta ronda

- Anadir un optimizador tipo Rotosolve (**NFT**, nativo de `qiskit_algorithms`) y
una dinamica de entrenamiento **Layer-wise Learning** al pipeline.
- Generalizar el codigo (experimentos + analisis) de 2 optimizadores fijos a **N**.
- Anadir metricas de gradiente reutilizables:
  - **Offline** (`exp-gradient-metrics.py`): Var(dC/dtheta) en la inicializacion
    (muchos theta_0) y decaimiento de norma en ventanas cortas de entrenamiento.
  - **Inline** (ideal + shot-noise): diagnostico de gradiente en los mismos runs
    de optimizacion, guardado en `optimizer_history.npz` junto al coste.
- Foco en **simulacion (ideal + shot-noise)**. No se lanza NFT/Layer-wise en  
hardware real todavia (grids en curso y coste alto de gradientes).

## Decisiones de diseno (y por que)

- **Registro de optimizadores** (`src/optimizers.py`): antes `"cobyla"`/`"qnspsa"`
estaban hardcodeados en 6 scripts y en todo el analisis. Un registro con
`OPTIMIZER_REGISTRY` permite N metodos sin duplicar y define color/label/necesidad
de fidelidad por optimizador.
- **NFT como equivalente de Rotosolve**: `qiskit_algorithms.optimizers.NFT` ya
implementa la minimizacion secuencial explotando la dependencia sinusoidal, asi
que se reutiliza en vez de programar Rotosolve a mano. Import protegido
(`try/except`) por si una version de qiskit-algorithms no lo trae.
- **Layer-wise como training-mode, no como optimizador**: es una medida sobre la
dinamica; se implementa como envoltorio (`src/layerwise.py`) que reutiliza un
optimizador interno del registro (sin fidelidad) sobre subconjuntos de
parametros enmascarados.
- **Metricas de gradiente solo en simulacion**: primitivas Qiskit exactas para
  cualquier estructura de puertas; en shot-noise el ruido viene de `precision`.
  En entrenamiento, el hook vive en `make_objective()` (`src/utils/utils.py`):
  en checkpoints fijos se llama al primitivo de gradiente **sin consumir budget**.
  `--track-params` sigue disponible para guardar theta en cada eval, pero ya no
  hace falta para la norma si se usa `--track-grad*`.
- **Compatibilidad hacia atras**: el analisis generico lee las claves que haya en
el `.npz`; los outputs antiguos (`cobyla`/`qnspsa`) siguen produciendo exactamente
las mismas columnas de CSV.



## Problemas encontrados y como se resolvieron

- **Blast radius del refactor de analisis**: `RunSummary` tenia campos fijos usados
en `utils.py` y `analyze_outputs.py`. Solucion: dicts por optimizador + propiedades
de compatibilidad (`cobyla_final_cost`, etc.) y columnas dinamicas. Se verifico que
para `cobyla,qnspsa` el esquema de CSV es identico al anterior.
- **Orden de binding de parametros en ResQNet**: los `ParameterVector` se enlazan en
orden de `circuit.parameters` (ordenado por nombre: `phi` < `theta1` < `theta2`).
Los grupos de capas se calcularon en ese layout, no en orden de insercion, y se
valido que cubren exactamente `0..p-1` en las 3 arquitecturas.
- **Fidelidad en Layer-wise**: QNSPSA necesita fidelidad sobre el espacio completo;
sobre el subespacio enmascarado es problematico. Se restringe el inner optimizer
a metodos sin fidelidad (cobyla/nft) con error explicito.
- **NFT en** `both` **de hardware**: para no cambiar el coste de los grids existentes,
`both` sigue siendo `cobyla+qnspsa`; `nft` es opt-in.
- `MAXFUN` **de COBYLA con budget pequeno**: si `budget < num_params + 2`, scipy
avisa. Solo ocurre con budgets muy bajos de prueba; en uso normal el budget es
mucho mayor.



## Como ejecutar y validar

Comparar los tres optimizadores en ideal:

```bash
bnd run --cpu python src/experiments/exp-ideal.py \
  --architecture resqnet --n_qubits 8 --seed 10 \
  --optimizers cobyla,qnspsa,nft --outdir outputs/test/seed_10/qubits_8
```

Layer-wise (inner NFT) en ideal:

```bash
bnd run --cpu python src/experiments/exp-ideal.py \
  --architecture resqnet --n_qubits 8 --seed 10 \
  --training-mode layerwise --layerwise-inner nft \
  --outdir outputs/test_lw/seed_10/qubits_8
```

Metricas de gradiente (Var vs qubits + decaimiento de norma):

```bash
bnd run --cpu python src/experiments/exp-gradient-metrics.py \
  --architectures baseline,qcnn,resqnet \
  --qubit-sizes 4,8,12,16 --n-samples 300 \
  --norm-sizes 4,8,12,16 --norm-seeds 10,20,30,40,50 \
  --norm-optimizers cobyla,qnspsa,nft --norm-budget 600 --norm-stride 10
```

Analisis (generico para N optimizadores):

```bash
bnd run --cpu python src/utils/analyze_outputs.py outputs/<run_root>
```



### Validaciones realizadas durante la implementacion

- `NFT` disponible en el entorno (`qiskit_algorithms.optimizers.NFT`).
- Analisis sobre un run antiguo (`..._phase2_ideal_resqnet`): CSVs con esquema
identico al previo (compatibilidad).
- Smoke ideal con `cobyla,qnspsa,nft` + `--track-params`: 3 columnas, `best_optimizer`
correcto, trayectoria de parametros guardada.
- Smoke Layer-wise (resqnet, inner nft): entrenamiento secuencial por capas y clave
`layerwise_nft` integrada en el analisis.
- Grupos de capas validados (cubren `0..p-1`) en baseline/qcnn/resqnet.
- Smoke del experimento de gradientes: Var por arquitectura (resqnet decae,
qcnn se mantiene) y norma por optimizador, con sus PNGs.



## Reorganizacion de `src/`

Tras la Fase 3 se eliminaron las carpetas `phase1/`, `phase2/` y `phase3/` en
favor de una estructura por proposito (se priorizaron los archivos de `phase2`,
los mas recientes; de `phase1` solo se conservo `analyze_outputs.py`):

```
src/
  hyperparams.py optimizers.py gradients.py layerwise.py
  utils/           architectures.py utils.py utils_runtime.py
                   analyze_outputs.py visualize_architectures.py
  experiments/     exp-ideal.py exp-shot-noise.py exp-real.py exp-gradient-metrics.py
  sbatches/        exp-*.sbatch submit.sh finalize_run_backup.sh
  notebooks/       local_real_hw_runner.ipynb
  circuits/        (diagramas estaticos, sin cambios)
```

Cambios asociados: imports `from phase2.architectures` -> `from architectures`;
sbatches apuntan a `src/experiments/...`; `submit.sh` y `finalize_run_backup.sh`
recalculan `REPO_ROOT` con `../..`. La convencion de nombres de salidas
(`outputs/<ts>_phase2_...`) se mantiene intacta.

## Correcciones posteriores (revision de NFT)

Tras revisar la logica y los resultados de NFT se detectaron y corrigieron los
siguientes puntos (fecha: 2026-07-04):

- **[1] Presupuesto de NFT igualado al del resto** (`src/optimizers.py`): la NFT de
`qiskit_algorithms` usa `maxfev=1024` por defecto y corta las evaluaciones a ~1024
aunque `maxiter` sea mayor. Con `--budget-k 37.5` (budget = 37.5*p) esto dejaba a
NFT con ~1024 evals mientras COBYLA/QNSPSA consumian hasta 37.5*p (2400-8400),
comparacion injusta. **Fix**: `NFT(maxiter=budget, maxfev=budget)`, de modo que
los tres optimizadores comparten el mismo tope de evaluaciones (el guard global
`BudgetExceeded` de `make_objective` sigue siendo el limite duro). Verificado:
con budget=120, NFT registra exactamente 120 evals.
- **[2] Nueva vista "best-so-far"** (`src/utils/utils.py`,
`save_optimizer_time_series`): NFT registra tambien sus sondas +-pi/2, asi que la
curva de coste cruda es un diente de sierra que no representa la convergencia. Se
**anade** (sin quitar las vistas previas) un tercer PNG
`optimizer_compare_best.png` con el minimo acumulado (`np.minimum.accumulate`),
que muestra el progreso monotono real.
- **[3]** `final_cost` **= ultima evaluacion vs solucion de NFT**: `parse_single_run`
toma `cost_series[-1]`, que para NFT puede ser una sonda +-pi/2 y no la solucion
devuelta (`OptimizeResult.x`/`.fun`). En resqnet infra-reporta a NFT
(p.ej. q8 s10: ultimo -0.941 vs mejor visto -0.995). Sin cambiar aun.
- **[4] Validez de la reconstruccion**: NFT es exacta solo para cortes de un unico
armonico (periodo 2*pi, generador con autovalores +-1/2). Se cumple en baseline
(RY/RZ), pero no estrictamente en qcnn (pooling con `cry` -> frecuencias {1/2, 1})
ni en resqnet (parametros `phi` compartidos y aplicados dos veces -> grado 2).
Empiricamente NFT converge igual (qcnn -> -1; resqnet -0.98/-0.999) pero mas lento
en resqnet; queda como caveat a documentar en la memoria.

> Nota: al cambiar el presupuesto efectivo de NFT ([1]), los grids de NFT ya
> ejecutados (`outputs/*phase3_*`) deberian relanzarse para que la comparacion sea
> homogenea con COBYLA/QNSPSA.



## Cambios en metricas de gradiente y renombrado (2026-07-04)

Segunda tanda de ajustes, tras revisar el experimento de metricas de gradiente:

- **Renombrado** `baseline_hea` **->** `baseline` en todo el codigo nuevo/activo:
`utils/architectures.py` (constante `BASELINE` y builder baseline),
`experiments/exp-{ideal,shot-noise,real}.py` (defaults de `--architecture`),
`utils/visualize_architectures.py`, los `sbatches/exp-*.sbatch` y `submit.sh`,
y `src/circuits/baseline_hea/` -> `src/circuits/baseline/` (con sus `summary.csv`).
**No** se tocaron runs historicos en `outputs/` (sus `run_status.json` conservan
`"architecture": "baseline_hea"` como registro). El label de salida ya era
`baseline`, asi que las carpetas `outputs/*_baseline` no cambian.
- **Var(grad) al inicio**: `--n-samples` por defecto **150 -> 300** (estimador de
varianza mas ajustado; error relativo ~= sqrt(2/(N-1)) ~ 8.2% con N=300).
- **Decaimiento de norma reescrito a barrido seed x size** (`exp-gradient-metrics.py`):
  - Nuevos flags `--norm-seeds` (default `10,20,30,40,50`) y `--norm-sizes`
  (default `4,8,12,16`); se elimina `--norm-qubits`.
  - Genera **una figura por (seed, size)** -> 5x4 = **20 PNGs**
  `grad_norm_decay_seed{S}_q{Q}.png`, cada una superponiendo arquitecturas x
  optimizadores. Payload JSON anidado `norm_decay[seed][size][arch][opt]`.
  - `--norm-stride` **5 -> 10**; `--norm-max-points` **40 -> 0 (ilimitado)**: solo
  el budget y el stride acotan el numero de checkpoints.
  - `--norm-budget` sigue en 600 (ventana de entrenamiento corta, no derivada de
  k*p; es un diagnostico de dinamica temprana, no una optimizacion completa).
  - Init de la trayectoria: `normal(0, 0.1)` (regimen mitigado tipo identity-init).
  NO se anadio el modo "peor caso vs mitigado" al sub-experimento 1: esa
  comparacion ya se hizo aparte y no es foco ahora.



### Gradiente: de parameter-shift manual a primitivas Qiskit (2026-07-05)

**No se cambio la funcion de coste** usada en optimizacion ni en metricas. Sigue siendo
la expectativa del observable local (Pauli Z en el qubit de lectura), evaluada con
`StatevectorEstimator` (ideal) o `MonteCarloEstimator` (shot-noise), igual que en
phase2.

Lo que se cambio para que la comparacion entre arquitecturas fuera **justa** fue el
**calculo del gradiente** en `src/gradients.py` (solo metricas de BP, no el motor de
los optimizadores):

| Regimen | Antes (sesgado en qcnn/resqnet) | Despues (exacto para cualquier puerta) |
|---|---|---|
| ideal | parameter-shift manual +-pi/2 | `ReverseEstimatorGradient` |
| shot-noise | parameter-shift manual +-pi/2 | `LinCombEstimatorGradient` + `precision` |

Motivo: el shift +-pi/2 es exacto solo en baseline (RY/RZ de un armonico). En **qcnn**
(`cry`) y **resqnet** (`phi` compartido) sesgaba Var(grad) y ||grad||. Las primitivas
de Qiskit aplican la regla correcta para cualquier estructura de circuito.

`exp-gradient-metrics.py` usa `--grad-method auto` (reverse en ideal, lincomb en
shot-noise). El coste que se minimiza en entrenamiento **no** usa estas primitivas;
COBYLA/QNSPSA/NFT optimizan el coste directamente.

### Caveat historico: parameter-shift manual (sustituido)

La regla antigua `dC/dtheta = 0.5*(C(theta+pi/2) - C(theta-pi/2))` era **exacta solo
para puertas de un unico armonico** (generador con autovalores +-1/2, periodo 2*pi):

- **baseline**: exacto (RY/RZ de un parametro por puerta).
- **qcnn**: el pooling usa `cry` -> gradiente sesgado con shift simple.
- **resqnet**: parametros `phi` compartidos -> gradiente sesgado con shift simple.

Esta regla manual fue **reemplazada** por las primitivas Qiskit anteriores. El mismo
limite estructural sigue afectando a **NFT** como optimizador (no al calculo de
gradientes en metricas).

## Shot-noise en metricas de gradiente + budget k*p (2026-07-05)

- **Regimen de ruido configurable** (`exp-gradient-metrics.py`): nuevo
`--noise-modes` (default `ideal,shot-noise`). Se ejecutan ambos sub-experimentos
(Var y decaimiento de norma) en cada regimen y se guardan en subcarpetas por modo:
`outputs/<ts>_phase3_grad-metrics/{ideal,shot-noise}/`.
  - `ideal`: `StatevectorEstimator` (exacto), como hasta ahora.
  - `shot-noise`: `MonteCarloEstimator` (version *batched*) que muestrea el readout con
  `shots ~= 1/precision^2` (`--target-precision`, default 0.1 -> ~100 shots).
  El gradiente usa `LinCombEstimatorGradient` (regla exacta; el ruido viene de
  `precision`). El gradiente ve el suelo de ruido estadistico que enmascara senales
  exponencialmente pequenas (verificado: en shot-noise la Var(grad) se aplana).
- **Budget k*p en decaimiento de norma**: se elimina el `--norm-budget=600` fijo;
ahora `--norm-budget` es override opcional (0 = dinamico) y `--norm-budget-k`
(default 37.5) fija `budget = round(k * num_params)` **por arquitectura**, misma
regla que los grids de entrenamiento. El circuito shot-noise se pasa
`decompose()` (Statevector necesita blueprint expandido).
- **sbatch** actualizado con `NOISE_MODES`, `TARGET_PRECISION`, `NORM_BUDGET_K`.

> Nota de coste: con budget k*p (hasta 37.5*p), stride 10 y max-points ilimitado,
> el numero de checkpoints de gradiente crece con p; en shot-noise ademas cada
> evaluacion samplea binomiales. Para tamanos grandes (q16) puede ser pesado; se
> puede acotar con `--norm-stride`/`--norm-max-points` o reduciendo `--norm-budget-k`.



## Ejecucion paralela de metricas de gradiente (2026-07-05)

El experimento de metricas de gradiente deja de ser un job monolitico y pasa a
**array job de Slurm** con tareas independientes:

- **42 tareas** con la config por defecto:
  - 2 tareas de varianza (una por modo: `ideal`, `shot-noise`): Var(grad) vs qubits,
  todas las arquitecturas en la misma figura.
  - 40 tareas de decaimiento de norma (2 modos x 5 semillas x 4 tamanos): una figura
  por `(seed, qubits)` con arquitecturas x optimizadores superpuestos.
- Cada tarea escribe su resultado parcial en
`{outdir}/{noise_mode}/partial/*.json` y su PNG correspondiente.
- Tras completar el array, un job de **merge** agrega todo en `grad_metrics.json`.

Archivos:


| Archivo                                    | Que hace                                                              |
| ------------------------------------------ | --------------------------------------------------------------------- |
| `src/sbatches/exp-gradient-metrics.sbatch` | Array job (`--array=1-N`); cada indice mapea a varianza o norm-decay. |
| `src/sbatches/submit_grad_metrics.sh`      | Calcula N, lanza el array y el merge con `--dependency=afterok`.      |
| `src/sbatches/merge_grad_metrics.sbatch`   | Ejecuta `--aggregate-only` sobre el run root compartido.              |


Flags nuevos en `exp-gradient-metrics.py`:

- `--noise-mode ideal|shot-noise`: una sola tarea de un modo (array).
- `--norm-seed S --norm-qubits Q`: una sola tarea de norm-decay.
- `--aggregate-only --outdir PATH`: fusiona los partial JSON.

Lanzar:

```bash
bash src/sbatches/submit_grad_metrics.sh
```

Modo monolitico (local, sin array) sigue disponible omitiendo `--noise-mode` /
`--norm-seed` / `--norm-qubits`.

## Metricas de gradiente inline en entrenamiento (2026-07-12)

Las metricas de gradiente pasan de ser solo un experimento aparte
(`exp-gradient-metrics.py`) a integrarse en los runs de simulacion
(`exp-ideal.py`, `exp-shot-noise.py`, y `layerwise` via los mismos flags).

### Principio

El optimizador **solo minimiza el coste** (igual que antes). En paralelo, en
ciertos pasos se calcula el gradiente completo con las primitivas de Qiskit
(`ReverseEstimatorGradient` en ideal, `LinCombEstimatorGradient` en shot-noise).
Esa llamada **no cuenta** hacia `budget_evals`.

En cada evaluacion `k` del optimizador:

1. `estimator.run` -> `C(theta_k)` (cuenta para budget; se apenda a `cost[]`).
2. Si `k` es checkpoint de gradiente: `gradient.run` -> `grad C(theta_k)`
   (diagnostico extra; no cuenta para budget).

**Checkpoints**: evaluacion 1 y luego cada `--grad-stride` evals (default
`SIM_GRAD_CHECKPOINT_STRIDE = 10` en `hyperparams.py` -> evals 1, 11, 21, ...).

### Que se guarda en `optimizer_history.npz`

Por optimizador, ademas de `evals` y `cost`:

| Clave | Significado |
| ----- | ----------- |
| `grad_checkpoint_evals` | Indice de evaluacion en el que se midio (p. ej. `[1, 11, 21, ...]`) |
| `grad_norm` | `||grad C||_2` en ese checkpoint |
| `grad_var_components` | `Var_i(dC/dtheta_i)` **entre las componentes de un solo vector** en ese `theta_t` (`np.var(grad)` en `gradient_vector_stats`) |

Los PNG de serie temporal incluyen, si hay datos:
`optimizer_compare_grad_norm.png` y `optimizer_compare_grad_var.png`.

### Flags CLI: `--track-grad`, `--track-grad-norm`, `--track-grad-var`

Los tres flags controlan **que escalares se escriben** a partir del **mismo**
vector `grad C` en cada checkpoint. El coste extra (una llamada al primitivo
de gradiente por checkpoint) es el mismo en todos los casos.

| Flag | Efecto |
| ---- | ------ |
| `--track-grad` | Atajo: activa norma **y** varianza entre componentes |
| `--track-grad-norm` | Solo guarda `grad_norm` |
| `--track-grad-var` | Solo guarda `grad_var_components` |

Implementacion en `exp-ideal.py` / `exp-shot-noise.py`:

```python
track_grad_norm = args.track_grad_norm or args.track_grad
track_grad_var = args.track_grad_var or args.track_grad
```

Es decir: `--track-grad` equivale a `--track-grad-norm` + `--track-grad-var`.

**Importante — no confundir `grad_var_components` con Var(grad) de barren plateau:**

- `grad_var_components` (inline): un solo `theta` (el del optimizador en ese paso);
  mide si las componentes del vector de gradiente son parecidas o desiguales.
- Var(grad) clasica (`exp-gradient-metrics.py`): Monte Carlo sobre **muchos**
  `theta_0` aleatorios; certifica el decaimiento exponencial con `n` qubits.

### Archivos tocados

| Archivo | Cambio |
| ------- | ------ |
| `src/gradients.py` | `gradient_vector_stats()`, `build_simulation_gradient()` |
| `src/utils/utils.py` | `_is_grad_checkpoint`, `_record_gradient_checkpoint`, hook en `make_objective`, PNGs de gradiente en `save_optimizer_time_series` |
| `src/layerwise.py` | Propaga flags de gradiente al entrenamiento por capas |
| `src/experiments/exp-ideal.py`, `exp-shot-noise.py` | Flags CLI y wiring |
| `src/sbatches/exp-ideal.sbatch`, `exp-shot-noise.sbatch` | `TRACK_GRAD`, `TRACK_GRAD_NORM`, `TRACK_GRAD_VAR`, `GRAD_STRIDE` |

`exp-real.py` **no** incluye tracking inline (hardware real fuera de alcance por ahora).

### Como ejecutar

Ideal (smoke):

```bash
bnd run --cpu python src/experiments/exp-ideal.py \
  --architecture baseline --n_qubits 8 --seed 10 \
  --optimizers cobyla,qnspsa,nft \
  --track-grad --grad-stride 10 \
  --outdir outputs/test_grad/seed_10/qubits_8
```

Slurm:

```bash
TRACK_GRAD=1 GRAD_STRIDE=10 sbatch src/sbatches/exp-ideal.sbatch
```

Solo norma o solo varianza entre componentes:

```bash
--track-grad-norm          # solo ||grad C||_2
--track-grad-var           # solo Var_i entre componentes
```

El experimento offline `exp-gradient-metrics.py` **sigue existiendo** para el
barrido sistematico (Var vs qubits, grids seed x size x modo de ruido). Los
runs de entrenamiento con `--track-grad*` alinean coste y senal de gradiente
en el mismo `.npz`, listos para analisis de fase 4 sin un job post-hoc.

## Eliminacion de early stopping (2026-07-12)

Desde phase2 el early stopping por ventana de coste (`--window` / `--tolerance`)
estaba **desactivado** en `with_early_stopping()` (no-op), pero el codigo muerto
seguia en hyperparams, CLI, `exp-real.sbatch` y `src/utils/utils.py`. En esta ronda
se **elimina por completo**:

- Constantes `EARLY_STOPPING_*` y alias `SIM_*` / `REAL_HW_*` en `hyperparams.py`.
- `ConvergenceReached`, `with_early_stopping()` y los flags `--window` /
  `--tolerance` en `exp-ideal.py`, `exp-shot-noise.py` y `exp-real.py`.
- Parametros `window` / `tolerance` de `run_optimizer_suite()`.
- Manejo de `ConvergenceReached` en `layerwise.py` y `run_optimizer()`.

**Criterio de parada unico**: el presupuesto compartido de evaluaciones
(`budget_evals = k * p` o override fijo). `make_objective()` lanza
`BudgetExceeded` al superar el tope; los optimizadores no paran antes por
convergencia heuristica.

## Estado final y pendiente para fases 4-6

- **Hecho**: registro N-optimizadores, NFT, Layer-wise, metricas de gradiente
  (offline + **inline en simulacion**), analisis generico, eliminacion de early
  stopping, renombrado `baseline_hea` -> `baseline`, docs de arquitectura (QCNN/ResQNet).
- **Pendiente (siguiente ronda)**:
  - Lanzar grids de simulacion con NFT, Layer-wise y `TRACK_GRAD=1`, y analizarlos.
  - Evaluar (si procede) NFT en hardware real.
  - **Fase 4**: encoding de datos + loss supervisada + datasets (MNIST reducido,
  Breast Cancer, Local Translation, Global Parity) y metrica de accuracy.
  - **Fase 5-6**: validacion en datasets reales y redaccion continua de la memoria.
  - Reescritura de `README.md` (dejada al final).

