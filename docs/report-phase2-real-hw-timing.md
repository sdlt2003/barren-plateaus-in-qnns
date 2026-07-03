# Informe hardware real — Phase 2 (IBM)

**Backend:** `ibm_basquecountry` (IBM Quantum Heron, 156 qubits)  
**Última actualización:** 2026-06-13  
**Datos en:** `outputs/20260608-011456_phase2_real-hw_baseline/`, `outputs/20260611-212545_phase2_real-hw_qcnn/`

Este documento recoge **estado del grid**, **resultados de optimización**, **tiempos Slurm** y **fallos** de los experimentos phase2 en hardware real. Los agregados numéricos salen de `src/phase1/analyze_outputs.py` (carpetas `analysis/` de cada run).

---

## Resumen ejecutivo

| Run | Arquitectura | Modo IBM | Job Slurm | Grid | Completados | Fallidos | Pendientes |
|-----|--------------|----------|-----------|------|-------------|----------|------------|
| `20260608-011456_*` | `baseline_hea` | **Session** (1×/seed) | **1208546** | 5×5 = **25** | **19** | **5** | **1** (s40 q=20) |
| `20260611-212545_*` | `qcnn` | **Batch** (1×/punto) | **1229150** | 5×4 = **20** | **11** | **0** | **9** |

**Total puntos con artefactos:** 35 carpetas con `run_status.json` (24 baseline + 11 qcnn).

**Lectura rápida:**

- **Baseline:** grid casi cerrado; los 5 fallos son q=20 (sesión cerrada ×3), q=16 en seeds 30 y 40 (error IBM / job cancelado). Seed 50 completó los 5 q.
- **QCNN (batch):** sin fallos hasta ahora; **todos los q=4** hechos; q=8 casi completo (falta seed 10); q=16 empezado (2/5); q=32 sin empezar.
- **COBYLA** converge en HW en la mayoría de puntos; **QNSPSA** suele quedarse con coste positivo en q bajos y mejora en algunos q=16 (baseline y qcnn).

---

## Configuración por run

### Baseline — `20260608-011456_phase2_real-hw_baseline`

| Parámetro | Valor |
|-----------|--------|
| Slurm | `#SBATCH --array=1-5%1` → **un seed a la vez** |
| IBM | **1 Session por seed**, q ∈ {4, 8, 12, 16, 20} en serie |
| `SESSION_MAX_TIME` | **8 h** (explícito en logs del job) |
| Timeout cliente | desactivado |
| Budget | `32 × 2^((Q−4)/4)` evals / optimizador |
| Shots | 1024 |

### QCNN — `20260611-212545_phase2_real-hw_qcnn`

| Parámetro | Valor |
|-----------|--------|
| Slurm | `#SBATCH --array=1-25%8` → **hasta 8 puntos en paralelo** |
| IBM | **1 Batch por (seed, q)** (`EXECUTION_MODE=batch`) |
| `RUNTIME_MAX_TIME` | default IBM (no fijado en cliente) |
| Timeout cliente | desactivado |
| Qubits | {4, 8, 16, 32} |
| Budget | misma fórmula (q=32 → **4096** evals/opt) |
| `SKIP_COMPLETED` | activo |

---

## Budget y parámetros por q

| q | budget_evals/opt | baseline params | qcnn params |
|---|------------------|-----------------|-------------|
| 4 | 32 | 24 | 18 |
| 8 | 64 | 64 | 47 |
| 12 | 128 | 120 | — |
| 16 | 256 | 160 | 108 |
| 20 | 512 | 200 | — |
| 32 | 4096 | — | 233 |

---

## Grid baseline — estado detallado

**Carpeta:** `outputs/20260608-011456_phase2_real-hw_baseline/`  
**Análisis:** `analysis/summary_per_run.csv` (21 puntos parseables; 3 omitidos por historial vacío en fallos tempranos)

| seed | q=4 | q=8 | q=12 | q=16 | q=20 |
|------|-----|-----|------|------|------|
| 10 | OK | OK | OK | OK | **FAIL** 1217 |
| 20 | OK | OK | OK | OK | **FAIL** 1217 |
| 30 | OK | OK | OK | **FAIL** 9705 | OK |
| 40 | OK† | OK† | OK | **FAIL** cancel | — |
| 50 | OK | OK | OK | OK | **FAIL** 1217 |

† Seed 40: q=4 y q=8 tienen artefactos recientes (re-ejecución parcial del job `1208546_4`); q=20 nunca arrancó.

### Fallos baseline (detalle)

| seed | q | Error | Progreso guardado |
|------|---|-------|-------------------|
| 10 | 20 | **1217** Session closed | COBYLA 512/512; QNSPSA 21/512 |
| 20 | 20 | **1217** Session closed | COBYLA 470/512 |
| 30 | 16 | **9705** IBM internal | COBYLA 75/256 |
| 40 | 16 | Job **cancelled** | COBYLA 17/256 |
| 50 | 20 | **1217** Session closed | COBYLA 512/512; QNSPSA 15/512 |

**Causa dominante en q=20:** TTL de sesión (~8 h) tras acumular q=4…16 en la misma sesión.

### Tiempos Slurm por seed (job 1208546)

| Task | Seed | Elapsed | Fin aprox. |
|------|------|---------|------------|
| `_1` | 10 | **12 h 49 min** | 2026-06-08 14:03 |
| `_2` | 20 | **16 h 22 min** | 2026-06-09 06:25 |
| `_3` | 30 | **8 h 10 min** | 2026-06-09 14:36 |
| `_5` | 50 | **8 h 15 min** | 2026-06-11 21:08 |
| `_4` | 40 | PENDING / reintentos | — |

### Tiempos por q (seed 10, mtimes artefactos)

| q | Fin (local) | Δ desde q anterior |
|---|-------------|-------------------|
| 4 | 2026-06-08 06:26 | ~5 h 12 min |
| 8 | 07:12 | ~46 min |
| 12 | 08:31 | ~1 h 19 min |
| 16 | 10:49 | ~2 h 19 min |
| 20 | 14:03 (fallo) | ~3 h 14 min |

---

## Resultados baseline — costes finales

Agregado (`analysis/summary_aggregated.csv`, solo puntos parseados):

| q | n | COBYLA final (mean±std) | QNSPSA final (mean±std) | Victorias C/Q |
|---|----|-------------------------|-------------------------|---------------|
| 4 | 5 | −0.60 ± 0.26 | +0.92 ± 0.03 | 5 / 0 |
| 8 | 5 | −0.38 ± 0.34 | +0.98 ± 0.02 | 5 / 0 |
| 12 | 5 | −0.67 ± 0.17 | +0.58 ± 0.35 | 5 / 0 |
| 16 | 3 | −0.86 ± 0.07 | −0.48 ± 0.07 | 3 / 0 |
| 20 | 3 | −0.74 ± 0.14 | +0.38 ± 0.81 | 3 / 0 |

**Destacable:** en q=16 (solo seeds 10, 20, 50 completos) QNSPSA ya alcanza coste negativo (−0.43…−0.58). En q=20 solo seed 30 completó ambos optimizadores con QNSPSA competitivo (−0.76 vs −0.78 COBYLA).

---

## Grid QCNN — estado detallado

**Carpeta:** `outputs/20260611-212545_phase2_real-hw_qcnn/`  
**Análisis:** `analysis/summary_per_run.csv` (11 puntos)

| seed | q=4 | q=8 | q=16 | q=32 |
|------|-----|-----|------|------|
| 10 | OK | pend. | OK | pend. |
| 20 | OK | OK | pend. | pend. |
| 30 | OK | OK | pend. | pend. |
| 40 | OK | OK | pend. | pend. |
| 50 | OK | OK | OK | pend. |

**Progreso:** **11 / 20** completados (55 %). **0 fallos.**

### Tiempos Slurm por tarea (job 1229150, puntos completados)

| Task | (seed, q) | Elapsed Slurm |
|------|-----------|---------------|
| 1 | (10, 4) | 21 h 40 min |
| 2 | (20, 4) | 22 h 52 min |
| 3 | (30, 4) | 23 h 28 min |
| 4 | (40, 4) | **1 d 0 h 7 min** |
| 5 | (50, 4) | **1 d 2 h 23 min** |
| 7 | (20, 8) | **1 d 1 h 7 min** |
| 8 | (30, 8) | **1 d 3 h 42 min** |
| 9 | (40, 8) | 8 h 1 min |
| 10 | (50, 8) | 7 h 41 min |
| 11 | (10, 16) | 9 h 29 min |
| 15 | (50, 16) | 2 h 44 min |

**Mapeo tarea → (seed, q)** (qcnn, qubit-major):

| Tareas | q | seeds |
|--------|---|-------|
| 1–5 | 4 | 10, 20, 30, 40, 50 |
| 6–10 | 8 | 10, 20, 30, 40, 50 |
| 11–15 | 16 | 10, 20, 30, 40, 50 |
| 16–20 | 32 | 10, 20, 30, 40, 50 |

El array Slurm es `1-25%8`; para qcnn solo hay **20** puntos — las tareas **21–25** salen al instante (`TOTAL_TASKS=20`).

**Pendientes en cola Slurm (2026-06-13):**

| Job | Tarea | Punto | Motivo |
|-----|-------|-------|--------|
| 1229150 | 6 | s10 q8 | Priority |
| 1229150 | 12–14 | s20–40 q16 | Nodes DOWN/DRAINED o Resources |
| 1229150 | 16–20 | q32 (todos) | Resources |
| 1208546 | 4 | s40 (sesión completa) | Priority |

**Nota:** con **8 batches en paralelo** sobre el mismo backend, q=4 tardó **~22–26 h** por punto (vs ~5 h el primer q=4 baseline en sesión dedicada sin competencia).

---

## Resultados QCNN — costes finales

| seed | q | COBYLA final | QNSPSA final | best (analyze) | Ganador |
|------|---|--------------|--------------|----------------|---------|
| 10 | 4 | −0.94 | +0.94 | cobyla | COBYLA |
| 20 | 4 | −0.93 | +0.98 | cobyla | COBYLA |
| 30 | 4 | −0.97 | +0.98 | cobyla | COBYLA |
| 40 | 4 | −0.90 | +0.97 | cobyla | COBYLA |
| 50 | 4 | −0.82 | +0.99 | cobyla | COBYLA |
| 20 | 8 | −0.64 | +0.98 | cobyla | COBYLA |
| 30 | 8 | −0.95 | +0.97 | cobyla | COBYLA |
| 40 | 8 | −0.94 | +0.98 | cobyla | COBYLA |
| 50 | 8 | −0.92 | +0.91 | cobyla | COBYLA |
| 10 | 16 | −0.83 | **−0.92** | **qnspsa** | **QNSPSA** |
| 50 | 16 | −0.75 | −0.53 | cobyla | COBYLA |

Agregado QCNN (`summary_aggregated.csv`):

| q | n | COBYLA (mean) | QNSPSA (mean) | Victorias |
|---|----|---------------|---------------|-----------|
| 4 | 5 | −0.91 | +0.97 | 5 / 0 |
| 8 | 4 | −0.86 | +0.96 | 4 / 0 |
| 16 | 2 | −0.79 | −0.72 | 1 / 1 |

**Primer punto donde QNSPSA gana en HW:** qcnn seed 10, q=16.

---

## Comparación Session vs Batch

| Aspecto | Baseline (session/seed) | QCNN (batch/punto) |
|---------|-------------------------|---------------------|
| TTL | 8 h **acumulado** en un seed → fallos q=20 | **Aislado** por (seed,q) |
| Paralelismo Slurm | `%1` serial | `%8` hasta 8 puntos |
| Tiempo q=4 | ~5 h (1.er q, sin competencia) | ~22–26 h (8 jobs paralelos) |
| Fallos por sesión | Sí (1217) | Ninguno aún |
| Coste IBM | Tiempo de reloj de sesión | Solo segundos cuánticos (batch) |

---

## Runs históricos (referencia)

| Carpeta | Job | Modo | Timeout cliente | Resultado |
|---------|-----|------|-----------------|-----------|
| `20260607-185519_*` | 1208416 | 1 job / (seed,q) | 120 s | Casi todo falló; 2 éxitos |
| `20260608-001651_*` | 1208532 | 1 sesión/seed | 600 s | 0/5 q OK |
| `20260608-011456_*` | **1208546** | 1 sesión/seed | ninguno | **19/25** OK |
| `20260611-212545_*` | **1229150** | **1 batch/(seed,q)** | ninguno | **11/20** OK (en curso) |

Los runs de **~2 min** del grid `20260607` son **timeout cliente**, no duración real del experimento.

---

## Pendiente / reintentos sugeridos

### Baseline (6 puntos)

- seed 10, 20, 50 — **q=20** (re-lanzar con batch o sesión solo para q=20)
- seed 30 — **q=16**
- seed 40 — **q=16**, **q=20**

### QCNN (9 puntos)

- seed 10 — **q=8**, **q=32**
- seeds 20, 30, 40 — **q=16**, **q=32**
- seed 50 — **q=32**

Comando de reintento (misma carpeta, skip completed):

```bash
ARCHITECTURE=qcnn sbatch src/phase2/sbatches/exp-real.sbatch
```

---

## Dónde mirar en el repo

| Fuente | Contenido |
|--------|-----------|
| `outputs/<run>/seed_*/qubits_*/run_status.json` | `completed` / `failed_runtime`, `execution_mode` |
| `outputs/<run>/analysis/summary_*.csv` | Agregados COBYLA vs QNSPSA |
| `outputs/<run>/analysis/final_cost_vs_qubits*.png` | Boxplots |
| `outputs/logs/p2_real_hw_M_<job>_<task>.out` | `Total runtime:` (s) al terminar |
| `sacct -j <JOBID>` | Elapsed Slurm por tarea |

```bash
# Regenerar análisis
bnd run --cpu python src/phase1/analyze_outputs.py 20260608-011456_phase2_real-hw_baseline
bnd run --cpu python src/phase1/analyze_outputs.py 20260611-212545_phase2_real-hw_qcnn

# Estado cola
squeue -u $USER
sacct -j 1208546,1229150 --format=JobID,JobName,State,Elapsed
```

---

## Referencias

- Budget: `[logic.md](logic.md)`
- Ideal + shot-noise (completado): `[report-phase2-ideal-shot-noise.md](report-phase2-ideal-shot-noise.md)`
- Launcher HW: `src/phase2/sbatches/exp-real.sbatch`
- Script: `src/phase2/exp-real.py`, `src/utils_runtime.py`
- Plan de trabajo: `TMP_PLAN.md`
