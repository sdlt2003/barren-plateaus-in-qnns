# Tiempos de ejecución — hardware real (IBM)

**Backend:** `ibm_basquecountry`  
**Arquitectura de referencia:** `baseline_hea`  
**Última actualización:** 2026-06-08

Este documento resume **cuánto tarda en arrancar** un job (Slurm + IBM) frente a **cuánto tarda en completarse** el punto `(seed, qubits)`, con datos de los runs en `outputs/`.

---

## Cómo está organizado el grid


| Capa            | Qué hace                                                                                             |
| --------------- | ---------------------------------------------------------------------------------------------------- |
| **Slurm**       | Array `1-5%1` → **un seed a la vez** (seeds 10…50). Partición `CPU-LARGE`, límite **7 días**.        |
| **Script**      | `exp-real.py` abre **una IBM Session por seed** y recorre `q ∈ {4,8,12,16,20}` en serie.             |
| **IBM Runtime** | Cada llamada al `Estimator` ≈ **1 job** en la cola de IBM (~segundos–minutos de espera + ejecución). |
| **Budget**      | Fijo por q (no el k=37.5 de simulación): `budget_evals = 32 × 2^((Q−4)/4)` → ver tabla abajo.        |


Parámetros relevantes (`exp-real.sbatch` / `hyperparams.py`):

- `RUNTIME_RESULT_TIMEOUT` = vacío → **sin timeout cliente** (espera lo que tarde IBM).
- `SESSION_MAX_TIME` = **8 h** por sesión IBM.
- `shots` = 1024, early stopping desactivado en el launcher actual.

---

## Dos “relojes” distintos

### 1. Cola Slurm (¿cuándo empieza la tarea?)

Con `%1`, el seed *N* no arranca hasta que termina el seed *N−1*.


| Job           | Seed  | Submit           | Start Slurm      | Espera en cola Slurm                |
| ------------- | ----- | ---------------- | ---------------- | ----------------------------------- |
| 1208546_1     | 10    | 2026-06-08 01:14 | 2026-06-08 01:14 | **~0** (inmediato)                  |
| 1208546_[2-5] | 20–50 | 2026-06-08 01:14 | —                | **PENDING** hasta que acabe seed 10 |


En el grid antiguo (**un job Slurm por (seed, q)**), la espera acumulada era mayor. Ejemplo job `1208416_19` (seed 40, q=16):

- Submit del array: **18:55:18**
- Start de esa tarea: **19:25:40** → **~30 min** esperando tareas anteriores del array serial.

### 2. Ejecución real (¿cuánto tarda el punto una vez arrancado?)

Aquí entran: preflight IBM, transpile, cola del backend, COBYLA + QNSPSA hasta budget.

**Medida fiable:** timestamp de `optimizer_history.npz` (fin del punto) o `Total runtime:` en el log cuando el run terminó bien.

---

## Budget por qubits (baseline real-hw)


| q   | budget_evals | p (params) |
| --- | ------------ | ---------- |
| 4   | 32           | 24         |
| 8   | 64           | 64         |
| 12  | 128          | 120        |
| 16  | 256          | 160        |
| 20  | 512          | 200        |


Cada evaluación de objetivo en hardware ≈ **1 job IBM**. El tiempo escala con **budget × (cola + shots + profundidad del circuito)**.

---

## Mediciones: run actual (recomendado)

**Carpeta:** `outputs/20260608-011456_phase2_baseline-real-hw/`  
**Job Slurm:** `1208546_1` (seed 10, sesión única, sin timeout cliente)

Inicio Slurm: **2026-06-08 01:14:56**


| q   | Fin (artifacto) | Duración desde q anterior | Acumulado desde inicio Slurm | Estado    |
| --- | --------------- | ------------------------- | ---------------------------- | --------- |
| 4   | 06:26           | **~5 h 12 min**           | ~5 h 12 min                  | completed |
| 8   | 07:12           | **~46 min**               | ~5 h 58 min                  | completed |
| 12  | 08:31           | **~1 h 19 min**           | ~7 h 16 min                  | completed |
| 16  | 10:49           | **~2 h 19 min**           | **~9 h 35 min**              | completed |
| 20  | —               | (en curso al redactar)    | >11 h Slurm                  | pendiente |


**Lectura:**

- El **primer q (q=4)** es mucho más lento que los siguientes: incluye arranque de sesión IBM, primeros jobs en cola, warmup del optimizador.
- **q=16** en modo sesión: **~2 h 20 min** de reloj entre fin de q=12 y fin de q=16.
- El job Slurm puede estar **RUNNING >10 h** con solo 4 puntos hechos; es normal en real-hw.

---

## Mediciones: runs anteriores (una tarea Slurm por q)

**Carpeta:** `outputs/20260607-185519_phase2_baseline-real-hw/`  
**Modo:** 25 tareas array (seed × q), `%1` serial, **timeout cliente 120 s** (muchas fallaron).

### Completados con éxito (log `Total runtime`)


| Job        | seed | q   | budget | Total runtime (script)   | Slurm Elapsed | Notas                     |
| ---------- | ---- | --- | ------ | ------------------------ | ------------- | ------------------------- |
| 1208416_1  | 10   | 4   | 32     | **1428 s (~24 min)**     | 23:54         | COBYLA + QNSPSA completos |
| 1208416_19 | 40   | 16  | 256    | **8392 s (~2 h 20 min)** | 2:19:57       | COBYLA + QNSPSA completos |


El tiempo de **q=16 ~2 h 20 min** coincide con el run por sesión actual → buena referencia.

### Fallos por timeout cliente (120 s / 600 s)

La mayoría de tareas del grid `20260607-185519` tienen `status: failed_runtime` en `run_status.json`. El job IBM **seguía en cola o ejecutándose**, pero el cliente dejó de esperar:

- Con **120 s**: fallo típico en **~2 min** de Slurm elapsed (solo COBYLA empezado).
- Con **600 s** (reintento `20260608-001651`): sesión entera seed 10 falló en todos los q en ~50 min total.

**No confundir** esos ~2 min con el tiempo que hubiera tardado un run completo: son **timeouts prematuros**, no duración real del experimento.

---

## Desglose conceptual de un punto (seed, q)

```
[Slurm START]
    → preflight IBM (service, backend)
    → Session.open (solo 1× por seed en modo actual)
    → transpile ansatz
    → bucle optimizador:
         cada evaluación → job IBM (cola + ejecución + shots)
    → guardar NPZ/PNG
[Slurm END para ese seed entero]
```


| Fase                                    | Orden de magnitud observado                                      |
| --------------------------------------- | ---------------------------------------------------------------- |
| Cola Slurm (otros seeds / tareas array) | 0 min – **horas** (depende de `%1` y duración del seed anterior) |
| Preflight + sesión IBM (1× por seed)    | minutos                                                          |
| **q=4** (primer q en sesión)            | **~20 min – 5 h** (gran varianza por cola IBM inicial)           |
| **q=8**                                 | **~45 min**                                                      |
| **q=12**                                | **~1 h 20 min**                                                  |
| **q=16**                                | **~2 h 20 min**                                                  |
| **q=20** (estimación)                   | **~4–6 h** (budget 2× vs q=16; circuito más grande)              |


---

## Estimaciones para planificar

### Un seed completo (5 qubits, modo sesión actual)

Suma aproximada de mediciones seed 10 hasta q=16: **~9,5 h**.  
Añadiendo q=20: **~12–16 h** por seed si la cola IBM se comporta como hasta ahora.

### Grid completo baseline (5 seeds)

- Serial `%1`: **~2,5–4 días** (5 × tiempo por seed), sin contar reintentos.
- Cuello de botella: **cola IBM + budget**, no CPU local del nodo Slurm.

### Límites a vigilar


| Límite                     | Valor                    | Riesgo                                                                                  |
| -------------------------- | ------------------------ | --------------------------------------------------------------------------------------- |
| Slurm `CPU-LARGE`          | 7 días                   | Bajo para un seed                                                                       |
| IBM `Session(max_time=8h)` | 8 h                      | Sesión larga; el seed 10 superó 8 h de wall clock y siguió — confirmar en dashboard IBM |
| Timeout cliente            | desactivado en `1208546` | Correcto para runs largos                                                               |


---

## Dónde mirar tiempos en el repo


| Fuente                                                      | Qué mide                                          |
| ----------------------------------------------------------- | ------------------------------------------------- |
| `sacct -j <JOBID>`                                          | Submit, Start, End, Elapsed Slurm                 |
| `outputs/logs/p2_real_hw_M_<job>_<task>.out`                | `Total runtime:` (segundos Python, fin de script) |
| `outputs/<run>/seed_*/qubits_*/optimizer_history.npz` mtime | Fin aproximado de cada punto                      |
| `run_status.json`                                           | `completed` vs `failed_runtime`                   |


Comandos útiles (solo lectura):

```bash
sacct -j 1208546 --format=JobID,State,Submit,Start,Elapsed,Timelimit
ls -lt outputs/20260608-011456_phase2_baseline-real-hw/seed_10/qubits_*/optimizer_history.npz
```

---

## Historial de configuración (por qué hay runs tan distintos)


| Run folder          | Job         | Modo Slurm       | Timeout cliente | Resultado                                              |
| ------------------- | ----------- | ---------------- | --------------- | ------------------------------------------------------ |
| `20260607-185519_*` | 1208416     | 1 job / (seed,q) | 120 s           | Casi todo `failed_runtime`; 2 éxitos (s10 q4, s40 q16) |
| `20260608-001651_*` | 1208532     | 1 sesión / seed  | 600 s           | 0/5 q completados (timeout)                            |
| `20260608-011456_*` | **1208546** | 1 sesión / seed  | **ninguno**     | seed 10: q4–16 OK; q=20 en curso                       |


**Conclusión práctica:** los tiempos largos (~~2 h en q=16) son reales; los runs de **~~2 min** del primer grid son artefacto del timeout, no del experimento.

---

## Referencias

- Budget y fórmulas: `[logic.md](logic.md)`
- Launcher: `src/phase2/sbatches/exp-real.sbatch`
- Script: `src/phase2/exp-real.py`
- Resultados de optimización (cuando existan): informe ideal/shot en `[report-phase2-ideal-shot-noise.md](report-phase2-ideal-shot-noise.md)`

