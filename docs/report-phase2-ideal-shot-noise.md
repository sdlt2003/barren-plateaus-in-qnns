# Informe Phase 2 — resultados ideal y shot-noise

**Fecha del informe:** 2026-06-08  
**Runs analizados:** grid Slurm del 2026-06-02 (job arrays 1205561–1205572)  
**Escenarios:** simulación ideal (statevector) y shot-noise (Monte Carlo)  
**Arquitecturas:** `baseline_hea`, `qcnn`, `resqnet`

**Figuras** (abrir en lateral desde `report-assets/phase2-ideal-shot-noise/`):


| Sección               | Archivo                   |
| --------------------- | ------------------------- |
| 1 Baseline ideal      | `baseline-ideal-zoom.png` |
| 2 Baseline shot-noise | `baseline-shot-zoom.png`  |
| 3 QCNN ideal          | `qcnn-ideal-zoom.png`     |
| 4 QCNN shot-noise     | `qcnn-shot-zoom.png`      |
| 5 ResQNet ideal       | `resqnet-ideal-zoom.png`  |
| 6 ResQNet shot-noise  | `resqnet-shot-zoom.png`   |


---

## Configuración común


| Parámetro      | Valor                                                            |
| -------------- | ---------------------------------------------------------------- |
| Seeds          | 10, 20, 30, 40, 50                                               |
| Optimizadores  | COBYLA vs QNSPSA                                                 |
| Budget         | Dinámico: `budget_evals = k × p` con **k = 37.5**                |
| Early stopping | Desactivado globalmente (solo para por budget)                   |
| Métrica        | Coste final = ⟨Z⟩ en el qubit de readout (mínimo teórico **−1**) |


**Cobertura por arquitectura**


| Arquitectura | Qubits (ideal / shot-noise) | Puntos esperados |
| ------------ | --------------------------- | ---------------- |
| baseline_hea | 4, 8, 12, 16, 20            | 25               |
| qcnn         | 4, 8, 16                    | 15               |
| resqnet      | 4, 8, 16                    | 15               |


**Lectura de los gráficos**

- Eje Y en zoom **[0, −1]** (`final_cost_vs_qubits.png`); resalta diferencias entre optimizadores cerca del mínimo.

En las tablas agregadas, **cobyla_wins / qnspsa_wins** cuenta cuántas semillas tienen menor coste final por optimizador en cada tamaño de qubits.

---

## Resumen comparativo


| Run                 | Completados | q=20                   | COBYLA (tendencia)          | QNSPSA (tendencia)                |
| ------------------- | ----------- | ---------------------- | --------------------------- | --------------------------------- |
| Baseline ideal      | 20/25       | 5× TIMEOUT             | ~−1 en q≤16                 | ~−0.95, peor que COBYLA           |
| Baseline shot-noise | 21/25       | 4× TIMEOUT, 1 OK (s50) | ~−0.86…−0.92                | Competitivo; gana en varios q     |
| QCNN ideal          | 15/15       | N/A                    | ~−1 en todos los q          | ~−0.99, ligeramente peor          |
| QCNN shot-noise     | 15/15       | N/A                    | Muy variable (q=16 mal)     | Más estable, ~−0.99 en q=8,16     |
| ResQNet ideal       | 15/15       | N/A                    | ~−1 en q=4,8; COBYLA domina | QNSPSA ~−0.55…−0.73, mucho peor   |
| ResQNet shot-noise  | 15/15       | N/A                    | Débil en todos los q        | También débil; QNSPSA gana en q=8 |


**Incidencias Slurm:** los jobs de **baseline q=20** (ideal y shot-noise, seeds 10–40) hicieron **TIMEOUT** a 8 h. Solo shot-noise **seed=50, q=20** completó (~2 h 50 min).

---

## 1. Baseline — ideal

**Carpeta:** `outputs/20260602-224946_phase2_baseline/`  
**Cobertura:** 20/25 (faltan q=20 excepto ninguno)

### Agregado por qubits


| q   | n   | COBYLA (media) | QNSPSA (media) | COBYLA wins | QNSPSA wins |
| --- | --- | -------------- | -------------- | ----------- | ----------- |
| 4   | 5   | −1.000         | −0.963         | 5           | 0           |
| 8   | 5   | −1.000         | −0.952         | 5           | 0           |
| 12  | 5   | −1.000         | −0.973         | 5           | 0           |
| 16  | 5   | −1.000         | −0.954         | 5           | 0           |


### Figura

**→ F1** · `[report-assets/phase2-ideal-shot-noise/baseline-ideal-zoom.png](report-assets/phase2-ideal-shot-noise/baseline-ideal-zoom.png)`

### Comentarios

- **COBYLA converge de forma excelente** en simulación ideal: coste final ≈ −1 en todos los q completados (4–16).
- **QNSPSA se queda sistemáticamente por encima** (~−0.95), probablemente por el coste extra de fidelidad que reduce iteraciones efectivas de objetivo dentro del mismo budget.
- No se observa barren plateau claro en baseline ideal en este rango: el coste empeora muy poco al subir qubits.
- **q=20 pendiente** de re-ejecutar (timeout Slurm).

---

## 2. Baseline HEA — shot-noise

**Carpeta:** `outputs/20260602-224946_phase2_baseline-shot-noise/`  
**Cobertura:** 21/25 (q=20: solo seed 50)

### Agregado por qubits


| q   | n   | COBYLA (media) | QNSPSA (media) | COBYLA wins | QNSPSA wins |
| --- | --- | -------------- | -------------- | ----------- | ----------- |
| 4   | 5   | −0.919         | −0.871         | 3           | 2           |
| 8   | 5   | −0.879         | −0.952         | 3           | 2           |
| 12  | 5   | −0.883         | −0.887         | 3           | 2           |
| 16  | 5   | −0.863         | −0.790         | 4           | 1           |
| 20  | 1   | −0.919         | −0.737         | 1           | 0           |


### Figura

**→ F2** · `[report-assets/phase2-ideal-shot-noise/baseline-shot-zoom.png](report-assets/phase2-ideal-shot-noise/baseline-shot-zoom.png)`

### Comentarios

- Con ruido de disparos el coste final **baja a ~−0.86…−0.92** (lejos de −1): el Monte Carlo introduce varianza y limita la optimización.
- **QNSPSA es más competitivo** que en ideal: gana en varias semillas en q=4, 8 y 12.
- En **q=16**, COBYLA tiene ventaja media; la dispersión entre semillas es notable.
- **q=20** casi sin datos: conviene re-lanzar seeds 10–40 con más tiempo Slurm; el único punto (seed 50) sugiere convergencia lenta pero posible.

---

## 3. QCNN — ideal

**Carpeta:** `outputs/20260602-224946_phase2_qcnn/`  
**Cobertura:** 15/15

### Agregado por qubits


| q   | n   | COBYLA (media) | QNSPSA (media) | COBYLA wins | QNSPSA wins |
| --- | --- | -------------- | -------------- | ----------- | ----------- |
| 4   | 5   | −1.000         | −0.991         | 5           | 0           |
| 8   | 5   | −1.000         | −0.993         | 5           | 0           |
| 16  | 5   | −1.000         | −0.994         | 5           | 0           |


### Figura

**→ F3** · `[report-assets/phase2-ideal-shot-noise/qcnn-ideal-zoom.png](report-assets/phase2-ideal-shot-noise/qcnn-ideal-zoom.png)`

### Comentarios

- QCNN en ideal es **muy fuerte**: COBYLA alcanza esencialmente el mínimo teórico en q ∈ {4, 8, 16}.
- QNSPSA también va bien (~−0.99) pero **no supera a COBYLA** en ninguna semilla.
- La arquitectura parece **menos sensible al tamaño** que baseline en ideal (curvas planas cerca de −1).
- Tiempos de cómputo moderados (máx ~156 s en q=16).

---

## 4. QCNN — shot-noise

**Carpeta:** `outputs/20260602-224946_phase2_qcnn-shot-noise/`  
**Cobertura:** 15/15

### Agregado por qubits


| q   | n   | COBYLA (media) | QNSPSA (media) | COBYLA wins | QNSPSA wins |
| --- | --- | -------------- | -------------- | ----------- | ----------- |
| 4   | 5   | −0.968         | −0.992         | 2           | 2           |
| 8   | 5   | −0.661         | −0.992         | 0           | 2           |
| 16  | 5   | −0.345         | −0.996         | 1           | 3           |


### Figura

**→ F4** · `[report-assets/phase2-ideal-shot-noise/qcnn-shot-zoom.png](report-assets/phase2-ideal-shot-noise/qcnn-shot-zoom.png)`

### Comentarios

- Resultado **muy distinto al ideal**: COBYLA se vuelve **inestable** al crecer q (media −0.35 en q=16, con semillas que casi no optimizan).
- **QNSPSA es claramente superior con ruido** en q=8 y q=16 (~−0.99 frente a medias positivas o cercanas a 0 en COBYLA).
- Hipótesis: la estructura QCNN + pooling hace el paisaje más difícil para COBYLA bajo ruido; QNSPSA tolera mejor la estocasticidad.
- Caso destacado para el paper: **arquitectura que en ideal favorece a COBYLA, pero en shot-noise favorece a QNSPSA**.

---

## 5. ResQNet — ideal

**Carpeta:** `outputs/20260602-224948_phase2_resqnet/`  
**Cobertura:** 15/15

### Agregado por qubits


| q   | n   | COBYLA (media) | QNSPSA (media) | COBYLA wins | QNSPSA wins |
| --- | --- | -------------- | -------------- | ----------- | ----------- |
| 4   | 5   | −1.000         | −0.554         | 5           | 0           |
| 8   | 5   | −0.9997        | −0.605         | 5           | 0           |
| 16  | 5   | −0.9999        | −0.731         | 5           | 0           |


### Figura

**→ F5** · `[report-assets/phase2-ideal-shot-noise/resqnet-ideal-zoom.png](report-assets/phase2-ideal-shot-noise/resqnet-ideal-zoom.png)`

### Comentarios

- **COBYLA sigue convergiendo casi a −1** en todos los q: ResQNet no parece empeorar el optimum en simulación exacta.
- **QNSPSA falla de forma sistemática** (coste final ~−0.55…−0.73): el optimizador no explota bien esta arquitectura con el budget actual.
- ResQNet tiene **más parámetros** (p=224 en q=16, budget ~8400): QNSPSA agota budget en fidelidad antes de refinar el coste (ver gráficos `optimizer_compare_budget.png` por run).
- Tiempos largos en q=16 (~15 min por semilla).

---

## 6. ResQNet — shot-noise

**Carpeta:** `outputs/20260602-224948_phase2_resqnet-shot-noise/`  
**Cobertura:** 15/15

### Agregado por qubits


| q   | n   | COBYLA (media) | QNSPSA (media) | COBYLA wins | QNSPSA wins |
| --- | --- | -------------- | -------------- | ----------- | ----------- |
| 4   | 5   | −0.701         | −0.269         | 5           | 0           |
| 8   | 5   | +0.014         | −0.204         | 0           | 5           |
| 16  | 5   | −0.349         | −0.515         | 3           | 2           |


### Figura

**→ F6** · `[report-assets/phase2-ideal-shot-noise/resqnet-shot-zoom.png](report-assets/phase2-ideal-shot-noise/resqnet-shot-zoom.png)`

### Comentarios

- **Ningún escenario alcanza −1**: ResQNet bajo shot-noise es el caso más difícil del grid.
- En **q=8**, COBYLA **empeora** (media ≈ 0); QNSPSA mantiene coste negativo (~−0.20).
- En **q=16**, QNSPSA ligeramente mejor que COBYLA en media, pero ambos lejos del óptimo.
- Posible **barren plateau o paisaje ruidoso** en ResQNet: la profundidad + residual + pooling amplifican el ruido.
- Prioridad para siguientes experimentos: más budget, más shots, o reducir profundidad en q grandes.

---

## Conclusiones generales

1. **Ideal:** baseline y QCNN convergen bien con COBYLA; ResQNet también con COBYLA pero no con QNSPSA.
2. **Shot-noise:** el ruido nivela o invierte ventajas — **QNSPSA gana en QCNN (q≥8) y ResQNet (q=8)**.
3. **COBYLA** es el optimizador más fiable en simulación exacta; **QNSPSA** merece estudiarse en escenarios ruidosos y arquitecturas profundas.
4. **Datos incompletos:** baseline **q=20** (4–5 seeds) por timeout; conviene re-ejecutar con partición `CPU-LARGE` / sin límite 8 h.
5. **Hardware real** (fuera de este informe) sigue en curso por separado.

---

## Referencias en el repo

- Parámetros y budgets: `[logic.md](logic.md)`
- Análisis por run: `outputs/<run>/analysis/run_diagnostics.md`
- Regenerar análisis:

```bash
bnd run python src/phase1/analyze_outputs.py 20260602-224946_phase2_baseline
```

