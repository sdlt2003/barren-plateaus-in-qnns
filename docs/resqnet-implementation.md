# ResQNet en este proyecto

Este documento resume:

1. que es una ResQNet segun el paper `arXiv:2305.03527`,
2. por que ayuda frente a barren plateaus,
3. como implementarla de forma practica en este repositorio.

---

## 1) Idea principal del paper

La propuesta de ResQNet (Residual Quantum Neural Network) toma una QNN "profunda" y la divide en varios **Quantum Nodes (QNs)**:

- cada QN contiene su propio PQC (subcircuito parametrico),
- los QNs se conectan en cascada,
- y se anaden **residual connections** entre nodos.

La intuicion es equivalente a ResNet clasica:

- en vez de aprender todo el mapeo de golpe, se facilita el flujo de informacion
- y se reduce la desaparicion de gradiente (en QNN: mitigacion de barren plateaus).

---

## 2) Estructura ResQNet (caso 2-QN, 1 residual)

El caso base del paper usa dos nodos:

- **QN1**: aplica un primer bloque parametrico.
- **QN2**: aplica un segundo bloque parametrico.

Conexion residual:

- se toma la salida intermedia de QN1,
- se combina con la entrada/base previa (skip),
- esa senal combinada alimenta QN2.

El paper lo expresa en forma hibrida (medicion intermedia + re-encoding), pero en practica se puede implementar en dos niveles:

1. **Version fiel (hibrida):**
   - medir salida intermedia de QN1,
   - sumar en clasico,
   - volver a codificar para QN2.
2. **Version practica (todo-cuantica):**
   - representar la conexion residual dentro del circuito compuesto,
   - sin cortar el flujo con medicion intermedia.

Para este repo, la segunda es la forma mas directa y robusta para un primer MVP.

---

## 3) Relacion con tu codigo actual

Tu arquitectura de phase2 vive en:

- `src/phase2/architectures.py`

Y el flujo de entrenamiento ya esta montado en:

- `src/phase2/exp-ideal.py`
- `src/phase2/exp-shot-noise.py`
- `src/phase2/exp-real.py`

Esos scripts consumen un `ArchitectureSpec` desde `build_architecture(...)`.

Por tanto, la integracion natural es:

- anadir `resqnet` como nueva arquitectura en `architectures.py`,
- mantener el resto de scripts casi intactos (solo pasar nuevos parametros de arquitectura por CLI).

---

## 4) Diseno recomendado para este proyecto

## 4.1 Bloque de capa cuantica

Segun el paper, el bloque base por repeticion usa:

- `RX` y `RY` por qubit,
- entrelazamiento lineal tipo vecinos con `CX`.

Ese bloque se repite un numero configurable de veces.

## 4.2 Division en nodos (depth split)

En vez de un unico bloque profundo, dividir profundidad total `D_L` en 2 nodos:

- `D1` para QN1
- `D2` para QN2
- con `D1 + D2 = D_L`

Ejemplos que el paper explora: `(5,1)`, `(4,2)`, `(3,3)`.

## 4.3 Residual

Para un MVP estable en este repo:

- implementar `residual_mode="structural"` (todo-cuantico),
- reflejar el skip entre QN1 y QN2 dentro del circuito total,
- evitar por ahora mediciones intermedias en medio del forward.

La version con medicion intermedia puede quedar como extension futura (`residual_mode="hybrid"`).

---

## 5) Plan de implementacion en archivos

## Paso A - Arquitectura nueva

Editar `src/phase2/architectures.py`:

- nueva constante: `RESQNET = "resqnet"`,
- incluirla en `available_architectures()`,
- implementar `_build_resqnet(n_qubits, depth_split, residual_mode)`,
- devolver `ArchitectureSpec` con metadata:
  - `family="resqnet"`
  - `depth_split`
  - `residual_mode`
  - `num_nodes=2`.

## Paso B - CLI en scripts phase2

Editar:

- `src/phase2/exp-ideal.py`
- `src/phase2/exp-shot-noise.py`
- `src/phase2/exp-real.py`

Anadir flags (opcionales):

- `--resqnet-depth-split 5,1`
- `--resqnet-total-depth 6` (si quieres derivar split automaticamente)
- `--resqnet-residual-mode structural`

Y pasar esos parametros a `build_architecture(...)`.

## Paso C - Launchers sbatch

Editar:

- `src/phase2/sbatches/submit_phase2.sh`
- `src/phase2/sbatches/exp-ideal.sbatch`
- `src/phase2/sbatches/exp-shot-noise.sbatch`
- `src/phase2/sbatches/exp-real.sbatch`

Para aceptar `ARCHITECTURE=resqnet` y opcionalmente `DEPTH_SPLIT`.

## Paso D - Documentacion

Actualizar:

- `docs/phase2-qcnn.md` o crear `docs/phase2-resqnet.md`

Con ejemplos de lanzamiento y recomendaciones de configuracion.

---

## 6) Configuracion inicial sugerida

Para arrancar sin mucha complejidad:

- `architecture = resqnet`
- `depth_split = (5,1)`
- `residual_mode = structural`
- mismo pipeline de optimizacion actual (COBYLA/QNSPSA y budget actual).

Motivo:

- es consistente con las configuraciones que mejor funcionan en el paper,
- minimiza cambios en el runtime actual.

---

## 7) Riesgos y decisiones tecnicas

1. **Fidelidad estricta al paper vs facilidad de integracion**
   - implementacion hibrida (medicion + re-encoding entre QNs) es mas fiel, pero mucho mas invasiva.
   - implementacion estructural es mas simple y estable para el repo actual.

2. **Comparabilidad con arquitecturas existentes**
   - conviene mantener mismos optimizadores, budget y criterio de parada para comparar limpio.

3. **Escalabilidad**
   - una vez estable 2-QN, extender a 3-QN y distintas posiciones de residual.

---

## 8) Roadmap posterior (opcional)

Despues del MVP:

- soporte 3-QN (`depth_split` triple, p.ej. `4,1,1`),
- diferentes ubicaciones de residual,
- version hibrida con medicion intermedia,
- analisis agregados por `depth_split` en `analyze_outputs.py`.

---

Si quieres, el siguiente paso es implementar directamente el **Paso A** (constructor `resqnet` en `architectures.py`) y dejarte un smoke test rapido para `n_qubits={4,8,16}`.
