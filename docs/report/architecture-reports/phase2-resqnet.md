# Arquitectura ResQNet (Residual Quantum Network)

Documento teórico de la arquitectura **resqnet** usada en este proyecto como tercer
contraste estructurado frente al ansatz hardware-efficient **baseline** y al **qcnn**.
Describe el diseño del circuito, sus hipótesis inductivas (conexión residual y
re-upload compartido) y las implicaciones para optimización y barren plateaus. No
incluye instrucciones de ejecución.

---

## Papel en el estudio comparativo

El ResQNet introduce **profundidad apilada con camino residual**: en lugar de
comprimir el registro jerárquicamente (QCNN) o repetir capas HEA homogéneas
(baseline), encadena dos bloques de red cuántica (**QN1** y **QN2**) separados por
una **re-inyección** del mismo bloque de re-upload. El protocolo experimental se
mantiene fijo respecto a las otras arquitecturas:

- Misma función de coste: expectativa de un operador de Pauli local (lectura en un
  único qubit).
- Mismos optimizadores y mismo criterio de parada por presupuesto de evaluaciones.
- Misma regla de budget dinámico proporcional al número de parámetros.

Las diferencias observadas — en particular la fuerte asimetría COBYLA vs QNSPSA en
ideal y el peor comportamiento bajo ruido de disparos — se interpretan como
efectos de la **topología profunda + parámetros compartidos**, no del protocolo de
entrenamiento.

---

## Motivación e inductiva

Las redes residuales clásicas mitigan la degradación del gradiente en redes profundas
mediante atajos que preservan información de capas anteriores. En el marco variacional
cuántico, un análogo estructural es:

1. **Apilar varias capas de rotación + entrelazado** sin reducir el número de qubits
   activos (profundidad efectiva alta sobre todo el registro).
2. **Reinyectar** un bloque de codificación compartido entre sub-redes, de modo que
   el segundo tramo (QN2) no parte de un estado ya «puromente procesado» por QN1 sin
   atajo hacia la codificación inicial.

En este repositorio la conexión residual no se implementa como suma de estados en
Hilbert, sino como **modo estructural** (`residual_mode = structural`): el bloque
compartido de re-upload se **vuelve a aplicar** con los mismos parámetros antes de
QN2, aproximando la idea de «saltar» información de la rama de entrada hacia la
rama profunda.

Efectos esperados en el estudio de barren plateaus:

- **Más parámetros y mayor profundidad** que baseline y QCNN a igual \(n\) → mayor
  riesgo de varianzas de gradiente pequeñas y paisajes difíciles.
- **Parámetros compartidos** (`phi` aplicado dos veces) → la dependencia del coste
  en cada \(\phi_i\) no es sinusoidal de un solo armónico; afecta gradientes exactos
  (si se usara shift manual) y optimizadores tipo NFT.
- En los resultados phase 2, ResQNet en **ideal** converge bien con COBYLA pero
  QNSPSA queda muy por detrás; bajo **shot-noise** es el escenario más difícil del
  grid (ningún punto alcanza el mínimo teórico con consistencia).

---

## Topología general

El circuito actúa sobre \(n\) qubits lineales. A diferencia del QCNN, **no exige**
\(n\) potencia de 2 en la implementación; en los grids del proyecto se usa
\(n \in \{4, 8, 16\}\) por alineación con el resto de arquitecturas estructuradas.

La profundidad se reparte entre dos sub-redes mediante un **depth split**
\((D_1, D_2)\) (por defecto \((5, 1)\)):

```
[ bloque compartido φ ]  →  [ QN1: D₁ capas ]  →  [ bloque compartido φ ]  →  [ QN2: D₂ capas ]
```

En palabras:

1. **Re-upload inicial:** una capa compartida parametrizada por \(\phi\) (vector
   de longitud \(2n\)).
2. **QN1:** \(D_1\) capas entrenables independientes (\(\theta_1\)).
3. **Re-upload residual:** se reaplica el **mismo** bloque \(\phi\) (mismos
   parámetros, no una copia independiente).
4. **QN2:** \(D_2\) capas entrenables (\(\theta_2\)).

Cada «capa» ResQNet del proyecto consiste en:

- Rotación **RX** y **RY** en **todos** los qubits (\(2n\) ángulos por capa).
- Cadena lineal de **CNOT** \((0,1), (1,2), \ldots, (n-2, n-1)\).

No hay pooling ni reducción de qubits: el registro completo permanece activo hasta
la medición.

---

## Bloques de parámetros

| Bloque | Símbolo | Tamaño | Rol |
|--------|---------|--------|-----|
| Re-upload compartido | \(\phi\) | \(2n\) | Codificación inicial y re-inyección residual (mismos parámetros en ambas aplicaciones) |
| Sub-red 1 | \(\theta_1\) | \(2n \cdot D_1\) | \(D_1\) capas entrenables antes del segundo \(\phi\) |
| Sub-red 2 | \(\theta_2\) | \(2n \cdot D_2\) | \(D_2\) capas entrenables tras la re-inyección |

**Número total de parámetros:**

\[
p = 2n + 2n D_1 + 2n D_2 = 2n\,(1 + D_1 + D_2).
\]

Con el default \((D_1, D_2) = (5, 1)\): \(p = 14n\).

**Invariancia del split:** si \(D_1 + D_2\) se mantiene constante, \(p\) no cambia.
Por ejemplo \((5,1)\), \((4,2)\) y \((3,3)\) tienen el mismo \(p\) para un dado
\(n\); solo cambia cómo se distribuye la profundidad entre QN1 y QN2.

---

## Escalado y comparación con otras arquitecturas

Ejemplos con \((D_1, D_2) = (5, 1)\):

| \(n\) | \(D_1, D_2\) | \(p\) | Budget sim (\(k=37.5\)) | Budget HW (\(k=20\)) |
|------:|:------------:|------:|------------------------:|----------------------:|
| 4 | 5, 1 | 56 | 2100 | 1120 |
| 8 | 5, 1 | 112 | 4200 | 2240 |
| 16 | 5, 1 | 224 | 8400 | 4480 |
| 32 | 5, 1 | 448 | 16800 | 8960 |

Frente al baseline HEA y al QCNN en el mismo \(n\), ResQNet tiene **más parámetros
y mayor profundidad de circuito** (p. ej. \(p=224\) vs \(p=160\) baseline y
\(p=108\) QCNN en \(n=16\)). Las comparaciones deben leerse junto con
\(\mathrm{budget} \propto p\): ResQNet recibe más evaluaciones nominales, pero
también consume más coste por paso debido al circuito más largo.

Métricas estructurales (default \(D_1=5, D_2=1\), \(n=8\)): profundidad \(\approx 52\),
tamaño del circuito \(\approx 184\) puertas; lectura en el qubit \(n-1\).

---

## Lectura y observable

El coste es \(\langle Z \rangle\) en el **último qubit** del registro
(`readout_qubit = n - 1`), igual que en el baseline HEA y a diferencia del QCNN
(donde el readout es el superviviente del pooling).

---

## Agrupación layer-wise

Para la dinámica de entrenamiento **layer-wise** del proyecto, los parámetros se
agrupan en este orden (según el binding de `circuit.parameters`, ordenado por nombre
de vector: \(\phi < \theta_1 < \theta_2\)):

1. Bloque compartido \(\phi\) (\(2n\) parámetros).
2. Cada una de las \(D_1\) capas de QN1 (\(2n\) parámetros por capa).
3. Cada una de las \(D_2\) capas de QN2 (\(2n\) parámetros por capa).

Total: \(1 + D_1 + D_2\) grupos (p. ej. **7 grupos** con \((5,1)\)). El optimizador
interno layer-wise solo puede ser **sin fidelidad** (COBYLA, NFT): QNSPSA no es
compatible sobre el subespacio enmascarado.

---

## Conexión residual: alcance de la aproximación

El modo **structural** implementado aquí es una aproximación práctica, no una
suma residual exacta en el espacio de estados. La re-aplicación del bloque \(\phi\)
introduce un atajo paramétrico entre la rama «entrada» y la rama «profunda», pero:

- Los parámetros \(\phi\) aparecen **dos veces** en el circuito → la derivada del
  coste respecto a \(\phi_i\) acumula contribuciones de ambas aplicaciones con
  estructura espectral más rica que una rotación simple.
- NFT (Rotosolve) asume dependencia sinusoidal **por parámetro independiente**; en
  ResQNet la reutilización de \(\phi\) y la profundidad de QN1/QN2 hacen que NFT
  sea solo **aproximadamente** válido (en la práctica converge, a veces más lento
  que en baseline/QCNN).

Para las **métricas de gradiente** del proyecto se usan primitivas Qiskit exactas
(`ReverseEstimatorGradient` / `LinCombEstimatorGradient`), que no sufren el sesgo
del parameter-shift manual \(\pm\pi/2\) que afectaba a implementaciones antiguas
con parámetros compartidos.

---

## Relación con barren plateaus y mitigaciones

**Varianza de gradientes.** En las métricas de certificación, ResQNet suele mostrar
un decaimiento más marcado de \(\mathrm{Var}(\partial C / \partial \theta_i)\) con
\(n\) que el QCNN en inicialización aleatoria — coherente con una ansatz más profunda
sobre todo el registro. El decaimiento de \(\|\nabla C\|_2\) a lo largo de trayectorias
cortas ayuda a contrastar si el optimizador entra pronto en régimen de gradiente
aplanado.

**Layer-wise learning.** La agrupación natural (φ → capas QN1 → capas QN2) encaja
con la hipótesis de mitigación: entrenar primero la codificación compartida y las
capas superficiales de QN1 mantiene baja la profundidad efectiva «libre» en cada
fase. Es la arquitectura donde layer-wise + NFT tiene más sentido interpretativo,
dado el coste alto de optimizar los \(p\) parámetros simultáneamente.

**Ruido de disparos.** ResQNet fue el caso **más difícil** del grid phase 2 bajo
Monte Carlo: costes finales lejos de \(-1\) en todos los tamaños; en \(q=8\) QNSPSA
supera a COBYLA en media. La profundidad + re-upload + entrelazado lineal amplifican
la varianza del estimador del coste y dificultan optimizadores que no explotan bien
la estructura bajo ruido.

**QNSPSA en ideal.** El presupuesto compartido se consume también en evaluaciones de
**fidelidad**; con \(p\) grande (p. ej. 224 en \(n=16\)) QNSPSA agota el budget
antes de refinar el coste, mientras COBYLA sigue convergiendo cerca de \(-1\). Esto
es un efecto del **optimizador + budget**, no una propiedad intrínseca del ansatz,
pero es central al interpretar tablas phase 2.

---

## Resumen

El ResQNet del proyecto es un ansatz **profundo de dos etapas** (QN1 + QN2) con
**re-upload compartido** \(\phi\) entre tramos, capas RX/RY + CNOT lineal, lectura
en el último qubit y \(p = 2n(1 + D_1 + D_2)\). Aporta el contraste «residual +
profundidad» frente al HEA baseline y al QCNN jerárquico: misma tarea de optimización,
mayor \(p\) y profundidad, parámetros compartidos y comportamiento distinto bajo
ruido y bajo QNSPSA. Las caveats de \(\phi\) reutilizado y de NFT/gradientes
estructurales deben explicitarse al redactar la memoria.
