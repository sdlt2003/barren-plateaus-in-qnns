# Arquitectura QCNN (Quantum Convolutional Neural Network)

Documento teórico de la arquitectura **qcnn** usada en este proyecto como contraste
estructurado frente al ansatz hardware-efficient baseline. Describe el diseño del
circuito, sus hipótesis inductivas y las implicaciones para optimización y barren
plateaus. No incluye instrucciones de ejecución.

---

## Papel en el estudio comparativo

El QCNN introduce **inductiva jerárquica**: en lugar de aplicar capas homogéneas de
rotaciones y entrelazado en todos los qubits (como en un HEA), alterna bloques de
**convolución local** y **pooling** que reducen progresivamente el número de grados
de libertad activos. El resto del protocolo experimental se mantiene deliberadamente
fijo respecto al baseline:

- Misma función de coste: expectativa de un operador de Pauli local (lectura en un
  único qubit).
- Mismos optimizadores y mismo criterio de parada por presupuesto de evaluaciones.
- Misma regla de budget dinámico proporcional al número de parámetros.

Así, las diferencias observadas en convergencia, sensibilidad al ruido de disparos o
dinámica de gradientes se atribuyen principalmente a la **topología del ansatz**, no
al protocolo de entrenamiento.

---

## Motivación e inductiva

La idea del QCNN cuántico, inspirada en las redes convolucionales clásicas, es que
patrones locales en el registro cuántico se procesen y compriman de forma
jerárquica antes de la medición. En un paisaje de optimización variacional esto puede:

1. **Reducir la profundidad efectiva** activa en cada etapa del circuito (menos qubits
   «relevantes» hacia el final).
2. **Imponer estructura** en cómo se mezcla la información (pares vecinos →
   supervivientes), en lugar de un entrelazado lineal global uniforme.
3. **Modificar la estadística de gradientes** respecto a un HEA del mismo tamaño de
   registro, lo cual es central para el estudio de barren plateaus en este trabajo.

En los resultados de simulación ideal, el QCNN suele converger de forma muy robusta
hacia el mínimo teórico del observable; bajo ruido de disparos la ventaja entre
optimizadores puede invertirse respecto al baseline, lo que sugiere un paisaje más
sensible a la estocasticidad en ciertos tamaños de registro.

---

## Topología general

El circuito actúa sobre \(n\) qubits y exige que \(n\) sea **potencia de 2**
(\(n = 2^L\)). Se definen \(L = \log_2 n\) **niveles jerárquicos**.

En cada nivel \( \ell = 0, \ldots, L-1 \):

1. **Convolución, subcapa 1:** se aplican bloques locales sobre pares vecinos
   \((0,1), (2,3), \ldots\) entre los qubits aún activos.
2. **Convolución, subcapa 2:** segunda pasada desplazada, sobre pares
   \((1,2), (3,4), \ldots\).
3. **Pooling:** cada par \((A, B)\) de qubits vecinos se fusiona; el qubit fuente
   \(A\) deja de participar y el **superviviente** es \(B\).

Tras \(L\) niveles queda **un único qubit activo**. La medición del coste no se hace
en el último índice físico del registro, sino en el **qubit superviviente** del árbol
de pooling (posición que depende de \(n\)).

Cada nivel completo (dos subcapas de convolución + pooling) constituye una **capa
lógica** para dinámicas de entrenamiento layer-wise: se puede congelar el nivel
anterior antes de entrenar el siguiente, manteniendo bajo el número de parámetros
libres y la profundidad efectiva en cada paso.

---

## Bloque de convolución

Cada bloque de convolución opera sobre un par de qubits vecinos \((q_L, q_R)\) y
combina rotaciones locales con un entrelazado de corto alcance:

- Rotación parametrizada en \(q_L\) y en \(q_R\).
- Puerta de entrelazado entre ambos.
- Rotación adicional en \(q_R\).
- Desentrelazado (misma puerta de dos qubits en sentido inverso al anterior).

Intuitivamente, el bloque actúa como un **filtro local** que mezcla información entre
vecinos antes del paso de compresión. Cada bloque aporta **tres** parámetros
independientes.

---

## Bloque de pooling

El módulo de pooling implementa una compresión controlada inspirada en la forma
condicional

\[
\mathcal{I} = |0\rangle\langle 0|_A \otimes U_0^{(B)} + |1\rangle\langle 1|_A \otimes U_1^{(B)},
\]

donde \(A\) es el qubit fuente y \(B\) el objetivo. Tras el bloque, **\(A\) queda
inactivo** y solo \(B\) sigue en el registro reducido.

En la implementación concreta de este repositorio, las ramas \(U_0\) y \(U_1\) se
realizan mediante rotaciones controladas (\texttt{cry}) con parámetros distintos.
Cada bloque de pooling aporta **dos** parámetros.

**Consecuencia teórica relevante:** las puertas controladas no son rotaciones de
un solo armónico con periodo \(2\pi\) en el sentido del parameter-shift estándar
\(\pm\pi/2\). Para el cálculo exacto del gradiente hace falta la regla adecuada al
tipo de puerta (en este proyecto, primitivas de gradiente de Qiskit en las métricas
de barren plateau). Como optimizador, métodos tipo NFT (Rotosolve) asumen
dependencia sinusoidal simple por parámetro y pueden ser solo **aproximadamente**
válidos en presencia de \texttt{cry}.

---

## Escalado en número de parámetros

Con \(L = \log_2 n\) niveles y la receta anterior:

| Cantidad | Fórmula |
|----------|---------|
| Bloques de convolución | \(2n - 2 - L\) |
| Bloques de pooling | \(n - 1\) |
| Parámetros totales \(p\) | \(3(2n - 2 - L) + 2(n - 1) = 8n - 8 - 3L\) |

Equivalentemente: \(p = 8n - 8 - 3\log_2 n\).

Ejemplos para los tamaños usados en los grids de simulación:

| \(n\) | \(L\) | \(p\) |
|------:|------:|------:|
| 4 | 2 | 18 |
| 8 | 3 | 47 |
| 16 | 4 | 108 |

Frente al baseline HEA en el mismo \(n\), el QCNN puede tener **menos o más**
parámetros según el tamaño (p. ej. \(p_{\mathrm{qcnn}}(8)=47\) vs \(p_{\mathrm{hea}}(8)=64\)),
de modo que las comparaciones deben interpretarse junto con el budget dinámico
\( \mathrm{budget} \propto p \).

---

## Lectura y observable

El coste es la expectativa \(\langle Z \rangle\) en el **qubit de readout** definido
por la arquitectura. En el QCNN ese qubit es el **superviviente final del árbol de
pooling**, no el qubit de índice máximo del registro inicial. Esta elección es coherente
con la semántica «extraer una característica comprimida» al final de la jerarquía.

---

## Restricción \(n\) potencia de 2

La construcción pairwise exige que en cada nivel el número de qubits activos se
reduzca exactamente a la mitad. Por ello solo están definidos tamaños
\(n \in \{4, 8, 16, 32, \ldots\}\). No es una limitación del simulador, sino de la
**definición topológica** del circuito en esta implementación.

---

## Relación con barren plateaus y mitigaciones

**Varianza de gradientes.** La estructura conv+pool puede mantener varianzas de
gradiente en la inicialización menos «colapsadas» que un HEA profundo en algunos
regímenes, pero el pooling con puertas controladas altera la estructura espectral de
las derivadas. Las métricas de certificación de barren plateau del proyecto miden
\(\mathrm{Var}(\partial C / \partial \theta_i)\) al inicio y el decaimiento de
\(\|\nabla C\|_2\) a lo largo de trayectorias cortas, usando gradientes exactos vía
primitivas de Qiskit.

**Layer-wise learning.** Agrupar parámetros por nivel jerárquico encaja de forma
natural con la hipótesis de mitigación: entrenar primero niveles superficiales con
pocos parámetros libres retrasa la entrada en regímenes donde el gradiente se
vuelve exponencialmente pequeño. En el QCNN, cada grupo corresponde a un nivel
completo (convolución doble + pooling).

**Ruido de disparos.** En simulación Monte Carlo, el QCNN muestra en algunos tamaños
mayor variabilidad de coste final bajo COBYLA que en ideal, mientras QNSPSA puede
tolerar mejor la estocasticidad — comportamiento coherente con un paisaje que, bajo
ruido, penaliza optimizadores basados en búsqueda directa sin explotar estructura de
gradiente fidelidad-based.

---

## Resumen

El QCNN del proyecto es un ansatz **jerárquico convolucional** con pooling pairwise
\(A \to B\), lectura en el qubit superviviente y \(n\) restringido a potencias de 2.
Aporta un contraste estructural frente al HEA baseline y al ResQNet residual: misma
tarea de optimización, distinta topología y, por tanto, distinta geometría del
paisaje, estadística de gradientes y comportamiento bajo ruido. Las caveats de
gradiente exacto (\texttt{cry}) y de optimizadores sinusoidales (NFT) deben tenerse
presentes al interpretar resultados y al redactar la memoria.
