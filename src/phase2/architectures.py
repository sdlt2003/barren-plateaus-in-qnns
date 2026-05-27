#!/usr/bin/env python3
"""Architecture factory for phase 2 experiments."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import efficient_su2


BASELINE_HEA = "baseline_hea"
QCNN = "qcnn"


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    circuit: QuantumCircuit
    readout_qubit: int
    metadata: dict[str, Any]

    @property
    def num_parameters(self) -> int:
        return int(self.circuit.num_parameters)


def available_architectures() -> tuple[str, ...]:
    return (BASELINE_HEA, QCNN)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _build_baseline_hea(n_qubits: int) -> ArchitectureSpec:
    reps_log = max(1, int(np.round(np.log2(n_qubits))))
    circuit = efficient_su2(num_qubits=n_qubits, reps=reps_log, entanglement="linear")
    return ArchitectureSpec(
        name=BASELINE_HEA,
        circuit=circuit,
        readout_qubit=n_qubits - 1,
        metadata={
            "reps": reps_log,
            "family": "hea",
            "entanglement": "linear",
        },
    )


def _apply_conv_block(qc: QuantumCircuit, left: int, right: int, params: ParameterVector, offset: int) -> int:
    qc.ry(params[offset], left)
    qc.ry(params[offset + 1], right)
    qc.cx(left, right)
    qc.rz(params[offset + 2], right)
    qc.cx(left, right)
    return offset + 3


def _apply_pool_block(qc: QuantumCircuit, source: int, target: int, params: ParameterVector, offset: int) -> int:
    # Pooling module approximates: I = |0><0|_A ⊗ U0_B + |1><1|_A ⊗ U1_B
    # with A=source and B=target. After this layer, source becomes inactive.
    qc.x(source)
    qc.cry(params[offset], source, target)      # controlled-U0 branch
    qc.x(source)
    qc.cry(params[offset + 1], source, target)  # controlled-U1 branch
    return offset + 2


def _build_qcnn(n_qubits: int) -> ArchitectureSpec:
    if not _is_power_of_two(n_qubits):
        raise ValueError(
            "qcnn requires n_qubits to be a power of 2 "
            f"(got n_qubits={n_qubits})."
        )

    levels = int(math.log2(n_qubits))
    conv_blocks = 0
    active_count = n_qubits
    for _ in range(levels):
        # Two convolution sublayers on alternating neighboring pairs.
        conv_blocks += active_count // 2
        conv_blocks += max(0, (active_count - 1) // 2)
        active_count //= 2
    pool_blocks = n_qubits - 1
    num_params = 3 * conv_blocks + 2 * pool_blocks
    params = ParameterVector("theta", num_params)
    circuit = QuantumCircuit(n_qubits, name="qcnn")

    active = list(range(n_qubits))
    offset = 0
    for level in range(levels):
        # Convolution sublayer 1: (0,1), (2,3), ...
        conv_pairs_even = [(active[idx], active[idx + 1]) for idx in range(0, len(active), 2)]
        for left, right in conv_pairs_even:
            offset = _apply_conv_block(circuit, left, right, params, offset)
        circuit.barrier()

        # Convolution sublayer 2: (1,2), (3,4), ...
        conv_pairs_odd = [(active[idx], active[idx + 1]) for idx in range(1, len(active) - 1, 2)]
        for left, right in conv_pairs_odd:
            offset = _apply_conv_block(circuit, left, right, params, offset)
        circuit.barrier()

        # Pooling: source=A (left) controls target=B (right); A is then inactive.
        pool_pairs = [(active[idx], active[idx + 1]) for idx in range(0, len(active), 2)]
        survivors: list[int] = []
        for source, target in pool_pairs:
            offset = _apply_pool_block(circuit, source=source, target=target, params=params, offset=offset)
            survivors.append(target)
        active = survivors
        if level < levels - 1:
            circuit.barrier()

    readout_qubit = active[0]
    return ArchitectureSpec(
        name=QCNN,
        circuit=circuit,
        readout_qubit=readout_qubit,
        metadata={
            "levels": levels,
            "family": "qcnn",
            "pooling": "hierarchical_pairwise",
            "conv_sublayers_per_level": 2,
            "pooling_direction": "A_to_B_keep_B",
            "requires_power_of_two": True,
        },
    )


def build_architecture(architecture: str, n_qubits: int) -> ArchitectureSpec:
    if n_qubits <= 1:
        raise ValueError(f"n_qubits must be >= 2, got {n_qubits}")

    if architecture == BASELINE_HEA:
        return _build_baseline_hea(n_qubits=n_qubits)
    if architecture == QCNN:
        return _build_qcnn(n_qubits=n_qubits)

    known = ", ".join(available_architectures())
    raise ValueError(f"Unknown architecture '{architecture}'. Available: {known}.")
