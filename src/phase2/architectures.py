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
RESQNET = "resqnet"


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
    return (BASELINE_HEA, QCNN, RESQNET)


def parse_depth_split(raw: str) -> tuple[int, int]:
    """Parse depth split like '5,1' into a validated tuple."""
    parts = [chunk.strip() for chunk in raw.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Depth split must have 2 integers like '5,1' (got: {raw!r})")
    try:
        left, right = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Depth split values must be integers (got: {raw!r})") from exc
    if left <= 0 or right <= 0:
        raise ValueError(f"Depth split values must be positive (got: {raw!r})")
    return left, right


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


def _apply_resqnet_layer(qc: QuantumCircuit, n_qubits: int, params, offset: int) -> int:
    """Apply one ResQNet layer: RX/RY on all qubits + linear CNOT chain."""
    for q in range(n_qubits):
        qc.rx(params[offset], q)
        offset += 1
        qc.ry(params[offset], q)
        offset += 1
    for q in range(n_qubits - 1):
        qc.cx(q, q + 1)
    return offset


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


def _build_resqnet(
    n_qubits: int,
    *,
    depth_split: tuple[int, int] = (5, 1),
    residual_mode: str = "structural",
) -> ArchitectureSpec:
    d1, d2 = depth_split
    if d1 <= 0 or d2 <= 0:
        raise ValueError(f"resqnet depth split must be positive, got {depth_split}")
    if residual_mode != "structural":
        raise ValueError(f"Unsupported resqnet residual_mode={residual_mode!r}. Use 'structural'.")

    # Paper-inspired residual approximation for this repository:
    # a shared "re-upload" layer is applied before QN1 and re-applied before QN2,
    # while QN1 and QN2 each keep their own trainable layers.
    shared_params = ParameterVector("phi", 2 * n_qubits)
    qn1_params = ParameterVector("theta1", 2 * n_qubits * d1)
    qn2_params = ParameterVector("theta2", 2 * n_qubits * d2)

    circuit = QuantumCircuit(n_qubits, name="resqnet")
    off_shared = 0
    off_qn1 = 0
    off_qn2 = 0

    off_shared = _apply_resqnet_layer(circuit, n_qubits, shared_params, off_shared)
    circuit.barrier()

    for _ in range(d1):
        off_qn1 = _apply_resqnet_layer(circuit, n_qubits, qn1_params, off_qn1)
    circuit.barrier()

    # Residual re-injection path (shared parameters tied to input-side block).
    off_shared = _apply_resqnet_layer(circuit, n_qubits, shared_params, 0)
    circuit.barrier()

    for _ in range(d2):
        off_qn2 = _apply_resqnet_layer(circuit, n_qubits, qn2_params, off_qn2)

    return ArchitectureSpec(
        name=RESQNET,
        circuit=circuit,
        readout_qubit=n_qubits - 1,
        metadata={
            "family": "resqnet",
            "num_nodes": 2,
            "depth_split": [d1, d2],
            "total_depth": d1 + d2,
            "residual_mode": residual_mode,
            "shared_reupload": True,
            "layer_design": "rx-ry + linear_cnot",
        },
    )


def build_architecture(
    architecture: str,
    n_qubits: int,
    *,
    resqnet_depth_split: tuple[int, int] | None = None,
    resqnet_residual_mode: str = "structural",
) -> ArchitectureSpec:
    if n_qubits <= 1:
        raise ValueError(f"n_qubits must be >= 2, got {n_qubits}")

    if architecture == BASELINE_HEA:
        return _build_baseline_hea(n_qubits=n_qubits)
    if architecture == QCNN:
        return _build_qcnn(n_qubits=n_qubits)
    if architecture == RESQNET:
        return _build_resqnet(
            n_qubits=n_qubits,
            depth_split=resqnet_depth_split or (5, 1),
            residual_mode=resqnet_residual_mode,
        )

    known = ", ".join(available_architectures())
    raise ValueError(f"Unknown architecture '{architecture}'. Available: {known}.")
