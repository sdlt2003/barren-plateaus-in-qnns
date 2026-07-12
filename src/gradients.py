#!/usr/bin/env python3
"""Gradient diagnostics for barren-plateau certification.

These helpers use **Qiskit's own gradient primitives**, which are exact for any
gate structure -- including controlled rotations (``cry`` in the QCNN pooling)
and shared parameters (``phi`` reused in ResQNet). This removes the bias of the
previous hand-rolled +-pi/2 parameter-shift, which was only exact for
single-frequency Pauli-rotation gates.

Two backends, one per noise regime:
  - ``reverse`` (ideal): :class:`ReverseEstimatorGradient`, exact and cheap on
    the statevector (reverse-mode, one pass for the whole gradient vector).
  - ``lincomb`` (shot-noise): :class:`LinCombEstimatorGradient` with a finite
    ``precision``. The gradient RULE stays exact (linear-combination-of-unitaries)
    while ``precision`` injects the statistical noise floor (precision ~= 1/sqrt(shots))
    that hides exponentially small signals on real hardware.

The metrics computed on top:
  - Var(dC/dtheta_i) at initialization (barren plateau => exponential decay with n).
  - L2 gradient-norm decay along an optimization trajectory.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.gradients import (
    LinCombEstimatorGradient,
    ReverseEstimatorGradient,
)

def build_gradient(method: str = "reverse", *, precision: float | None = None, estimator=None):
    """Instantiate a Qiskit gradient primitive.

    - ``reverse``: exact statevector gradient (ideal, no sampling noise).
    - ``lincomb``: exact-rule gradient evaluated through ``estimator`` with the
      given ``precision`` (finite precision => shot-like noise).
    """
    method = method.strip().lower()
    if method == "reverse":
        return ReverseEstimatorGradient()
    if method == "lincomb":
        return LinCombEstimatorGradient(estimator or StatevectorEstimator(), precision=precision)
    raise ValueError(f"Unknown gradient method '{method}'. Use 'reverse' or 'lincomb'.")

def _run_gradients(gradient, circuit, observable, param_sets: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Full gradient vector for each parameter set, via one batched primitive run."""
    param_list = [np.asarray(p, dtype=float) for p in param_sets]
    n = len(param_list)
    if n == 0:
        return []
    result = gradient.run([circuit] * n, [observable] * n, param_list).result()
    return [np.asarray(grad, dtype=float).reshape(-1) for grad in result.gradients]

def full_gradient(gradient, circuit, observable, params) -> np.ndarray:
    """Exact gradient vector for a single parameter vector."""
    return _run_gradients(gradient, circuit, observable, [params])[0]


def gradient_vector_stats(grad: np.ndarray) -> dict[str, float]:
    """Scalar diagnostics from one full gradient vector at a training checkpoint."""
    g = np.asarray(grad, dtype=float).reshape(-1)
    return {
        "grad_norm": float(np.linalg.norm(g)),
        "grad_var_components": float(np.var(g)),
    }


def build_simulation_gradient(
    noise_mode: str,
    *,
    estimator=None,
    precision: float | None = None,
):
    """Gradient primitive for inline training diagnostics (ideal or shot-noise)."""
    mode = noise_mode.strip().lower()
    if mode == "ideal":
        return build_gradient("reverse")
    if mode in {"shot-noise", "shot_noise", "shotnoise"}:
        return build_gradient("lincomb", precision=precision, estimator=estimator)
    raise ValueError(f"Unknown noise_mode '{noise_mode}'. Use 'ideal' or 'shot-noise'.")

def sample_initial_parameters(
    num_params: int,
    *,
    init_dist: str = "uniform",
    init_scale: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw one initial parameter vector.

    - ``uniform``: U(-pi, pi), the Haar-like regime used to expose barren
      plateaus (worst case).
    - ``normal``: N(0, init_scale^2), the near-identity mitigated regime used by
      the training experiments (default scale 0.1).
    """
    rng = rng or np.random.default_rng()
    if init_dist == "uniform":
        return rng.uniform(-np.pi, np.pi, size=num_params)
    if init_dist == "normal":
        return rng.normal(loc=0.0, scale=init_scale, size=num_params)
    raise ValueError(f"Unknown init_dist '{init_dist}'. Use 'uniform' or 'normal'.")

def gradient_variance_at_init(
    ansatz,
    observable,
    *,
    gradient,
    n_samples: int = 300,
    index: int = 0,
    init_dist: str = "uniform",
    init_scale: float = 0.1,
    rng: np.random.Generator | None = None,
) -> dict:
    """Monte Carlo estimate of Var(dC/dtheta_index) at initialization.

    Samples ``n_samples`` random parameter vectors, computes the (exact-rule)
    gradient for each via the Qiskit ``gradient`` primitive, and returns summary
    statistics for the ``index`` component. A barren plateau manifests as this
    variance decaying exponentially with the number of qubits.
    """
    rng = rng or np.random.default_rng()
    num_params = int(ansatz.num_parameters)
    if not (0 <= index < num_params):
        raise ValueError(f"index {index} out of range for num_params={num_params}")

    param_sets = [
        sample_initial_parameters(num_params, init_dist=init_dist, init_scale=init_scale, rng=rng)
        for _ in range(n_samples)
    ]
    grads = _run_gradients(gradient, ansatz, observable, param_sets)
    component = np.array([g[index] for g in grads], dtype=float)

    return {
        "variance": float(np.var(component)),
        "mean": float(np.mean(component)),
        "abs_mean": float(np.mean(np.abs(component))),
        "n_samples": int(n_samples),
        "index": int(index),
        "init_dist": init_dist,
        "samples": component,
    }

def gradient_norm_trajectory(
    ansatz,
    observable,
    param_trajectory: Sequence[np.ndarray],
    *,
    gradient,
    stride: int = 1,
    max_points: int | None = None,
) -> dict:
    """L2 gradient norm along an optimization trajectory (checkpoints).

    ``param_trajectory`` is the per-evaluation list of parameter vectors saved
    when running with ``--track-params``. To keep cost bounded, points are
    subsampled by ``stride`` and/or capped at ``max_points``.
    """
    trajectory = [np.asarray(p, dtype=float) for p in param_trajectory]
    if not trajectory:
        return {"iterations": np.array([], dtype=int), "grad_norm": np.array([], dtype=float)}

    indices = list(range(0, len(trajectory), max(1, stride)))
    if max_points is not None and len(indices) > max_points:
        # Evenly subsample down to max_points checkpoints.
        pick = np.linspace(0, len(indices) - 1, max_points).round().astype(int)
        indices = [indices[i] for i in sorted(set(pick.tolist()))]

    grads = _run_gradients(gradient, ansatz, observable, [trajectory[idx] for idx in indices])
    norms = np.array([float(np.linalg.norm(g)) for g in grads], dtype=float)

    # Report 1-based iteration numbers to match the cost-trajectory convention.
    return {
        "iterations": np.array([idx + 1 for idx in indices], dtype=int),
        "grad_norm": norms,
    }
