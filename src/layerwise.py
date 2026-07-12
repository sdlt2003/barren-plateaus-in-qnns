#!/usr/bin/env python3
"""Layer-wise Learning training dynamic.

Layer-wise Learning is a training *strategy* (not an optimizer): instead of
optimizing all parameters simultaneously, it trains the circuit layer by layer,
freezing previously trained layers. By keeping a small number of free
parameters and a low effective depth active at each step, it aims to keep the
gradient variance polynomial and avoid reaching the Haar (2-design) regime.

This module wraps any fidelity-free registry optimizer (e.g. ``cobyla`` or
``nft``) as the inner per-layer optimizer and records a single, continuous cost
trajectory under a history key like ``layerwise_nft`` so the standard analysis
and plotting code treats it as one more method.
"""
from __future__ import annotations

import numpy as np

from optimizers import get_optimizer_spec
from utils.utils import (
    BudgetExceeded,
    make_objective,
)


class _BestTracker:
    """Keeps the lowest-cost full parameter vector seen during training."""

    def __init__(self, initial_full: np.ndarray):
        self.best_cost = np.inf
        self.best_full = np.array(initial_full, dtype=float).copy()

    def update(self, cost: float, full: np.ndarray) -> None:
        if cost < self.best_cost:
            self.best_cost = float(cost)
            self.best_full = np.array(full, dtype=float).copy()


def _make_masked_objective(base_objective, theta_full, active_indices, tracker):
    """Objective over the active subset, mapping into the frozen full vector."""
    active = list(active_indices)

    def masked(reduced_params):
        full = theta_full.copy()
        for i, idx in enumerate(active):
            full[idx] = reduced_params[i]
        cost = base_objective(full)
        tracker.update(cost, full)
        return cost

    return masked


def run_layerwise_training(
    *,
    inner_optimizer: str,
    layer_groups: list[list[int]],
    estimator,
    ansatz,
    observable,
    budget_evals: int,
    initial_parameters,
    history: dict,
    counts: dict,
    key: str,
    result_timeout_s: float | None = None,
    track_params: bool = False,
    gradient=None,
    track_grad_norm: bool = False,
    track_grad_var: bool = False,
    grad_stride: int = 10,
) -> np.ndarray:
    """Train ``ansatz`` layer by layer, freezing previously trained layers.

    Returns the final full parameter vector (best seen). The shared evaluation
    budget is split evenly across layers; the global cap in ``make_objective``
    still bounds the total number of evaluations under ``key``.
    """
    spec = get_optimizer_spec(inner_optimizer)
    if spec.needs_fidelity:
        raise ValueError(
            f"Layer-wise inner optimizer '{inner_optimizer}' requires a fidelity "
            "callback, which is not supported over the masked parameter subspace. "
            "Use a fidelity-free optimizer such as 'cobyla' or 'nft'."
        )

    groups = [list(g) for g in layer_groups if g]
    if not groups:
        raise ValueError("layer_groups must contain at least one non-empty group.")

    theta_full = np.array(initial_parameters, dtype=float).copy()
    tracker = _BestTracker(theta_full)

    base_objective = make_objective(
        key,
        counts=counts,
        history=history,
        estimator=estimator,
        ansatz=ansatz,
        observable=observable,
        budget_evals=budget_evals,
        result_timeout_s=result_timeout_s,
        track_params=track_params,
        gradient=gradient,
        track_grad_norm=track_grad_norm,
        track_grad_var=track_grad_var,
        grad_stride=grad_stride,
    )

    per_layer_budget = max(1, budget_evals // len(groups))

    for layer_idx, active in enumerate(groups):
        print(
            f"[layerwise:{key}] layer {layer_idx + 1}/{len(groups)} "
            f"| active_params={len(active)} | per_layer_budget={per_layer_budget}"
        )
        x0_reduced = theta_full[np.array(active, dtype=int)]
        masked = _make_masked_objective(base_objective, theta_full, active, tracker)
        optimizer = spec.build(per_layer_budget, None)
        try:
            optimizer.minimize(fun=masked, x0=np.array(x0_reduced, dtype=float))
        except BudgetExceeded:
            print(f"[layerwise:{key}] global budget reached at layer {layer_idx + 1}.")

        # Freeze this layer at the best full vector found so far.
        theta_full = tracker.best_full.copy()

        if counts.get(key, 0) >= budget_evals:
            print(f"[layerwise:{key}] stopping: global budget {budget_evals} exhausted.")
            break

    return theta_full
