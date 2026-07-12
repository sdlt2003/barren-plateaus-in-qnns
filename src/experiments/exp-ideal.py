#!/usr/bin/env python3
"""Phase 2: optimizer comparison on ideal statevector backend."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hyperparams import (  # noqa: E402
    SIM_DEFAULT_BUDGET_K as DEFAULT_BUDGET_K,
    SIM_GRAD_CHECKPOINT_STRIDE as DEFAULT_GRAD_STRIDE,
    SIM_MAX_BUDGET_EVALS as MAX_BUDGET_EVALS,
)
from gradients import build_simulation_gradient  # noqa: E402
from optimizers import DEFAULT_OPTIMIZERS, parse_optimizer_list  # noqa: E402
from layerwise import run_layerwise_training  # noqa: E402
from utils.architectures import available_architectures, build_architecture, parse_depth_split  # noqa: E402
from utils.utils import (  # noqa: E402
    init_trackers,
    resolve_budget_evals,
    run_optimizer_suite,
    save_optimizer_time_series,
)


def optimizer_compare(
    n_qubits: int,
    outdir: str,
    seed: int | None = None,
    budget_evals: int | None = None,
    budget_k: float = DEFAULT_BUDGET_K,
    architecture: str = "baseline",
    resqnet_depth_split: tuple[int, int] = (5, 1),
    resqnet_residual_mode: str = "structural",
    optimizers: list[str] | tuple[str, ...] = DEFAULT_OPTIMIZERS,
    track_params: bool = False,
    track_grad_norm: bool = False,
    track_grad_var: bool = False,
    grad_stride: int = DEFAULT_GRAD_STRIDE,
    training_mode: str = "full",
    layerwise_inner: str = "nft",
) -> None:
    if seed is not None:
        np.random.seed(seed)

    arch = build_architecture(
        architecture=architecture,
        n_qubits=n_qubits,
        resqnet_depth_split=resqnet_depth_split,
        resqnet_residual_mode=resqnet_residual_mode,
    )
    ansatz = arch.circuit
    observable = SparsePauliOp.from_list([("I" * arch.readout_qubit + "Z" + "I" * (n_qubits - arch.readout_qubit - 1), 1.0)])
    num_params = ansatz.num_parameters
    budget_evals, budget_mode = resolve_budget_evals(
        num_params=num_params,
        budget_evals=budget_evals,
        budget_k=budget_k,
        min_budget_evals=1,
        max_budget_evals=MAX_BUDGET_EVALS,
    )
    initial_parameters = np.random.normal(loc=0.0, scale=0.1, size=num_params)
    estimator = StatevectorEstimator()

    gradient = None
    if track_grad_norm or track_grad_var:
        gradient = build_simulation_gradient("ideal")

    grad_note = ""
    if track_grad_norm or track_grad_var:
        parts = []
        if track_grad_norm:
            parts.append("||grad||")
        if track_grad_var:
            parts.append("Var_i(grad)")
        grad_note = f" | grad_checkpoints={'+'.join(parts)} stride={grad_stride}"

    if training_mode == "layerwise":
        key = f"layerwise_{layerwise_inner}"
        print(
            f"Architecture={architecture} | Budget mode: {budget_mode} | "
            f"num_params={num_params} | k={budget_k} | budget_evals={budget_evals} | "
            f"training_mode=layerwise | inner={layerwise_inner}{grad_note}"
        )
        history, counts = init_trackers(key)
        run_layerwise_training(
            inner_optimizer=layerwise_inner,
            layer_groups=arch.resolved_layer_groups(),
            estimator=estimator,
            ansatz=ansatz,
            observable=observable,
            budget_evals=budget_evals,
            initial_parameters=initial_parameters,
            history=history,
            counts=counts,
            key=key,
            track_params=track_params,
            gradient=gradient,
            track_grad_norm=track_grad_norm,
            track_grad_var=track_grad_var,
            grad_stride=grad_stride,
        )
    else:
        print(
            f"Architecture={architecture} | Budget mode: {budget_mode} | "
            f"num_params={num_params} | k={budget_k} | budget_evals={budget_evals} | "
            f"optimizers={list(optimizers)}{grad_note}"
        )
        history, counts = init_trackers(*optimizers)
        run_optimizer_suite(
            optimizers,
            estimator=estimator,
            ansatz=ansatz,
            observable=observable,
            budget_evals=budget_evals,
            initial_parameters=initial_parameters,
            history=history,
            counts=counts,
            track_params=track_params,
            gradient=gradient,
            track_grad_norm=track_grad_norm,
            track_grad_var=track_grad_var,
            grad_stride=grad_stride,
        )

    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "optimizer_history.npz"), history=history)
    save_optimizer_time_series(history, outdir)
    print("Saved optimizer results to", outdir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 optimizer comparison runner (ideal)")
    parser.add_argument("--n_qubits", type=int, default=8, help="Number of qubits")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output folder")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--architecture",
        type=str,
        default="baseline",
        choices=available_architectures(),
        help="Circuit architecture to benchmark",
    )
    parser.add_argument(
        "--resqnet-depth-split",
        type=str,
        default="5,1",
        help="Depth split for resqnet as 'D1,D2' (e.g. '5,1'). Ignored for other architectures.",
    )
    parser.add_argument(
        "--resqnet-residual-mode",
        type=str,
        default="structural",
        help="Residual mode for resqnet. Current supported value: structural.",
    )
    parser.add_argument(
        "--budget-evals",
        type=int,
        default=None,
        help="Shared fixed evaluation budget for each optimizer. If omitted, dynamic budget k*p is used.",
    )
    parser.add_argument(
        "--budget-k",
        type=float,
        default=DEFAULT_BUDGET_K,
        help="Dynamic budget multiplier k used in budget_evals = k * num_params",
    )
    parser.add_argument(
        "--optimizers",
        type=str,
        default=",".join(DEFAULT_OPTIMIZERS),
        help="Comma-separated optimizers to run, e.g. 'cobyla,qnspsa,nft'.",
    )
    parser.add_argument(
        "--track-params",
        action="store_true",
        help="Store the full parameter vector at every cost evaluation.",
    )
    parser.add_argument(
        "--track-grad",
        action="store_true",
        help="At gradient checkpoints, store ||grad C||_2 and Var_i(dC/dtheta_i) in the .npz.",
    )
    parser.add_argument(
        "--track-grad-norm",
        action="store_true",
        help="At gradient checkpoints, store ||grad C||_2 (extra primitive call; not counted in budget).",
    )
    parser.add_argument(
        "--track-grad-var",
        action="store_true",
        help="At gradient checkpoints, store Var_i(dC/dtheta_i) across gradient components.",
    )
    parser.add_argument(
        "--grad-stride",
        type=int,
        default=DEFAULT_GRAD_STRIDE,
        help="Record gradient diagnostics every N cost evaluations (always includes eval 1).",
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        default="full",
        choices=["full", "layerwise"],
        help="'full' optimizes all parameters; 'layerwise' trains layer by layer.",
    )
    parser.add_argument(
        "--layerwise-inner",
        type=str,
        default="nft",
        help="Inner (fidelity-free) optimizer used per layer in layerwise mode.",
    )
    args = parser.parse_args()

    track_grad_norm = args.track_grad_norm or args.track_grad
    track_grad_var = args.track_grad_var or args.track_grad

    t0 = time.time()
    optimizer_compare(
        n_qubits=args.n_qubits,
        outdir=args.outdir,
        seed=args.seed,
        architecture=args.architecture,
        resqnet_depth_split=parse_depth_split(args.resqnet_depth_split),
        resqnet_residual_mode=args.resqnet_residual_mode,
        budget_evals=args.budget_evals,
        budget_k=args.budget_k,
        optimizers=parse_optimizer_list(args.optimizers),
        track_params=args.track_params,
        track_grad_norm=track_grad_norm,
        track_grad_var=track_grad_var,
        grad_stride=args.grad_stride,
        training_mode=args.training_mode,
        layerwise_inner=args.layerwise_inner,
    )
    print("Total runtime:", time.time() - t0)


if __name__ == "__main__":
    main()
