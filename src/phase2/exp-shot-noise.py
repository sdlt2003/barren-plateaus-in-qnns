#!/usr/bin/env python3
"""Phase 2: optimizer comparison under Monte Carlo / shot noise."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_algorithms.optimizers import COBYLA, QNSPSA

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hyperparams import (  # noqa: E402
    EARLY_STOPPING_TOLERANCE,
    SIM_DEFAULT_BUDGET_K as DEFAULT_BUDGET_K,
    SHOT_NOISE_DEFAULT_TARGET_PRECISION as DEFAULT_TARGET_PRECISION,
    SIM_EARLY_STOPPING_WINDOW as EARLY_STOPPING_WINDOW,
    SIM_MAX_BUDGET_EVALS as MAX_BUDGET_EVALS,
    MIN_BUDGET_EVALS,
)
from phase2.architectures import available_architectures, build_architecture  # noqa: E402
from utils import (  # noqa: E402
    ConvergenceReached,
    init_trackers,
    make_fidelity,
    make_objective,
    resolve_budget_evals,
    run_optimizer,
    save_optimizer_time_series,
    with_early_stopping,
)


class MonteCarloEstimator:
    def __init__(self, shots: int):
        self.shots = int(shots)

    def run(self, pubs):
        records = []
        for ansatz, observable, parameter_sets in pubs:
            parameters = parameter_sets[0]
            state = Statevector(ansatz.assign_parameters(parameters))
            exact_value = float(np.real(state.expectation_value(observable)))
            exact_value = float(np.clip(exact_value, -1.0, 1.0))
            plus_probability = 0.5 * (1.0 + exact_value)
            plus_probability = float(np.clip(plus_probability, 0.0, 1.0))
            plus_count = np.random.binomial(self.shots, plus_probability)
            sampled_value = (2 * plus_count - self.shots) / self.shots
            records.append(SimpleNamespace(data=SimpleNamespace(evs=np.array([sampled_value]))))
        return SimpleNamespace(result=lambda: records)


def optimizer_compare(
    n_qubits: int,
    outdir: str,
    seed: int | None = None,
    budget_evals: int | None = None,
    budget_k: float = DEFAULT_BUDGET_K,
    window: int = EARLY_STOPPING_WINDOW,
    tol: float = EARLY_STOPPING_TOLERANCE,
    target_precision: float = DEFAULT_TARGET_PRECISION,
    architecture: str = "baseline_hea",
) -> None:
    if seed is not None:
        np.random.seed(seed)

    arch = build_architecture(architecture=architecture, n_qubits=n_qubits)
    ansatz = arch.circuit
    observable = SparsePauliOp.from_list([("I" * arch.readout_qubit + "Z" + "I" * (n_qubits - arch.readout_qubit - 1), 1.0)])
    num_params = ansatz.num_parameters
    budget_evals, budget_mode = resolve_budget_evals(
        num_params=num_params,
        budget_evals=budget_evals,
        budget_k=budget_k,
        min_budget_evals=MIN_BUDGET_EVALS,
        max_budget_evals=MAX_BUDGET_EVALS,
    )
    initial_parameters = np.random.normal(loc=0.0, scale=0.1, size=num_params)
    print(
        f"Architecture={architecture} | Budget mode: {budget_mode} | "
        f"num_params={num_params} | k={budget_k} | budget_evals={budget_evals}"
    )

    ansatz_decomposed = ansatz.decompose()
    num_shots = int(1 / (target_precision**2))
    estimator = MonteCarloEstimator(num_shots)
    history, counts = init_trackers("cobyla", "qnspsa")

    objective_cobyla = with_early_stopping(
        make_objective(
            "cobyla",
            counts=counts,
            history=history,
            estimator=estimator,
            ansatz=ansatz_decomposed,
            observable=observable,
            budget_evals=budget_evals,
        ),
        history=history,
        key="cobyla",
        window=window,
        tolerance=tol,
        stop_exception=ConvergenceReached,
    )
    objective_qnspsa = with_early_stopping(
        make_objective(
            "qnspsa",
            counts=counts,
            history=history,
            estimator=estimator,
            ansatz=ansatz_decomposed,
            observable=observable,
            budget_evals=budget_evals,
        ),
        history=history,
        key="qnspsa",
        window=window,
        tolerance=tol,
        stop_exception=ConvergenceReached,
    )
    fidelity = make_fidelity(
        counts=counts,
        key="qnspsa",
        circuit=ansatz_decomposed,
        budget_evals=budget_evals,
    )

    print(f"Running COBYLA with Monte Carlo noise (approx. {num_shots} shots)...")
    run_optimizer("cobyla", COBYLA(maxiter=budget_evals), objective_cobyla, initial_parameters, counts=counts)
    print(f"Running QNSPSA with Monte Carlo noise (approx. {num_shots} shots)...")
    run_optimizer(
        "qnspsa",
        QNSPSA(fidelity=fidelity, maxiter=budget_evals),
        objective_qnspsa,
        initial_parameters,
        counts=counts,
    )

    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "optimizer_history.npz"), history=history)
    save_optimizer_time_series(history, outdir)
    print("Saved optimizer results to", outdir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 optimizer comparison under Monte Carlo / shot noise")
    parser.add_argument("--n_qubits", type=int, default=12, help="Number of qubits")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output folder")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--architecture",
        type=str,
        default="baseline_hea",
        choices=available_architectures(),
        help="Circuit architecture to benchmark",
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
    parser.add_argument("--window", type=int, default=EARLY_STOPPING_WINDOW)
    parser.add_argument("--tolerance", type=float, default=EARLY_STOPPING_TOLERANCE)
    parser.add_argument(
        "--target-precision",
        type=float,
        default=DEFAULT_TARGET_PRECISION,
        help="Target precision used to simulate Monte Carlo shot noise",
    )
    args = parser.parse_args()

    t0 = time.time()
    optimizer_compare(
        n_qubits=args.n_qubits,
        outdir=args.outdir,
        seed=args.seed,
        architecture=args.architecture,
        budget_evals=args.budget_evals,
        budget_k=args.budget_k,
        window=args.window,
        tol=args.tolerance,
        target_precision=args.target_precision,
    )
    print("Total runtime:", time.time() - t0)


if __name__ == "__main__":
    main()
