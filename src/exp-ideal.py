#!/usr/bin/env python3
"""
Optimizer comparison (portado desde QNSPSA-vs-COBYLA.ipynb).
Usage:
  python exp-ideal.py --n_qubits 8 --outdir outputs/opt_run --seed 7
"""
import argparse
import os
import time
import numpy as np

from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.optimizers import COBYLA, QNSPSA

from hyperparams import (
    EARLY_STOPPING_TOLERANCE,
    IDEAL_DEFAULT_BUDGET_K as DEFAULT_BUDGET_K,
    IDEAL_EARLY_STOPPING_WINDOW as EARLY_STOPPING_WINDOW,
    IDEAL_MAX_BUDGET_EVALS as MAX_BUDGET_EVALS,
    MIN_BUDGET_EVALS,
)
from utils import (
    ConvergenceReached,
    init_trackers,
    make_objective,
    with_early_stopping,
    make_fidelity,
    run_optimizer,
    resolve_budget_evals,
)


# --- Core experiment: optimizer comparison ---
def optimizer_compare(
    n_qubits,
    outdir,
    seed=None,
    budget_evals=None,
    budget_k=DEFAULT_BUDGET_K,
    window=EARLY_STOPPING_WINDOW,
    tol=EARLY_STOPPING_TOLERANCE,
):
    if seed is not None:
        np.random.seed(seed)

    reps_log = max(1, int(np.round(np.log2(n_qubits))))                                 # depth logarítmico
    ansatz = efficient_su2(num_qubits=n_qubits, reps=reps_log, entanglement='linear')   # ansatz con profundidad logarítmica
    observable = SparsePauliOp.from_list([("I" * (n_qubits - 1) + "Z", 1)])             # local observable
    num_params = ansatz.num_parameters                                                  
    budget_evals, budget_mode = resolve_budget_evals(
        num_params=num_params,
        budget_evals=budget_evals,
        budget_k=budget_k,
        min_budget_evals=MIN_BUDGET_EVALS,
        max_budget_evals=MAX_BUDGET_EVALS,
    )
    initial_parameters = np.random.normal(loc=0.0, scale=0.1, size=num_params)          # Identity block Initialization
    print(
        f"Budget mode: {budget_mode} | num_params={num_params} | "
        f"k={budget_k} | budget_evals={budget_evals}"
    )

    estimator = StatevectorEstimator()                                                  
    history, counts = init_trackers("cobyla", "qnspsa")

    objective_cobyla = with_early_stopping(
        make_objective("cobyla", 
                        counts=counts,
                        history=history,
                        estimator=estimator,
                        ansatz=ansatz,
                        observable=observable,
                        budget_evals=budget_evals),
        history=history,
        key="cobyla",
        window=window,
        tolerance=tol,
        stop_exception=ConvergenceReached
    )
    objective_qnspsa = with_early_stopping(
        make_objective("qnspsa",
                        counts=counts,
                        history=history,
                        estimator=estimator,
                        ansatz=ansatz,
                        observable=observable,
                        budget_evals=budget_evals),
        history=history, 
        key="qnspsa",
        window=window,
        tolerance=tol,
        stop_exception=ConvergenceReached
    )

    fidelity = make_fidelity(counts=counts, key="qnspsa", circuit=ansatz, budget_evals=budget_evals)

    # Run optimizers
    run_optimizer(  "cobyla",
                    COBYLA(maxiter=budget_evals),
                    objective_cobyla,
                    initial_parameters,
                    counts=counts)
    run_optimizer(  "qnspsa",
                    QNSPSA(fidelity=fidelity, maxiter=budget_evals),
                    objective_qnspsa,
                    initial_parameters,
                    counts=counts)

    # Save results and a plot
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "optimizer_history.npz"), history=history)
    from utils import save_optimizer_time_series

    save_optimizer_time_series(history, outdir)
    print("Saved optimizer results to", outdir)


# --- CLI ---
def main():
    p = argparse.ArgumentParser(description="Optimizer comparison runner")
    p.add_argument("--n_qubits", type=int, default=8, help="Number of qubits")
    p.add_argument("--outdir", type=str, default="outputs", help="Output folder")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--budget-evals",
        type=int,
        default=None,
        help="Shared fixed evaluation budget for each optimizer. If omitted, dynamic budget k*p is used.",
    )
    p.add_argument(
        "--budget-k",
        type=float,
        default=DEFAULT_BUDGET_K,
        help="Dynamic budget multiplier k used in budget_evals = k * num_params",
    )
    p.add_argument("--window", type=int, default=EARLY_STOPPING_WINDOW)
    p.add_argument("--tolerance", type=float, default=EARLY_STOPPING_TOLERANCE)
    args = p.parse_args()

    t0 = time.time()
    optimizer_compare(args.n_qubits, args.outdir, seed=args.seed,
                     budget_evals=args.budget_evals,
                     budget_k=args.budget_k,
                     window=args.window, tol=args.tolerance)
    print("Total runtime:", time.time() - t0)

if __name__ == "__main__":
    main()
