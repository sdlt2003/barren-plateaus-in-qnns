#!/usr/bin/env python3
"""
Run optimizer comparisons on a real IBM Quantum backend using Runtime Estimator.

Requires credentials in .env or environment variables:
  - QISKIT_IBM_TOKEN (required)
  - QISKIT_IBM_CHANNEL (optional, default: ibm_quantum)
  - QISKIT_IBM_INSTANCE (optional)

Usage:
  python src/exp-real.py --n_qubits 4 --outdir outputs/real_run --seed 7 --backend ibm_oslo
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_algorithms.optimizers import COBYLA, QNSPSA
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Estimator as RuntimeEstimator
from qiskit_ibm_runtime.options import EstimatorOptions
from qiskit_ibm_runtime.utils.validations import validate_isa_circuits

from hyperparams import (
    EARLY_STOPPING_TOLERANCE,
    REAL_HW_DEFAULT_BUDGET_K as DEFAULT_BUDGET_K,
    REAL_HW_EARLY_STOPPING_WINDOW as EARLY_STOPPING_WINDOW,
    REAL_HW_DEFAULT_SHOTS as DEFAULT_SHOTS,
    REAL_HW_MAX_BUDGET_EVALS as MAX_BUDGET_EVALS,
    MIN_BUDGET_EVALS,
)
from utils import (
    ConvergenceReached,
    init_trackers,
    make_objective,
    with_early_stopping,
    make_fidelity,
    run_optimizer,
    save_optimizer_time_series,
    resolve_budget_evals,
)


def load_env_file(path: str) -> None:
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"").strip("'")
            os.environ.setdefault(key, value)


def get_env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def configure_options(shots: int, optimization_level: int | None, resilience_level: int | None) -> EstimatorOptions:
    # qiskit-ibm-runtime>=0.46 uses EstimatorV2 + EstimatorOptions.
    # Shots are configured via default_shots (not OptionsV1/OptionsV2.shots).
    options = EstimatorOptions()
    options.default_shots = shots

    # EstimatorOptions does not expose transpilation optimization level; keep arg for CLI compatibility.
    if resilience_level is not None:
        options.resilience_level = resilience_level
    return options


def build_runtime_service(channel: str, token: str, instance: str | None) -> QiskitRuntimeService:
    """Create QiskitRuntimeService, trying both common channels if needed."""
    channels_to_try = [channel] if channel in ["ibm_cloud", "ibm_quantum_platform"] else ["ibm_quantum_platform", "ibm_cloud"]
    last_error = None
    for ch in channels_to_try:
        try:
            kwargs = {"channel": ch, "token": token}
            if instance:
                kwargs["instance"] = instance
            return QiskitRuntimeService(**kwargs)
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Failed to connect to IBM Quantum. Last error: {last_error}") from last_error


def select_backend(service: QiskitRuntimeService, backend_name: str | None, n_qubits: int):
    """Select a backend by name or use least_busy."""
    if backend_name:
        return service.backend(backend_name)
    return service.least_busy(simulator=False, min_num_qubits=n_qubits)



def optimizer_compare(
    n_qubits: int,
    outdir: str,
    seed: int | None,
    budget_evals: int | None,
    budget_k: float | None,
    window: int,
    tol: float,
    shots: int,
    backend_name: str | None,
    channel: str,
    optimization_level: int | None,
    resilience_level: int | None,
):
    if seed is not None:
        np.random.seed(seed)

    reps_log = max(1, int(np.round(np.log2(n_qubits))))
    ansatz_logical = efficient_su2(num_qubits=n_qubits, reps=reps_log, entanglement="linear")
    observable_logical = SparsePauliOp.from_list([("I" * (n_qubits - 1) + "Z", 1)])

    load_env_file(".env")
    token = get_env_first("QISKIT_IBM_TOKEN", "API_KEY")
    instance = get_env_first("QISKIT_IBM_INSTANCE", "CRN_KEY")
    
    # Create service directly (tries both channels if needed)
    service = build_runtime_service(channel=channel, token=token, instance=instance)
    backend = select_backend(service, backend_name, n_qubits)

    # Runtime requires ISA circuits. Transpile ansatz to the backend target and
    # apply the resulting layout to the observable.
    opt_level = 1 if optimization_level is None else optimization_level
    pm = generate_preset_pass_manager(backend=backend, optimization_level=opt_level)
    ansatz = pm.run(ansatz_logical)
    observable = observable_logical.apply_layout(ansatz.layout)
    validate_isa_circuits([ansatz], backend.target)

    logical_num_params = ansatz_logical.num_parameters
    isa_num_params = ansatz.num_parameters
    if logical_num_params != isa_num_params:
        raise RuntimeError(
            "Parameter mismatch between logical and ISA ansatz "
            f"(logical={logical_num_params}, isa={isa_num_params})."
        )

    num_params = logical_num_params
    resolved_budget_evals, budget_mode = resolve_budget_evals(
        num_params=num_params,
        budget_evals=budget_evals,
        budget_k=budget_k,
        min_budget_evals=MIN_BUDGET_EVALS,
        max_budget_evals=MAX_BUDGET_EVALS,
    )
    print(
        f"Budget mode: {budget_mode} | num_params={num_params} | "
        f"k={budget_k} | budget_evals={resolved_budget_evals}"
    )
    initial_parameters = np.random.normal(loc=0.0, scale=0.1, size=num_params)

    options = configure_options(shots, optimization_level, resilience_level)

    history, counts = init_trackers("cobyla", "qnspsa")

    with Session(backend=backend) as session:
        estimator = RuntimeEstimator(mode=session, options=options)

        objective_cobyla = with_early_stopping(
            make_objective(
                "cobyla",
                counts=counts,
                history=history,
                estimator=estimator,
                ansatz=ansatz,
                observable=observable,
                budget_evals=resolved_budget_evals,
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
                ansatz=ansatz,
                observable=observable,
                budget_evals=resolved_budget_evals,
            ),
            history=history,
            key="qnspsa",
            window=window,
            tolerance=tol,
            stop_exception=ConvergenceReached,
        )

        # QNSPSA needs a fidelity function. Keep this on the logical ansatz to
        # avoid huge statevectors from backend ISA layouts.
        fidelity = make_fidelity(
            counts=counts,
            key="qnspsa",
            circuit=ansatz_logical,
            budget_evals=resolved_budget_evals,
        )

        print(f"Backend: {backend.name}")
        print(
            f"Fidelity circuit qubits: logical={ansatz_logical.num_qubits}, "
            f"isa={ansatz.num_qubits}"
        )
        print(f"Running COBYLA on hardware ({shots} shots)...")
        run_optimizer(
            "cobyla",
            COBYLA(maxiter=resolved_budget_evals),
            objective_cobyla,
            initial_parameters,
            counts=counts,
        )

        print(f"Running QNSPSA on hardware ({shots} shots)...")
        run_optimizer(
            "qnspsa",
            QNSPSA(fidelity=fidelity, maxiter=resolved_budget_evals),
            objective_qnspsa,
            initial_parameters,
            counts=counts,
        )

    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "optimizer_history.npz"), history=history)
    save_optimizer_time_series(history, outdir)
    print("Saved optimizer results to", outdir)


def main() -> None:
    p = argparse.ArgumentParser(description="Run optimizer comparison on real IBM Quantum hardware")
    p.add_argument("--n_qubits", type=int, default=4, help="Number of qubits")
    p.add_argument("--outdir", type=str, default="outputs/real_run", help="Output folder")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--budget-evals",
        type=int,
        default=None,
        help="Shared fixed evaluation budget. If omitted, dynamic budget k*p is used.",
    )
    p.add_argument(
        "--budget-k",
        type=float,
        default=DEFAULT_BUDGET_K,
        help="Dynamic budget multiplier k used in budget_evals = k * num_params",
    )
    p.add_argument("--window", type=int, default=EARLY_STOPPING_WINDOW)
    p.add_argument("--tolerance", type=float, default=EARLY_STOPPING_TOLERANCE)
    p.add_argument("--shots", type=int, default=DEFAULT_SHOTS, help="Shots per estimator call")
    p.add_argument("--backend", type=str, default=None, help="Backend name (defaults to least busy)")
    p.add_argument("--channel", type=str, default=os.environ.get("QISKIT_IBM_CHANNEL", "ibm_quantum"))
    p.add_argument("--optimization-level", type=int, default=None, help="Transpilation optimization level")
    p.add_argument("--resilience-level", type=int, default=None, help="Runtime resilience level")
    args = p.parse_args()

    t0 = time.time()
    optimizer_compare(
        n_qubits=args.n_qubits,
        outdir=args.outdir,
        seed=args.seed,
        budget_evals=args.budget_evals,
        budget_k=args.budget_k,
        window=args.window,
        tol=args.tolerance,
        shots=args.shots,
        backend_name=args.backend,
        channel=args.channel,
        optimization_level=args.optimization_level,
        resilience_level=args.resilience_level,
    )
    print("Total runtime:", time.time() - t0)


if __name__ == "__main__":
    main()
