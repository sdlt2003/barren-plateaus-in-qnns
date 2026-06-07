#!/usr/bin/env python3
"""Phase 2: optimizer comparisons on real IBM Quantum backends."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_algorithms.optimizers import COBYLA, QNSPSA
from qiskit_ibm_runtime import Estimator as RuntimeEstimator
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime.options import EstimatorOptions
from qiskit_ibm_runtime.utils.validations import validate_isa_circuits

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hyperparams import (  # noqa: E402
    EARLY_STOPPING_TOLERANCE,
    MIN_BUDGET_EVALS,
    REAL_HW_DEFAULT_BUDGET_K as DEFAULT_BUDGET_K,
    REAL_HW_DEFAULT_SHOTS as DEFAULT_SHOTS,
    REAL_HW_EARLY_STOPPING_WINDOW as EARLY_STOPPING_WINDOW,
    REAL_HW_MAX_BUDGET_EVALS as MAX_BUDGET_EVALS,
)
from phase2.architectures import available_architectures, build_architecture, parse_depth_split  # noqa: E402
from utils import (  # noqa: E402
    ConvergenceReached,
    init_trackers,
    make_fidelity,
    make_objective,
    resolve_budget_evals,
    resolve_runtime_outdir,
    run_optimizer,
    save_optimizer_time_series,
    with_early_stopping,
)


def load_env_file(path: str) -> None:
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"").strip("'"))


def get_env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def configure_options(shots: int, resilience_level: int | None) -> EstimatorOptions:
    options = EstimatorOptions()
    options.default_shots = shots
    if resilience_level is not None:
        options.resilience_level = resilience_level
    return options


def build_runtime_service(channel: str, token: str, instance: str | None) -> QiskitRuntimeService:
    channels_to_try = (
        [channel]
        if channel in ["ibm_cloud", "ibm_quantum_platform"]
        else ["ibm_quantum_platform", "ibm_cloud"]
    )
    last_error = None
    for ch in channels_to_try:
        try:
            kwargs: dict[str, str] = {"channel": ch, "token": token}
            if instance:
                kwargs["instance"] = instance
            return QiskitRuntimeService(**kwargs)
        except Exception as exc:  # pragma: no cover - network/runtime-specific branch
            last_error = exc
    raise RuntimeError(f"Failed to connect to IBM Quantum. Last error: {last_error}") from last_error


def select_backend(service: QiskitRuntimeService, backend_name: str | None, n_qubits: int):
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
    runtime_result_timeout: float,
    architecture: str,
    resqnet_depth_split: tuple[int, int],
    resqnet_residual_mode: str,
) -> None:
    outdir = resolve_runtime_outdir(outdir)
    if seed is not None:
        np.random.seed(seed)

    arch = build_architecture(
        architecture=architecture,
        n_qubits=n_qubits,
        resqnet_depth_split=resqnet_depth_split,
        resqnet_residual_mode=resqnet_residual_mode,
    )
    ansatz_logical = arch.circuit
    observable_logical = SparsePauliOp.from_list(
        [("I" * arch.readout_qubit + "Z" + "I" * (n_qubits - arch.readout_qubit - 1), 1.0)]
    )

    load_env_file(".env")
    token = get_env_first("QISKIT_IBM_TOKEN", "API_KEY")
    instance = get_env_first("QISKIT_IBM_INSTANCE", "CRN_KEY")
    if not token:
        raise RuntimeError("Missing IBM token. Set QISKIT_IBM_TOKEN or API_KEY in .env/environment.")

    print("Runtime preflight: creating IBM Runtime service...")
    service = build_runtime_service(channel=channel, token=token, instance=instance)
    print("Runtime preflight: selecting backend...")
    backend = select_backend(service, backend_name, n_qubits)
    print(f"Runtime preflight: selected backend={backend.name}")

    opt_level = 1 if optimization_level is None else optimization_level
    print(f"Transpilation: optimization_level={opt_level}")
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

    resolved_budget_evals, budget_mode = resolve_budget_evals(
        num_params=logical_num_params,
        budget_evals=budget_evals,
        budget_k=budget_k,
        min_budget_evals=MIN_BUDGET_EVALS,
        max_budget_evals=MAX_BUDGET_EVALS,
    )
    print(
        f"Architecture={architecture} | Budget mode: {budget_mode} | num_params={logical_num_params} | "
        f"k={budget_k} | budget_evals={resolved_budget_evals}"
    )
    initial_parameters = np.random.normal(loc=0.0, scale=0.1, size=logical_num_params)

    options = configure_options(shots=shots, resilience_level=resilience_level)
    history, counts = init_trackers("cobyla", "qnspsa")

    run_status = {
        "status": "failed",
        "backend": backend.name,
        "architecture": architecture,
        "runtime_result_timeout_s": runtime_result_timeout,
        "error": None,
    }
    try:
        print("Runtime execution: opening session...")
        with Session(backend=backend) as session:
            estimator = RuntimeEstimator(mode=session, options=options)
            print("Runtime execution: estimator initialized.")
            objective_cobyla = with_early_stopping(
                make_objective(
                    "cobyla",
                    counts=counts,
                    history=history,
                    estimator=estimator,
                    ansatz=ansatz,
                    observable=observable,
                    budget_evals=resolved_budget_evals,
                    result_timeout_s=runtime_result_timeout,
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
                    result_timeout_s=runtime_result_timeout,
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
                circuit=ansatz_logical,
                budget_evals=resolved_budget_evals,
            )

            print(f"Backend: {backend.name} | Architecture: {architecture}")
            run_optimizer(
                "cobyla",
                COBYLA(maxiter=resolved_budget_evals),
                objective_cobyla,
                initial_parameters,
                counts=counts,
            )
            run_optimizer(
                "qnspsa",
                QNSPSA(fidelity=fidelity, maxiter=resolved_budget_evals),
                objective_qnspsa,
                initial_parameters,
                counts=counts,
            )
        run_status["status"] = "completed"
    except Exception as exc:
        run_status["status"] = "failed_runtime"
        run_status["error"] = repr(exc)
        print(f"Runtime execution failed: {exc!r}")
        raise
    finally:
        os.makedirs(outdir, exist_ok=True)
        np.savez(os.path.join(outdir, "optimizer_history.npz"), history=history)
        if history["cobyla"]["cost"] or history["qnspsa"]["cost"]:
            save_optimizer_time_series(history, outdir)
        with open(os.path.join(outdir, "run_status.json"), "w", encoding="utf-8") as handle:
            json.dump(run_status, handle, indent=2, sort_keys=True)
        print("Saved optimizer results to", outdir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 optimizer comparison on IBM Quantum hardware")
    parser.add_argument("--n_qubits", type=int, default=4, help="Number of qubits")
    parser.add_argument("--outdir", type=str, default="outputs/real_run", help="Output folder")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--architecture",
        type=str,
        default="baseline_hea",
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
        help="Shared fixed evaluation budget. If omitted, dynamic budget k*p is used.",
    )
    parser.add_argument(
        "--budget-k",
        type=float,
        default=DEFAULT_BUDGET_K,
        help="Dynamic budget multiplier k used in budget_evals = k * num_params",
    )
    parser.add_argument("--window", type=int, default=EARLY_STOPPING_WINDOW)
    parser.add_argument("--tolerance", type=float, default=EARLY_STOPPING_TOLERANCE)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS, help="Shots per estimator call")
    parser.add_argument("--backend", type=str, default=None, help="Backend name (defaults to least busy)")
    parser.add_argument("--channel", type=str, default=os.environ.get("QISKIT_IBM_CHANNEL", "ibm_quantum"))
    parser.add_argument("--optimization-level", type=int, default=None, help="Transpilation optimization level")
    parser.add_argument("--resilience-level", type=int, default=None, help="Runtime resilience level")
    parser.add_argument(
        "--runtime-result-timeout",
        type=float,
        default=120.0,
        help="Maximum seconds to wait for each Estimator result() call before raising timeout.",
    )
    args = parser.parse_args()

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
        window=args.window,
        tol=args.tolerance,
        shots=args.shots,
        backend_name=args.backend,
        channel=args.channel,
        optimization_level=args.optimization_level,
        resilience_level=args.resilience_level,
        runtime_result_timeout=args.runtime_result_timeout,
    )
    print("Total runtime:", time.time() - t0)


if __name__ == "__main__":
    main()
