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
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.options import EstimatorOptions
from qiskit_ibm_runtime.utils.validations import validate_isa_circuits

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hyperparams import (  # noqa: E402
    EARLY_STOPPING_TOLERANCE,
    MIN_BUDGET_EVALS,
    REAL_HW_DEFAULT_BUDGET_K as DEFAULT_BUDGET_K,
    REAL_HW_FIXED_BUDGET_BASE,
    REAL_HW_DEFAULT_SHOTS as DEFAULT_SHOTS,
    REAL_HW_EARLY_STOPPING_WINDOW as EARLY_STOPPING_WINDOW,
    REAL_HW_MAX_BUDGET_EVALS as MAX_BUDGET_EVALS,
    REAL_HW_EXECUTION_MODE as DEFAULT_EXECUTION_MODE,
    REAL_HW_RUNTIME_RESULT_TIMEOUT as DEFAULT_RUNTIME_RESULT_TIMEOUT,
    REAL_HW_RUNTIME_MAX_TIME as DEFAULT_RUNTIME_MAX_TIME,
    REAL_HW_SESSION_MAX_TIME as DEFAULT_SESSION_MAX_TIME,
)
from utils_runtime import (  # noqa: E402
    describe_runtime_mode,
    normalize_execution_mode,
    normalize_runtime_max_time,
    runtime_execution_mode,
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
    return service.least_busy(operational=True, simulator=False, min_num_qubits=n_qubits)


def real_hw_fixed_budget_evals(n_qubits: int) -> int:
    """Legacy fixed budget by qubits only: base * 2^((Q-4)/4)."""
    return REAL_HW_FIXED_BUDGET_BASE * (2 ** ((n_qubits - 4) // 4))


def parse_qubit_sizes(raw: str) -> list[int]:
    sizes = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not sizes:
        raise ValueError("qubit_sizes must contain at least one integer")
    return sizes


def _load_ibm_credentials() -> tuple[str, str | None]:
    load_env_file(".env")
    token = get_env_first("QISKIT_IBM_TOKEN", "API_KEY")
    instance = get_env_first("QISKIT_IBM_INSTANCE", "CRN_KEY")
    if not token:
        raise RuntimeError("Missing IBM token. Set QISKIT_IBM_TOKEN or API_KEY in .env/environment.")
    return token, instance


def _point_is_completed(outdir: str) -> bool:
    status_path = os.path.join(resolve_runtime_outdir(outdir), "run_status.json")
    if not os.path.isfile(status_path):
        return False
    with open(status_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("status") == "completed"


def _save_run_results(outdir: str, history: dict, run_status: dict) -> None:
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "optimizer_history.npz"), history=history)
    if history["cobyla"]["cost"] or history["qnspsa"]["cost"]:
        save_optimizer_time_series(history, outdir)
    with open(os.path.join(outdir, "run_status.json"), "w", encoding="utf-8") as handle:
        json.dump(run_status, handle, indent=2, sort_keys=True)
    print("Saved optimizer results to", outdir)


def run_qubit_point(
    *,
    estimator: RuntimeEstimator,
    backend,
    pm,
    n_qubits: int,
    outdir: str,
    seed: int | None,
    budget_evals: int | None,
    budget_k: float | None,
    window: int,
    tol: float,
    runtime_result_timeout: float | None,
    architecture: str,
    execution_mode: str,
    resqnet_depth_split: tuple[int, int],
    resqnet_residual_mode: str,
    optimizer: str = "both",
) -> bool:
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

    print(f"Transpilation: qubits={n_qubits}")
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
        f"Architecture={architecture} | qubits={n_qubits} | Budget mode: {budget_mode} | "
        f"num_params={logical_num_params} | k={budget_k} | budget_evals={resolved_budget_evals}"
    )
    initial_parameters = np.random.normal(loc=0.0, scale=0.1, size=logical_num_params)

    history, counts = init_trackers("cobyla", "qnspsa")
    run_status = {
        "status": "failed",
        "backend": backend.name,
        "architecture": architecture,
        "execution_mode": execution_mode,
        "optimizer": optimizer,
        "runtime_result_timeout_s": runtime_result_timeout,
        "error": None,
    }
    try:
        print(f"Backend: {backend.name} | Architecture: {architecture} | qubits={n_qubits}")

        if optimizer in ("both", "cobyla"):
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
            run_optimizer(
                "cobyla",
                COBYLA(maxiter=resolved_budget_evals),
                objective_cobyla,
                initial_parameters,
                counts=counts,
            )

        if optimizer in ("both", "qnspsa"):
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
            run_optimizer(
                "qnspsa",
                QNSPSA(fidelity=fidelity, maxiter=resolved_budget_evals),
                objective_qnspsa,
                initial_parameters,
                counts=counts,
            )

        run_status["status"] = "completed"
        return True
    except Exception as exc:
        run_status["status"] = "failed_runtime"
        run_status["error"] = repr(exc)
        print(f"Runtime execution failed for qubits={n_qubits}: {exc!r}")
        return False
    finally:
        _save_run_results(outdir, history, run_status)


def run_single_point_hw(
    *,
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
    runtime_result_timeout: float | None,
    runtime_max_time: str | None,
    execution_mode: str,
    architecture: str,
    resqnet_depth_split: tuple[int, int],
    resqnet_residual_mode: str,
    optimizer: str = "both",
    skip_completed: bool = False,
) -> bool:
    """Run one (seed, qubits) point inside a single IBM session/batch/job container."""
    outdir = resolve_runtime_outdir(outdir)
    if skip_completed and _point_is_completed(outdir):
        print(f"Skipping completed point: seed={seed} qubits={n_qubits} outdir={outdir}")
        return True

    mode = normalize_execution_mode(execution_mode)
    runtime_max_time = normalize_runtime_max_time(runtime_max_time)
    token, instance = _load_ibm_credentials()

    print("Runtime preflight: creating IBM Runtime service...")
    service = build_runtime_service(channel=channel, token=token, instance=instance)
    print("Runtime preflight: selecting backend...")
    backend = select_backend(service, backend_name, n_qubits)
    print(f"Runtime preflight: selected backend={backend.name}")

    opt_level = 1 if optimization_level is None else optimization_level
    pm = generate_preset_pass_manager(backend=backend, optimization_level=opt_level)
    options = configure_options(shots=shots, resilience_level=resilience_level)

    print(f"Runtime execution: {describe_runtime_mode(mode, runtime_max_time)}")
    print(
        f"Runtime execution: result timeout="
        f"{'disabled (IBM default wait)' if runtime_result_timeout is None else runtime_result_timeout}s"
    )
    with runtime_execution_mode(backend, mode=mode, max_time=runtime_max_time) as container:
        print(f"Runtime execution: {mode} container ready.")
        estimator = RuntimeEstimator(mode=container, options=options)
        return run_qubit_point(
            estimator=estimator,
            backend=backend,
            pm=pm,
            n_qubits=n_qubits,
            outdir=outdir,
            seed=seed,
            budget_evals=budget_evals,
            budget_k=budget_k,
            window=window,
            tol=tol,
            runtime_result_timeout=runtime_result_timeout,
            architecture=architecture,
            execution_mode=mode,
            resqnet_depth_split=resqnet_depth_split,
            resqnet_residual_mode=resqnet_residual_mode,
            optimizer=optimizer,
        )


def run_seed_session_grid(
    *,
    seed: int,
    run_root: str,
    qubit_sizes: list[int],
    budget_evals: int | None,
    budget_k: float | None,
    window: int,
    tol: float,
    shots: int,
    backend_name: str | None,
    channel: str,
    optimization_level: int | None,
    resilience_level: int | None,
    runtime_result_timeout: float | None,
    runtime_max_time: str | None,
    execution_mode: str,
    architecture: str,
    resqnet_depth_split: tuple[int, int],
    resqnet_residual_mode: str,
    skip_completed: bool = False,
) -> None:
    """Legacy: multiple qubit sizes in one IBM container (not recommended for long grids)."""
    run_root = resolve_runtime_outdir(run_root)
    os.makedirs(run_root, exist_ok=True)

    completed = 0
    failed = 0
    skipped = 0
    for n_qubits in qubit_sizes:
        outdir = os.path.join(run_root, f"seed_{seed}", f"qubits_{n_qubits}")
        print(f"\n========== seed={seed} qubits={n_qubits} ==========")
        if skip_completed and _point_is_completed(outdir):
            print(f"Skipping completed point: {outdir}")
            skipped += 1
            continue
        ok = run_single_point_hw(
            n_qubits=n_qubits,
            outdir=outdir,
            seed=seed,
            budget_evals=budget_evals,
            budget_k=budget_k,
            window=window,
            tol=tol,
            shots=shots,
            backend_name=backend_name,
            channel=channel,
            optimization_level=optimization_level,
            resilience_level=resilience_level,
            runtime_result_timeout=runtime_result_timeout,
            runtime_max_time=runtime_max_time,
            execution_mode=execution_mode,
            architecture=architecture,
            resqnet_depth_split=resqnet_depth_split,
            resqnet_residual_mode=resqnet_residual_mode,
            skip_completed=False,
        )
        if ok:
            completed += 1
        else:
            failed += 1

    print(
        f"Seed {seed} grid complete: completed={completed}, failed={failed}, "
        f"skipped={skipped}, total={len(qubit_sizes)}"
    )


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
    runtime_result_timeout: float | None,
    runtime_max_time: str | None,
    execution_mode: str,
    architecture: str,
    resqnet_depth_split: tuple[int, int],
    resqnet_residual_mode: str,
    optimizer: str = "both",
    skip_completed: bool = False,
) -> None:
    """Run a single (seed, qubits) point (CLI / Slurm batch task)."""
    ok = run_single_point_hw(
        n_qubits=n_qubits,
        outdir=outdir,
        seed=seed,
        budget_evals=budget_evals,
        budget_k=budget_k,
        window=window,
        tol=tol,
        shots=shots,
        backend_name=backend_name,
        channel=channel,
        optimization_level=optimization_level,
        resilience_level=resilience_level,
        runtime_result_timeout=runtime_result_timeout,
        runtime_max_time=runtime_max_time,
        execution_mode=execution_mode,
        architecture=architecture,
        resqnet_depth_split=resqnet_depth_split,
        resqnet_residual_mode=resqnet_residual_mode,
        optimizer=optimizer,
        skip_completed=skip_completed,
    )
    if not ok:
        raise RuntimeError(f"Real-hardware run failed for seed={seed}, qubits={n_qubits}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 optimizer comparison on IBM Quantum hardware")
    parser.add_argument("--n_qubits", type=int, default=None, help="Number of qubits (single-point mode)")
    parser.add_argument("--outdir", type=str, default=None, help="Output folder (single-point mode)")
    parser.add_argument("--run-root", type=str, default=None, help="Run root for seed-session grid mode")
    parser.add_argument(
        "--qubit-sizes",
        type=str,
        default=None,
        help="Legacy multi-q mode: comma-separated sizes run sequentially (one container per q).",
    )
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
    parser.add_argument("--channel", type=str, default=os.environ.get("QISKIT_IBM_CHANNEL", "ibm_quantum_platform"))
    parser.add_argument("--optimization-level", type=int, default=None, help="Transpilation optimization level")
    parser.add_argument("--resilience-level", type=int, default=None, help="Runtime resilience level")
    parser.add_argument(
        "--runtime-result-timeout",
        type=float,
        default=DEFAULT_RUNTIME_RESULT_TIMEOUT,
        help="Optional client-side timeout (seconds) for each Estimator result(). "
        "Default: wait without timeout (IBM-recommended for hardware).",
    )
    parser.add_argument(
        "--execution-mode",
        type=str,
        default=DEFAULT_EXECUTION_MODE,
        choices=["session", "batch", "job"],
        help="IBM Runtime container: batch (default), session, or job.",
    )
    parser.add_argument(
        "--runtime-max-time",
        type=str,
        default=DEFAULT_RUNTIME_MAX_TIME,
        help="IBM Runtime container TTL for session/batch, e.g. 8h. Omit for IBM plan default.",
    )
    parser.add_argument(
        "--session-max-time",
        type=str,
        default=None,
        help="Deprecated alias for --runtime-max-time.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip if outdir/run_status.json already has status=completed.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="both",
        choices=["both", "cobyla", "qnspsa"],
        help="Which optimizer(s) to run. Default: both.",
    )
    args = parser.parse_args()

    runtime_max_time = args.runtime_max_time
    if args.session_max_time not in (None, "", "none", "None"):
        runtime_max_time = args.session_max_time
    runtime_max_time = normalize_runtime_max_time(runtime_max_time)

    common_kwargs = dict(
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
        runtime_result_timeout=args.runtime_result_timeout,
        runtime_max_time=runtime_max_time,
        execution_mode=args.execution_mode,
        architecture=args.architecture,
        resqnet_depth_split=parse_depth_split(args.resqnet_depth_split),
        resqnet_residual_mode=args.resqnet_residual_mode,
        optimizer=args.optimizer,
        skip_completed=args.skip_completed,
    )

    t0 = time.time()
    if args.qubit_sizes:
        if not args.run_root:
            raise SystemExit("Multi-q grid mode requires --run-root.")
        run_seed_session_grid(
            run_root=args.run_root,
            qubit_sizes=parse_qubit_sizes(args.qubit_sizes),
            **common_kwargs,
        )
    else:
        if args.n_qubits is None or args.outdir is None:
            raise SystemExit("Single-point mode requires --n_qubits and --outdir.")
        optimizer_compare(
            n_qubits=args.n_qubits,
            outdir=args.outdir,
            **common_kwargs,
        )
    print("Total runtime:", time.time() - t0)


if __name__ == "__main__":
    main()
