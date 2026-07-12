#!/usr/bin/env python3
"""Phase 3 gradient diagnostics: barren-plateau certification metrics.

Two sub-experiments, complementary to the final-cost metric:

1. Gradient variance at initialization vs number of qubits, per architecture.
   A barren plateau shows up as Var(dC/dtheta_i) decaying exponentially with n.

2. Gradient-norm decay along a short optimization trajectory, per architecture,
   swept over seeds and sizes. A vanishing gradient norm early in training
   indicates the optimizer is stuck on a plateau.

Both sub-experiments can run in two noise regimes:
  - ``ideal``: exact statevector estimator (parameter-shift is exact & cheap).
  - ``shot-noise``: Monte Carlo estimator that samples the readout with a finite
    number of shots, so the gradient sees the statistical noise floor that hides
    exponentially small signals on real hardware.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from qiskit.primitives import StatevectorEstimator  # noqa: E402
from qiskit.quantum_info import SparsePauliOp, Statevector  # noqa: E402

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hyperparams import (  # noqa: E402
    SIM_DEFAULT_BUDGET_K as DEFAULT_BUDGET_K,
    SHOT_NOISE_DEFAULT_TARGET_PRECISION as DEFAULT_TARGET_PRECISION,
    SIM_MAX_BUDGET_EVALS as MAX_BUDGET_EVALS,
)
from gradients import build_gradient, gradient_norm_trajectory, gradient_variance_at_init  # noqa: E402
from optimizers import optimizer_label, parse_optimizer_list  # noqa: E402
from helpers.architectures import (  # noqa: E402
    available_architectures,
    build_architecture,
    parse_depth_split,
)
from helpers.utils import init_trackers, resolve_budget_evals, run_optimizer_suite  # noqa: E402


class MonteCarloEstimator:
    """Shot-noise estimator for a single +-1 observable (e.g. <Z_readout>).

    Batched: evaluates every parameter vector in a pub, so it is compatible with
    the batched parameter-shift gradient in ``gradients.py`` (which passes ``2p``
    parameter sets in one pub) as well as the single-set objective in
    ``make_objective``.
    """

    def __init__(self, shots: int):
        self.shots = int(shots)

    def run(self, pubs):
        records = []
        for ansatz, observable, parameter_sets in pubs:
            values = []
            for parameters in parameter_sets:
                state = Statevector(ansatz.assign_parameters(parameters))
                exact = float(np.clip(np.real(state.expectation_value(observable)), -1.0, 1.0))
                plus_probability = float(np.clip(0.5 * (1.0 + exact), 0.0, 1.0))
                plus_count = np.random.binomial(self.shots, plus_probability)
                values.append((2 * plus_count - self.shots) / self.shots)
            records.append(SimpleNamespace(data=SimpleNamespace(evs=np.array(values))))
        return SimpleNamespace(result=lambda: records)


def make_estimator(noise_mode: str, target_precision: float):
    """Return (estimator, shots) for the requested regime."""
    if noise_mode == "ideal":
        return StatevectorEstimator(), None
    if noise_mode == "shot-noise":
        shots = int(1 / (target_precision**2))
        return MonteCarloEstimator(shots), shots
    raise ValueError(f"Unknown noise_mode '{noise_mode}'. Use 'ideal' or 'shot-noise'.")


def _circuit_for(arch, noise_mode: str):
    """Circuit to feed the estimator.

    The Monte Carlo estimator binds parameters via ``Statevector``, which needs
    a decomposed circuit for library blueprints (e.g. efficient_su2); the exact
    StatevectorEstimator handles the blueprint directly.
    """
    return arch.circuit.decompose() if noise_mode == "shot-noise" else arch.circuit


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _valid_sizes(architecture: str, sizes: list[int]) -> list[int]:
    if architecture in ("qcnn", "resqnet"):
        keep = [n for n in sizes if _is_power_of_two(n)]
    else:
        keep = list(sizes)
    dropped = sorted(set(sizes) - set(keep))
    if dropped:
        print(f"[{architecture}] skipping non-power-of-two sizes: {dropped}")
    return keep


def _build_observable(arch, n_qubits: int) -> SparsePauliOp:
    label = "I" * arch.readout_qubit + "Z" + "I" * (n_qubits - arch.readout_qubit - 1)
    return SparsePauliOp.from_list([(label, 1.0)])


def run_variance_experiment(
    architectures: list[str],
    sizes: list[int],
    *,
    n_samples: int,
    grad_index: int,
    init_dist: str,
    init_scale: float,
    resqnet_depth_split: tuple[int, int],
    seed: int,
    gradient,
    noise_mode: str,
    outdir: str,
) -> dict:
    results: dict[str, dict[str, list[float]]] = {}

    for architecture in architectures:
        arch_sizes = _valid_sizes(architecture, sizes)
        var_list: list[float] = []
        used_sizes: list[int] = []
        for n_qubits in arch_sizes:
            arch = build_architecture(
                architecture=architecture,
                n_qubits=n_qubits,
                resqnet_depth_split=resqnet_depth_split,
            )
            observable = _build_observable(arch, n_qubits)
            # Independent RNG per (arch, n) for reproducibility.
            rng = np.random.default_rng(seed + n_qubits)
            stats = gradient_variance_at_init(
                arch.circuit.decompose(),
                observable,
                gradient=gradient,
                n_samples=n_samples,
                index=grad_index,
                init_dist=init_dist,
                init_scale=init_scale,
                rng=rng,
            )
            print(
                f"[var:{noise_mode}] {architecture:9s} n={n_qubits:2d} "
                f"Var={stats['variance']:.3e} abs_mean={stats['abs_mean']:.3e}"
            )
            var_list.append(stats["variance"])
            used_sizes.append(n_qubits)
        results[architecture] = {"qubits": used_sizes, "variance": var_list}

    # Plot: Var vs qubits (log scale), one line per architecture.
    plt.figure(figsize=(8, 5))
    for architecture, data in results.items():
        if not data["qubits"]:
            continue
        plt.plot(
            data["qubits"],
            data["variance"],
            marker="o",
            label=architecture,
        )
    plt.yscale("log")
    plt.xlabel("Number of qubits")
    plt.ylabel(f"Var(dC/dtheta_{grad_index}) [{init_dist} init]")
    plt.title(f"Gradient variance at initialization ({noise_mode})")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    var_plot = os.path.join(outdir, "grad_var_vs_qubits.png")
    plt.savefig(var_plot, dpi=180)
    plt.close()
    print(f"Saved: {var_plot}")
    return results


def _norm_decay_single(
    architectures: list[str],
    *,
    n_qubits: int,
    seed: int,
    optimizers: list[str],
    budget_evals: int | None,
    budget_k: float,
    stride: int,
    max_points: int | None,
    resqnet_depth_split: tuple[int, int],
    cost_estimator,
    gradient,
    noise_mode: str,
    outdir: str,
) -> dict:
    """One gradient-norm-decay figure for a fixed (seed, n_qubits).

    Overlays every requested architecture x optimizer. Architectures that
    require a power-of-two size are skipped for incompatible ``n_qubits``. The
    training budget scales per architecture as ``k * num_params`` (same rule as
    the ideal/shot-noise training grids) unless a fixed ``budget_evals`` override
    is given.
    """
    results: dict[str, dict] = {}
    plt.figure(figsize=(8, 5))
    any_curve = False
    for architecture in architectures:
        if architecture in ("qcnn", "resqnet") and not _is_power_of_two(n_qubits):
            print(f"[norm:{noise_mode}] seed={seed} q={n_qubits}: skipping {architecture} (not power of two).")
            continue
        arch = build_architecture(
            architecture=architecture,
            n_qubits=n_qubits,
            resqnet_depth_split=resqnet_depth_split,
        )
        observable = _build_observable(arch, n_qubits)
        circuit = _circuit_for(arch, noise_mode)
        arch_budget, _mode = resolve_budget_evals(
            num_params=arch.num_parameters,
            budget_evals=budget_evals,
            budget_k=budget_k,
            min_budget_evals=1,
            max_budget_evals=MAX_BUDGET_EVALS,
        )
        # Near-identity (normal, small scale) init: the mitigated training regime.
        np.random.seed(seed)
        initial_parameters = np.random.normal(loc=0.0, scale=0.1, size=arch.num_parameters)

        history, counts = init_trackers(*optimizers)
        run_optimizer_suite(
            optimizers,
            estimator=cost_estimator,
            ansatz=circuit,
            observable=observable,
            budget_evals=arch_budget,
            initial_parameters=initial_parameters,
            history=history,
            counts=counts,
            track_params=True,
        )

        arch_result: dict[str, dict] = {}
        for name in optimizers:
            trajectory = history[name].get("params", [])
            norm_data = gradient_norm_trajectory(
                arch.circuit.decompose(),
                observable,
                trajectory,
                gradient=gradient,
                stride=stride,
                max_points=max_points,
            )
            iterations = norm_data["iterations"]
            norms = norm_data["grad_norm"]
            arch_result[name] = {
                "budget_evals": int(arch_budget),
                "iterations": iterations.tolist(),
                "grad_norm": norms.tolist(),
            }
            if len(iterations):
                any_curve = True
                plt.plot(
                    iterations,
                    norms,
                    marker=".",
                    label=f"{architecture}:{optimizer_label(name)}",
                )
                print(
                    f"[norm:{noise_mode}] seed={seed} q={n_qubits:2d} {architecture:9s} {name:8s} "
                    f"budget={arch_budget} start={norms[0]:.3e} end={norms[-1]:.3e} points={len(norms)}"
                )
        results[architecture] = arch_result

    plt.yscale("log")
    plt.xlabel("Optimizer iteration")
    plt.ylabel("||grad C||_2")
    plt.title(f"Gradient-norm decay ({noise_mode}, seed={seed}, n_qubits={n_qubits})")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    if any_curve:
        plt.legend(fontsize=8)
    plt.tight_layout()
    norm_plot = os.path.join(outdir, f"grad_norm_decay_seed{seed}_q{n_qubits}.png")
    plt.savefig(norm_plot, dpi=180)
    plt.close()
    print(f"Saved: {norm_plot}")
    return results


def _partial_path(outdir: str, noise_mode: str, tag: str) -> str:
    partial_dir = os.path.join(outdir, noise_mode, "partial")
    os.makedirs(partial_dir, exist_ok=True)
    return os.path.join(partial_dir, f"{tag}.json")


def write_partial(outdir: str, noise_mode: str, tag: str, payload: dict) -> None:
    path = _partial_path(outdir, noise_mode, tag)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
    print(f"Wrote partial: {path}")


def aggregate_grad_metrics(outdir: str) -> dict:
    """Merge per-task partial JSON files into a single grad_metrics payload."""
    config_path = os.path.join(outdir, "run_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing run config: {config_path}")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)

    payload: dict = {"config": config, "variance": {}, "norm_decay": {}}
    for noise_mode in config.get("noise_modes", []):
        partial_dir = os.path.join(outdir, noise_mode, "partial")
        if not os.path.isdir(partial_dir):
            continue
        for fname in sorted(os.listdir(partial_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(partial_dir, fname), encoding="utf-8") as handle:
                partial = json.load(handle)
            task = partial.get("task")
            data = partial.get("data", {})
            if task == "variance":
                payload["variance"][noise_mode] = data
            elif task == "norm_decay":
                seed = str(partial["seed"])
                n_qubits = str(partial["n_qubits"])
                payload["norm_decay"].setdefault(noise_mode, {}).setdefault(seed, {})[n_qubits] = data

    np.savez(os.path.join(outdir, "grad_metrics.npz"), payload=payload)
    with open(os.path.join(outdir, "grad_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
    print(f"Aggregated gradient metrics -> {outdir}/grad_metrics.json")
    return payload


def run_norm_decay_experiment(
    architectures: list[str],
    *,
    sizes: list[int],
    seeds: list[int],
    optimizers: list[str],
    budget_evals: int | None,
    budget_k: float,
    stride: int,
    max_points: int | None,
    resqnet_depth_split: tuple[int, int],
    cost_estimator,
    gradient,
    noise_mode: str,
    outdir: str,
) -> dict:
    """Gradient-norm decay swept over seeds x sizes.

    Produces one figure per (seed, n_qubits) pair -> ``len(seeds) * len(sizes)``
    plots, each overlaying the requested architectures and optimizers. Results
    are returned nested as ``results[seed][n_qubits][architecture][optimizer]``.
    """
    results: dict[str, dict] = {}
    for seed in seeds:
        results[str(seed)] = {}
        for n_qubits in sizes:
            results[str(seed)][str(n_qubits)] = _norm_decay_single(
                architectures,
                n_qubits=n_qubits,
                seed=seed,
                optimizers=optimizers,
                budget_evals=budget_evals,
                budget_k=budget_k,
                stride=stride,
                max_points=max_points,
                resqnet_depth_split=resqnet_depth_split,
                cost_estimator=cost_estimator,
                gradient=gradient,
                noise_mode=noise_mode,
                outdir=outdir,
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 gradient diagnostics (BP metrics)")
    parser.add_argument(
        "--architectures",
        type=str,
        default=",".join(available_architectures()),
        help="Comma-separated architectures to evaluate.",
    )
    parser.add_argument(
        "--qubit-sizes",
        type=str,
        default="4,8,12,16",
        help="Comma-separated qubit sizes for the variance experiment.",
    )
    parser.add_argument("--n-samples", type=int, default=300, help="Random samples for Var(grad).")
    parser.add_argument("--grad-index", type=int, default=0, help="Gradient component to sample.")
    parser.add_argument(
        "--init-dist",
        type=str,
        default="uniform",
        choices=["uniform", "normal"],
        help="Init distribution: uniform[-pi,pi] (Haar-like) or normal(0,scale).",
    )
    parser.add_argument("--init-scale", type=float, default=0.1, help="Std for normal init.")
    parser.add_argument(
        "--norm-sizes",
        type=str,
        default="4,8,12,16",
        help="Comma-separated qubit sizes swept in the norm-decay experiment (one plot per seed x size).",
    )
    parser.add_argument(
        "--norm-seeds",
        type=str,
        default="10,20,30,40,50",
        help="Comma-separated seeds swept in the norm-decay experiment (one plot per seed x size).",
    )
    parser.add_argument(
        "--norm-optimizers",
        type=str,
        default="cobyla",
        help="Optimizers whose trajectory is traced for the norm-decay experiment.",
    )
    parser.add_argument(
        "--norm-budget",
        type=int,
        default=0,
        help="Fixed training budget override. 0 (default) = dynamic budget k*p per architecture.",
    )
    parser.add_argument(
        "--norm-budget-k",
        type=float,
        default=DEFAULT_BUDGET_K,
        help="Dynamic budget multiplier k (budget = round(k * num_params)), same rule as the training grids.",
    )
    parser.add_argument("--norm-stride", type=int, default=10, help="Checkpoint stride for norms.")
    parser.add_argument(
        "--norm-max-points",
        type=int,
        default=0,
        help="Max norm checkpoints. 0 (default) = unlimited: only the budget/stride bound the count.",
    )
    parser.add_argument(
        "--noise-modes",
        type=str,
        default="ideal,shot-noise",
        help="Comma-separated noise regimes (full sweep). Ignored if --noise-mode is set.",
    )
    parser.add_argument(
        "--noise-mode",
        type=str,
        default=None,
        choices=["ideal", "shot-noise"],
        help="Run a single noise regime (used by Slurm array tasks).",
    )
    parser.add_argument(
        "--norm-seed",
        type=int,
        default=None,
        help="Run norm-decay for one seed only (Slurm array slice). Requires --norm-qubits.",
    )
    parser.add_argument(
        "--norm-qubits",
        type=int,
        default=None,
        help="Run norm-decay for one qubit count only (Slurm array slice). Requires --norm-seed.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Merge partial/*.json under --outdir into grad_metrics.json and exit.",
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=DEFAULT_TARGET_PRECISION,
        help="Target precision for shot-noise mode (shots ~= 1/precision^2; also the gradient precision).",
    )
    parser.add_argument(
        "--grad-method",
        type=str,
        default="auto",
        choices=["auto", "reverse", "lincomb"],
        help=(
            "Qiskit gradient primitive (exact for any gate structure). 'auto' = "
            "reverse for ideal, lincomb (+precision noise) for shot-noise."
        ),
    )
    parser.add_argument("--resqnet-depth-split", type=str, default="5,1")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--outdir", type=str, default=None, help="Output folder (defaults to timestamped).")
    parser.add_argument("--skip-variance", action="store_true")
    parser.add_argument("--skip-norm", action="store_true")
    args = parser.parse_args()

    if args.aggregate_only:
        if not args.outdir:
            parser.error("--aggregate-only requires --outdir")
        aggregate_grad_metrics(args.outdir)
        return

    if (args.norm_seed is None) ^ (args.norm_qubits is None):
        parser.error("--norm-seed and --norm-qubits must be given together")

    architectures = [a.strip() for a in args.architectures.split(",") if a.strip()]
    sizes = [int(s.strip()) for s in args.qubit_sizes.split(",") if s.strip()]
    norm_sizes = [int(s.strip()) for s in args.norm_sizes.split(",") if s.strip()]
    norm_seeds = [int(s.strip()) for s in args.norm_seeds.split(",") if s.strip()]
    norm_max_points = args.norm_max_points if args.norm_max_points and args.norm_max_points > 0 else None
    norm_budget_override = args.norm_budget if args.norm_budget and args.norm_budget > 0 else None
    if args.noise_mode:
        noise_modes = [args.noise_mode]
    else:
        noise_modes = [m.strip() for m in args.noise_modes.split(",") if m.strip()]
    depth_split = parse_depth_split(args.resqnet_depth_split)
    full_norm_sizes = [int(s.strip()) for s in args.norm_sizes.split(",") if s.strip()]
    full_norm_seeds = [int(s.strip()) for s in args.norm_seeds.split(",") if s.strip()]
    single_norm = args.norm_seed is not None
    if single_norm:
        norm_seeds = [args.norm_seed]
        norm_sizes = [args.norm_qubits]
    else:
        norm_seeds = full_norm_seeds
        norm_sizes = full_norm_sizes

    if args.outdir:
        outdir = args.outdir
    else:
        ts = time.strftime("%Y%m%d-%H%M%S")
        outdir = os.path.join("outputs", f"{ts}_phase3_grad-metrics")
    os.makedirs(outdir, exist_ok=True)

    config = {
        "architectures": architectures,
        "qubit_sizes": sizes,
        "n_samples": args.n_samples,
        "grad_index": args.grad_index,
        "init_dist": args.init_dist,
        "init_scale": args.init_scale,
        "norm_sizes": full_norm_sizes,
        "norm_seeds": full_norm_seeds,
        "norm_optimizers": args.norm_optimizers,
        "norm_budget_override": norm_budget_override,
        "norm_budget_k": args.norm_budget_k,
        "norm_stride": args.norm_stride,
        "norm_max_points": norm_max_points,
        "noise_modes": [m.strip() for m in args.noise_modes.split(",") if m.strip()],
        "target_precision": args.target_precision,
        "grad_method": args.grad_method,
        "seed": args.seed,
    }
    config_path = os.path.join(outdir, "run_config.json")
    if not os.path.isfile(config_path):
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)

    payload: dict = {
        "config": config,
        "variance": {},
        "norm_decay": {},
    }

    def _grad_for(noise_mode: str):
        """Qiskit gradient primitive for a mode (exact for any gate structure)."""
        method = args.grad_method
        if method == "auto":
            method = "lincomb" if noise_mode == "shot-noise" else "reverse"
        precision = args.target_precision if (method == "lincomb" and noise_mode == "shot-noise") else None
        return build_gradient(method, precision=precision), method

    t0 = time.time()
    for noise_mode in noise_modes:
        cost_estimator, shots = make_estimator(noise_mode, args.target_precision)
        gradient, grad_method = _grad_for(noise_mode)
        mode_dir = os.path.join(outdir, noise_mode)
        os.makedirs(mode_dir, exist_ok=True)
        print(f"\n=== noise_mode={noise_mode} (shots={shots}, grad={grad_method}) -> {mode_dir} ===")

        if not args.skip_variance:
            variance_data = run_variance_experiment(
                architectures,
                sizes,
                n_samples=args.n_samples,
                grad_index=args.grad_index,
                init_dist=args.init_dist,
                init_scale=args.init_scale,
                resqnet_depth_split=depth_split,
                seed=args.seed,
                gradient=gradient,
                noise_mode=noise_mode,
                outdir=mode_dir,
            )
            payload["variance"][noise_mode] = variance_data
            if args.noise_mode:
                write_partial(
                    outdir,
                    noise_mode,
                    "variance",
                    {"task": "variance", "noise_mode": noise_mode, "data": variance_data},
                )

        if not args.skip_norm:
            if single_norm:
                norm_data = _norm_decay_single(
                    architectures,
                    n_qubits=norm_sizes[0],
                    seed=norm_seeds[0],
                    optimizers=parse_optimizer_list(args.norm_optimizers),
                    budget_evals=norm_budget_override,
                    budget_k=args.norm_budget_k,
                    stride=args.norm_stride,
                    max_points=norm_max_points,
                    resqnet_depth_split=depth_split,
                    cost_estimator=cost_estimator,
                    gradient=gradient,
                    noise_mode=noise_mode,
                    outdir=mode_dir,
                )
                payload["norm_decay"].setdefault(noise_mode, {}).setdefault(str(norm_seeds[0]), {})[
                    str(norm_sizes[0])
                ] = norm_data
                write_partial(
                    outdir,
                    noise_mode,
                    f"norm_seed{norm_seeds[0]}_q{norm_sizes[0]}",
                    {
                        "task": "norm_decay",
                        "noise_mode": noise_mode,
                        "seed": norm_seeds[0],
                        "n_qubits": norm_sizes[0],
                        "data": norm_data,
                    },
                )
            else:
                payload["norm_decay"][noise_mode] = run_norm_decay_experiment(
                    architectures,
                    sizes=norm_sizes,
                    seeds=norm_seeds,
                    optimizers=parse_optimizer_list(args.norm_optimizers),
                    budget_evals=norm_budget_override,
                    budget_k=args.norm_budget_k,
                    stride=args.norm_stride,
                    max_points=norm_max_points,
                    resqnet_depth_split=depth_split,
                    cost_estimator=cost_estimator,
                    gradient=gradient,
                    noise_mode=noise_mode,
                    outdir=mode_dir,
                )

    if args.noise_mode:
        print("Partial task complete (array slice). Run --aggregate-only to merge.")
    else:
        np.savez(os.path.join(outdir, "grad_metrics.npz"), payload=payload)
        with open(os.path.join(outdir, "grad_metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
        print("Saved gradient metrics to", outdir)
    print("Total runtime:", time.time() - t0)


if __name__ == "__main__":
    main()
