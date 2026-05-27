#!/usr/bin/env python3
"""Shared helper utilities for optimizer experiments."""
from __future__ import annotations

from typing import Callable
from types import SimpleNamespace

import numpy as np
from qiskit.quantum_info import Statevector
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob
import os
import csv
import re
from dataclasses import dataclass
from typing import Dict, List

FLAT_RUN_DIR_RE = re.compile(r"seed_(?P<seed>-?\d+)_qubits_(?P<qubits>\d+)$")
SEED_DIR_RE = re.compile(r"seed_(?P<seed>-?\d+)$")
QUBITS_DIR_RE = re.compile(r"qubits_(?P<qubits>\d+)$")


@dataclass
class RunSummary:
    seed: int
    qubits: int
    cobyla_final_cost: float
    qnspsa_final_cost: float
    cobyla_evals: int
    qnspsa_evals: int


def resolve_budget_evals(
    *,
    num_params: int,
    budget_evals: int | None = None,
    budget_k: int | float | None = None,
    min_budget_evals: int = 1,
    max_budget_evals: int | None = None,
) -> tuple[int, str]:
    """Resolve optimization budget from explicit value or dynamic rule.

    Priority:
    1) If `budget_evals` is provided, use it directly.
    2) Otherwise, compute dynamic budget as round(budget_k * num_params).

    Returns:
        (resolved_budget, mode) where mode is "fixed" or "dynamic".
    """
    if num_params <= 0:
        raise ValueError(f"num_params must be positive, got {num_params}")
    if min_budget_evals <= 0:
        raise ValueError(f"min_budget_evals must be positive, got {min_budget_evals}")
    if max_budget_evals is not None and max_budget_evals < min_budget_evals:
        raise ValueError(
            f"max_budget_evals ({max_budget_evals}) cannot be lower than "
            f"min_budget_evals ({min_budget_evals})"
        )

    if budget_evals is not None:
        resolved = int(budget_evals)
        mode = "fixed"
    else:
        if budget_k is None:
            raise ValueError("budget_k is required when budget_evals is not provided")
        resolved = int(round(float(budget_k) * int(num_params)))
        mode = "dynamic"

    resolved = max(int(min_budget_evals), resolved)
    if max_budget_evals is not None:
        resolved = min(int(max_budget_evals), resolved)

    return resolved, mode


def find_optimizer_history_files(run_root: str) -> List[str]:
    """Return list of optimizer_history.npz files under run_root (recursive)."""
    flat_pattern = os.path.join(run_root, "**", "seed_*_qubits_*", "optimizer_history.npz")
    nested_pattern = os.path.join(run_root, "**", "seed_*", "qubits_*", "optimizer_history.npz")
    files = sorted(set(glob.glob(flat_pattern, recursive=True)) | set(glob.glob(nested_pattern, recursive=True)))
    return files


def parse_single_run(npz_path: str) -> RunSummary:
    run_dir = os.path.basename(os.path.dirname(npz_path))
    flat_match = FLAT_RUN_DIR_RE.match(run_dir)

    if flat_match:
        seed = int(flat_match.group("seed"))
        qubits = int(flat_match.group("qubits"))
    else:
        qubits_match = QUBITS_DIR_RE.match(run_dir)
        seed_dir = os.path.basename(os.path.dirname(os.path.dirname(npz_path)))
        seed_match = SEED_DIR_RE.match(seed_dir)

        if not (seed_match and qubits_match):
            raise ValueError(f"Unexpected run directory name: {os.path.dirname(npz_path)}")

        seed = int(seed_match.group("seed"))
        qubits = int(qubits_match.group("qubits"))

    payload = np.load(npz_path, allow_pickle=True)
    history = payload["history"].item()

    cobyla_cost = history["cobyla"]["cost"]
    qnspsa_cost = history["qnspsa"]["cost"]

    if not cobyla_cost or not qnspsa_cost:
        raise ValueError(f"Empty cost history found in {npz_path}")

    return RunSummary(
        seed=seed,
        qubits=qubits,
        cobyla_final_cost=float(cobyla_cost[-1]),
        qnspsa_final_cost=float(qnspsa_cost[-1]),
        cobyla_evals=len(cobyla_cost),
        qnspsa_evals=len(qnspsa_cost),
    )


def write_per_run_csv(rows: List[RunSummary], csv_path: str) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed",
                "qubits",
                "cobyla_final_cost",
                "qnspsa_final_cost",
                "cobyla_evals",
                "qnspsa_evals",
                "best_optimizer",
            ]
        )
        for row in sorted(rows, key=lambda r: (r.qubits, r.seed)):
            best = "cobyla" if row.cobyla_final_cost < row.qnspsa_final_cost else "qnspsa"
            writer.writerow(
                [
                    row.seed,
                    row.qubits,
                    row.cobyla_final_cost,
                    row.qnspsa_final_cost,
                    row.cobyla_evals,
                    row.qnspsa_evals,
                    best,
                ]
            )


def aggregate_rows(rows: List[RunSummary]) -> Dict[int, Dict[str, float]]:
    by_qubits: Dict[int, List[RunSummary]] = {}
    for row in rows:
        by_qubits.setdefault(row.qubits, []).append(row)

    agg: Dict[int, Dict[str, float]] = {}
    for q, grp in by_qubits.items():
        c_final = np.array([r.cobyla_final_cost for r in grp], dtype=float)
        q_final = np.array([r.qnspsa_final_cost for r in grp], dtype=float)
        c_eval = np.array([r.cobyla_evals for r in grp], dtype=float)
        q_eval = np.array([r.qnspsa_evals for r in grp], dtype=float)

        qnspsa_wins = int(np.sum(q_final < c_final))
        cobyla_wins = int(np.sum(c_final < q_final))

        agg[q] = {
            "n_runs": len(grp),
            "cobyla_final_mean": float(np.mean(c_final)),
            "cobyla_final_std": float(np.std(c_final)),
            "qnspsa_final_mean": float(np.mean(q_final)),
            "qnspsa_final_std": float(np.std(q_final)),
            "cobyla_evals_mean": float(np.mean(c_eval)),
            "qnspsa_evals_mean": float(np.mean(q_eval)),
            "qnspsa_wins": qnspsa_wins,
            "cobyla_wins": cobyla_wins,
        }

    return agg


def write_aggregated_csv(agg: Dict[int, Dict[str, float]], csv_path: str) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "qubits",
                "n_runs",
                "cobyla_final_mean",
                "cobyla_final_std",
                "qnspsa_final_mean",
                "qnspsa_final_std",
                "cobyla_evals_mean",
                "qnspsa_evals_mean",
                "cobyla_wins",
                "qnspsa_wins",
            ]
        )

        for q in sorted(agg):
            a = agg[q]
            writer.writerow(
                [
                    q,
                    int(a["n_runs"]),
                    a["cobyla_final_mean"],
                    a["cobyla_final_std"],
                    a["qnspsa_final_mean"],
                    a["qnspsa_final_std"],
                    a["cobyla_evals_mean"],
                    a["qnspsa_evals_mean"],
                    int(a.get("cobyla_wins", 0)),
                    int(a.get("qnspsa_wins", 0)),
                ]
            )


class ConvergenceReached(Exception):
    pass


class BudgetExceeded(Exception):
    pass


def init_trackers(*names):
    history = {name: {"evals": [], "cost": []} for name in names}
    counts = {name: 0 for name in names}
    return history, counts


def make_objective(
    name: str,
    *,
    counts: dict,
    history: dict,
    estimator: SimpleNamespace,
    ansatz,
    observable,
    budget_evals: int | None = None,
) -> Callable:
    def objective(parameters):
        next_count = counts[name] + 1
        if budget_evals is not None and next_count > budget_evals:
            raise BudgetExceeded()

        counts[name] = next_count
        pub = (ansatz, observable, [parameters])
        result = estimator.run([pub]).result()[0]
        cost = result.data.evs[0]
        history[name]["evals"].append(counts[name])
        history[name]["cost"].append(cost)
        return cost

    return objective


def with_early_stopping(objective: Callable, *, history: dict, key: str, window: int, tolerance: float, stop_exception: type) -> Callable:
    def wrapped(parameters):
        cost = objective(parameters)
        if len(history[key]["cost"]) >= window:
            last_window = history[key]["cost"][-window:]
            if max(last_window) - min(last_window) < tolerance:
                raise stop_exception()
        return cost

    return wrapped


def make_fidelity(*, counts: dict, key: str, circuit, budget_evals: int | None = None):
    def fidelity(params1, params2):
        next_count = counts[key] + 2
        if budget_evals is not None and next_count > budget_evals:
            raise BudgetExceeded()

        counts[key] = next_count
        sv1 = Statevector(circuit.assign_parameters(params1))
        sv2 = Statevector(circuit.assign_parameters(params2))
        return np.abs(sv1.inner(sv2)) ** 2

    return fidelity


def run_optimizer(name: str, optimizer, objective: Callable, x0, *, counts: dict | None = None) -> None:
    print(f"Running {name}...")
    try:
        optimizer.minimize(fun=objective, x0=x0)
    except BudgetExceeded:
        if counts is not None and name in counts:
            print(f"-> {name} stopped at budget after evaluation {counts[name]}.")
        else:
            print(f"-> {name} stopped at budget.")
    except ConvergenceReached:
        if counts is not None and name in counts:
            print(f"-> {name} stopped early at evaluation {counts[name]}.")
        else:
            print(f"-> {name} stopped early.")


def save_optimizer_time_series(history: dict, outdir: str, *, filename: str = "optimizer_compare.png") -> None:
    """Save time-series plots for optimizer progress.

    Expects `history` to be a dict with keys like 'cobyla' and 'qnspsa', each containing
    {'evals': [...], 'cost': [...]}.

    Outputs:
    - `<filename>`: cost vs optimizer iteration (more comparable across optimizers)
    - `<filename_stem>_budget<ext>`: cost vs global budget/eval count (legacy view)
    """
    os.makedirs(outdir, exist_ok=True)

    stem, ext = os.path.splitext(filename)
    if not ext:
        ext = ".png"
    iteration_filename = f"{stem}{ext}"
    budget_filename = f"{stem}_budget{ext}"

    def _plot(x_key: str, x_label: str, out_filename: str) -> None:
        plt.figure(figsize=(8, 5))
        if "cobyla" in history:
            if x_key == "iteration":
                x_vals = list(range(1, len(history["cobyla"]["cost"]) + 1))
            else:
                x_vals = history["cobyla"]["evals"]
            plt.plot(x_vals, history["cobyla"]["cost"], label="COBYLA", color="blue")
        if "qnspsa" in history:
            if x_key == "iteration":
                x_vals = list(range(1, len(history["qnspsa"]["cost"]) + 1))
            else:
                x_vals = history["qnspsa"]["evals"]
            plt.plot(x_vals, history["qnspsa"]["cost"], label="QNSPSA", color="green")
        plt.axhline(y=-1.0, color="r", linestyle="--", label="theoretical min -1")
        plt.xlabel(x_label)
        plt.ylabel("Cost")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, out_filename))
        plt.close()

    _plot("iteration", "Optimizer iteration", iteration_filename)
    _plot("budget", "Eval count", budget_filename)


def save_final_cost_boxplot(rows: list, agg: dict, outdir: str, *, filename: str = "final_cost_vs_qubits.png") -> None:
    """Save boxplots of final costs per optimizer grouped by qubit counts.

    `rows` is a list of RunSummary-like objects with attributes `qubits`,
    `cobyla_final_cost`, and `qnspsa_final_cost`.
    `agg` is the aggregated dictionary produced by `analyze_outputs.aggregate`.
    """
    qubits = sorted(agg)

    cobyla_samples = []
    qnspsa_samples = []
    for q in qubits:
        by_qubit = [row for row in rows if row.qubits == q]
        cobyla_samples.append([row.cobyla_final_cost for row in by_qubit])
        qnspsa_samples.append([row.qnspsa_final_cost for row in by_qubit])

    positions = np.arange(len(qubits), dtype=float)
    box_width = 0.32

    plt.figure(figsize=(9.5, 5.8))
    common_box_style = dict(
        patch_artist=True,
        widths=0.26,
        showmeans=True,
        meanline=True,
        whis=1.5,
    )

    cobyla_box = plt.boxplot(
        cobyla_samples,
        positions=positions - box_width / 2,
        **common_box_style,
    )
    qnspsa_box = plt.boxplot(
        qnspsa_samples,
        positions=positions + box_width / 2,
        **common_box_style,
    )

    for patch in cobyla_box["boxes"]:
        patch.set_facecolor("#4C78A8")
        patch.set_alpha(0.55)
    for patch in qnspsa_box["boxes"]:
        patch.set_facecolor("#54A24B")
        patch.set_alpha(0.55)

    for element in ["whiskers", "caps", "medians", "means"]:
        for artist in cobyla_box[element]:
            artist.set_color("#2F4B7C")
        for artist in qnspsa_box[element]:
            artist.set_color("#2E6E2F")

    plt.axhline(y=-1.0, color="r", linestyle="--", linewidth=1.2, label="Theoretical min")
    plt.xticks(positions, [str(q) for q in qubits])
    plt.xlabel("Qubits")
    plt.ylabel("Final cost")
    plt.title("Final Cost vs Qubits")
    # Keep a fixed cost scale across analyses for direct visual comparison.
    plt.ylim(-1.0, 0.0)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.legend([cobyla_box["boxes"][0], qnspsa_box["boxes"][0]], ["COBYLA", "QNSPSA"], loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename), dpi=180)
    plt.close()
