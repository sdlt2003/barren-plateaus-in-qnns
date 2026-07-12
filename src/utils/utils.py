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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from gradients import full_gradient
from optimizers import (
    get_optimizer_spec,
    optimizer_color,
    optimizer_label,
    ordered_optimizer_names,
)

FLAT_RUN_DIR_RE = re.compile(r"seed_(?P<seed>-?\d+)_qubits_(?P<qubits>\d+)$")
SEED_DIR_RE = re.compile(r"seed_(?P<seed>-?\d+)$")
QUBITS_DIR_RE = re.compile(r"qubits_(?P<qubits>\d+)$")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_runtime_outdir(outdir: str, *, project_root: Path | None = None) -> str:
    """Map outdir to a project-relative path safe inside bnd/container runs.

    ``bnd run`` executes Python with the repo mounted at ``/work``. Host absolute
    paths such as ``/home/.../barren-plateaus-qnns/outputs/run`` do not exist there
    and make ``os.makedirs`` fail after a successful hardware run.
    """
    outdir_path = Path(outdir)
    root = project_root or PROJECT_ROOT

    if not outdir_path.is_absolute():
        return str(outdir_path)

    parts = outdir_path.parts

    # Prefer anchoring at outputs/, the standard artifact root for this repo.
    if "outputs" in parts:
        idx = parts.index("outputs")
        return str(Path(*parts[idx:]))

    repo_name = root.name
    if repo_name in parts:
        suffix = Path(*parts[parts.index(repo_name) + 1 :])
        if str(suffix):
            return str(suffix)

    try:
        return str(outdir_path.resolve().relative_to(root.resolve()))
    except (ValueError, OSError):
        pass

    raise ValueError(
        f"Cannot map outdir {outdir!r} to a project-relative path. "
        "Pass a path relative to the repo root, e.g. outputs/my_run/seed_10/qubits_4."
    )


@dataclass
class RunSummary:
    """Per-(seed, qubits) summary generalized to N optimizers.

    ``final_cost`` and ``evals`` map each optimizer name present in the run to
    its final cost / number of evaluations. Backward compatible convenience
    properties (``cobyla_final_cost`` etc.) are provided for older callers.
    """

    seed: int
    qubits: int
    final_cost: Dict[str, float | None] = field(default_factory=dict)
    evals: Dict[str, int | None] = field(default_factory=dict)

    @property
    def optimizers(self) -> list[str]:
        return ordered_optimizer_names(self.final_cost.keys())

    @property
    def cobyla_final_cost(self) -> float | None:
        return self.final_cost.get("cobyla")

    @property
    def qnspsa_final_cost(self) -> float | None:
        return self.final_cost.get("qnspsa")

    @property
    def cobyla_evals(self) -> int | None:
        return self.evals.get("cobyla")

    @property
    def qnspsa_evals(self) -> int | None:
        return self.evals.get("qnspsa")


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
    nested_opt_pattern = os.path.join(run_root, "**", "seed_*", "qubits_*", "opt_*", "optimizer_history.npz")
    files = sorted(
        set(glob.glob(flat_pattern, recursive=True))
        | set(glob.glob(nested_pattern, recursive=True))
        | set(glob.glob(nested_opt_pattern, recursive=True))
    )
    return files


def parse_single_run(npz_path: str) -> RunSummary:
    # Support both layouts:
    # - flat:   outputs/seed_<SEED>_qubits_<Q>/optimizer_history.npz
    # - nested: outputs/<run>/seed_<SEED>/qubits_<Q>/optimizer_history.npz
    # - nested split: outputs/<run>/seed_<SEED>/qubits_<Q>/opt_<OPT>/optimizer_history.npz
    parent_dir = os.path.basename(os.path.dirname(npz_path))
    flat_match = FLAT_RUN_DIR_RE.match(parent_dir)
    if flat_match:
        seed = int(flat_match.group("seed"))
        qubits = int(flat_match.group("qubits"))
    else:
        seed = None
        qubits = None
        for ancestor in Path(npz_path).parents:
            name = ancestor.name
            if qubits is None:
                qm = QUBITS_DIR_RE.match(name)
                if qm:
                    qubits = int(qm.group("qubits"))
            if seed is None:
                sm = SEED_DIR_RE.match(name)
                if sm:
                    seed = int(sm.group("seed"))
            if seed is not None and qubits is not None:
                break
        if seed is None or qubits is None:
            raise ValueError(f"Unexpected optimizer_history location: {npz_path}")

    payload = np.load(npz_path, allow_pickle=True)
    history = payload["history"].item()

    def _final_cost_and_evals(cost_series) -> tuple[float | None, int | None]:
        try:
            n = len(cost_series)
        except Exception:
            n = 0
        if n <= 0:
            return None, None
        try:
            return float(cost_series[-1]), int(n)
        except Exception:
            return None, int(n)

    final_cost: Dict[str, float | None] = {}
    evals: Dict[str, int | None] = {}
    for name in ordered_optimizer_names(history.keys()):
        cost_series = history.get(name, {}).get("cost", [])
        final, n_evals = _final_cost_and_evals(cost_series)
        final_cost[name] = final
        evals[name] = n_evals

    if not final_cost or all(v is None for v in final_cost.values()):
        raise ValueError(f"Empty cost history found in {npz_path}")

    return RunSummary(
        seed=seed,
        qubits=qubits,
        final_cost=final_cost,
        evals=evals,
    )


def collect_optimizer_names(rows: List[RunSummary]) -> list[str]:
    """Union of optimizer names across rows, in canonical order."""
    names: set[str] = set()
    for row in rows:
        names.update(row.final_cost.keys())
    return ordered_optimizer_names(names)


def _best_optimizer(final_cost: Dict[str, float | None]) -> str:
    """Optimizer with the strictly lowest final cost, or 'n/a'/'tie'."""
    valid = {name: cost for name, cost in final_cost.items() if cost is not None}
    if not valid:
        return "n/a"
    best_cost = min(valid.values())
    winners = [name for name, cost in valid.items() if cost == best_cost]
    ordered = ordered_optimizer_names(winners)
    if len(ordered) > 1:
        return "tie:" + "/".join(ordered)
    return ordered[0]


def write_per_run_csv(rows: List[RunSummary], csv_path: str) -> None:
    names = collect_optimizer_names(rows)
    header = (
        ["seed", "qubits"]
        + [f"{name}_final_cost" for name in names]
        + [f"{name}_evals" for name in names]
        + ["best_optimizer"]
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in sorted(rows, key=lambda r: (r.qubits, r.seed)):
            record = [row.seed, row.qubits]
            for name in names:
                cost = row.final_cost.get(name)
                record.append("" if cost is None else cost)
            for name in names:
                n_evals = row.evals.get(name)
                record.append("" if n_evals is None else n_evals)
            record.append(_best_optimizer(row.final_cost))
            writer.writerow(record)


def aggregate_rows(rows: List[RunSummary]) -> Dict[int, Dict[str, float]]:
    by_qubits: Dict[int, List[RunSummary]] = {}
    for row in rows:
        by_qubits.setdefault(row.qubits, []).append(row)

    names = collect_optimizer_names(rows)

    agg: Dict[int, Dict[str, float]] = {}
    for q, grp in by_qubits.items():
        entry: Dict[str, float] = {"n_runs": len(grp)}
        final_arrays: Dict[str, np.ndarray] = {}
        for name in names:
            final_arr = np.array(
                [np.nan if r.final_cost.get(name) is None else float(r.final_cost[name]) for r in grp],
                dtype=float,
            )
            eval_arr = np.array(
                [np.nan if r.evals.get(name) is None else float(r.evals[name]) for r in grp],
                dtype=float,
            )
            final_arrays[name] = final_arr
            entry[f"{name}_final_mean"] = float(np.nanmean(final_arr)) if np.any(np.isfinite(final_arr)) else float("nan")
            entry[f"{name}_final_std"] = float(np.nanstd(final_arr)) if np.any(np.isfinite(final_arr)) else float("nan")
            entry[f"{name}_evals_mean"] = float(np.nanmean(eval_arr)) if np.any(np.isfinite(eval_arr)) else float("nan")

        # Per-run winner among available optimizers (strict minimum only).
        wins = {name: 0 for name in names}
        for idx in range(len(grp)):
            costs = {name: final_arrays[name][idx] for name in names if np.isfinite(final_arrays[name][idx])}
            if not costs:
                continue
            best_cost = min(costs.values())
            winners = [name for name, cost in costs.items() if cost == best_cost]
            if len(winners) == 1:
                wins[winners[0]] += 1
        for name in names:
            entry[f"{name}_wins"] = wins[name]

        agg[q] = entry

    return agg


def write_aggregated_csv(agg: Dict[int, Dict[str, float]], csv_path: str) -> None:
    # Derive optimizer names from the aggregated entries (columns *_final_mean).
    names: list[str] = []
    for entry in agg.values():
        for key in entry:
            if key.endswith("_final_mean"):
                candidate = key[: -len("_final_mean")]
                if candidate not in names:
                    names.append(candidate)
    names = ordered_optimizer_names(names)

    header = ["qubits", "n_runs"]
    for name in names:
        header += [f"{name}_final_mean", f"{name}_final_std"]
    for name in names:
        header += [f"{name}_evals_mean"]
    for name in names:
        header += [f"{name}_wins"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for q in sorted(agg):
            a = agg[q]
            record: list = [q, int(a["n_runs"])]
            for name in names:
                record.append(a.get(f"{name}_final_mean", float("nan")))
                record.append(a.get(f"{name}_final_std", float("nan")))
            for name in names:
                record.append(a.get(f"{name}_evals_mean", float("nan")))
            for name in names:
                record.append(int(a.get(f"{name}_wins", 0)))
            writer.writerow(record)


class BudgetExceeded(Exception):
    pass


def init_trackers(*names):
    history = {name: {"evals": [], "cost": []} for name in names}
    counts = {name: 0 for name in names}
    return history, counts


def _is_grad_checkpoint(eval_index: int, *, grad_stride: int) -> bool:
    stride = max(1, int(grad_stride))
    return eval_index == 1 or (eval_index - 1) % stride == 0


def _record_gradient_checkpoint(
    *,
    history: dict,
    name: str,
    eval_index: int,
    parameters,
    ansatz,
    observable,
    gradient,
    track_grad_norm: bool,
    track_grad_var: bool,
) -> None:
    """Append gradient diagnostics at this optimizer eval (does not use budget)."""
    from gradients import gradient_vector_stats

    grad = full_gradient(gradient, ansatz, observable, parameters)
    stats = gradient_vector_stats(grad)
    history[name].setdefault("grad_checkpoint_evals", []).append(int(eval_index))
    if track_grad_norm:
        history[name].setdefault("grad_norm", []).append(stats["grad_norm"])
    if track_grad_var:
        history[name].setdefault("grad_var_components", []).append(stats["grad_var_components"])


def make_objective(
    name: str,
    *,
    counts: dict,
    history: dict,
    estimator: SimpleNamespace,
    ansatz,
    observable,
    budget_evals: int | None = None,
    result_timeout_s: float | None = None,
    track_params: bool = False,
    gradient=None,
    track_grad_norm: bool = False,
    track_grad_var: bool = False,
    grad_stride: int = 10,
) -> Callable:
    track_grad = (track_grad_norm or track_grad_var) and gradient is not None

    def objective(parameters):
        next_count = counts[name] + 1
        if budget_evals is not None and next_count > budget_evals:
            raise BudgetExceeded()

        counts[name] = next_count
        pub = (ansatz, observable, [parameters])
        job = estimator.run([pub])
        if result_timeout_s is None:
            result = job.result()[0]
        else:
            try:
                result = job.result(timeout=result_timeout_s)[0]
            except TypeError:
                # Some local estimators may not support timeout keyword.
                result = job.result()[0]
        cost = result.data.evs[0]
        history[name]["evals"].append(counts[name])
        history[name]["cost"].append(cost)
        if track_params:
            history[name].setdefault("params", []).append(np.array(parameters, dtype=float).copy())
        if track_grad and _is_grad_checkpoint(counts[name], grad_stride=grad_stride):
            _record_gradient_checkpoint(
                history=history,
                name=name,
                eval_index=counts[name],
                parameters=parameters,
                ansatz=ansatz,
                observable=observable,
                gradient=gradient,
                track_grad_norm=track_grad_norm,
                track_grad_var=track_grad_var,
            )
        return cost

    return objective


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


def run_optimizer_suite(
    optimizer_names,
    *,
    estimator,
    ansatz,
    observable,
    budget_evals: int,
    initial_parameters,
    history: dict,
    counts: dict,
    fidelity_circuit=None,
    result_timeout_s: float | None = None,
    track_params: bool = False,
    gradient=None,
    track_grad_norm: bool = False,
    track_grad_var: bool = False,
    grad_stride: int = 10,
) -> None:
    """Run a list of registry optimizers sharing the same problem/budget.

    Removes the duplicated COBYLA/QNSPSA wiring from the experiment scripts.
    Fidelity callbacks are only built for optimizers that declare
    ``needs_fidelity`` (currently QNSPSA), using ``fidelity_circuit`` if given
    (e.g. the logical, pre-transpilation ansatz) or ``ansatz`` otherwise.
    """
    for name in optimizer_names:
        spec = get_optimizer_spec(name)
        objective = make_objective(
            name,
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
        fidelity = None
        if spec.needs_fidelity:
            fid_circuit = fidelity_circuit if fidelity_circuit is not None else ansatz
            fidelity = make_fidelity(
                counts=counts,
                key=name,
                circuit=fid_circuit,
                budget_evals=budget_evals,
            )
        optimizer = spec.build(budget_evals, fidelity)
        run_optimizer(name, optimizer, objective, initial_parameters, counts=counts)


def save_optimizer_time_series(history: dict, outdir: str, *, filename: str = "optimizer_compare.png") -> None:
    """Save time-series plots for optimizer progress.

    Expects `history` to be a dict with keys like 'cobyla' and 'qnspsa', each containing
    {'evals': [...], 'cost': [...]}.

    Outputs:
    - `<filename>`: cost vs optimizer iteration (more comparable across optimizers)
    - `<filename_stem>_budget<ext>`: cost vs global budget/eval count (legacy view)
    - `<filename_stem>_best<ext>`: best-so-far cost vs iteration. Sequential
      minimizers like NFT log their +-pi/2 probe evaluations, so the raw cost
      curve is a sawtooth that misrepresents convergence; the running minimum
      (np.minimum.accumulate) shows the actual monotone progress instead.
    """
    os.makedirs(outdir, exist_ok=True)

    stem, ext = os.path.splitext(filename)
    if not ext:
        ext = ".png"
    iteration_filename = f"{stem}{ext}"
    budget_filename = f"{stem}_budget{ext}"
    best_filename = f"{stem}_best{ext}"

    names = ordered_optimizer_names(
        [name for name, series in history.items() if series.get("cost")]
    )

    def _plot(x_key: str, x_label: str, out_filename: str) -> None:
        plt.figure(figsize=(8, 5))
        for name in names:
            cost = history[name].get("cost", [])
            if not cost:
                continue
            if x_key == "iteration":
                x_vals = list(range(1, len(cost) + 1))
            else:
                x_vals = history[name].get("evals") or list(range(1, len(cost) + 1))
            plt.plot(x_vals, cost, label=optimizer_label(name), color=optimizer_color(name))
        plt.axhline(y=-1.0, color="r", linestyle="--", label="theoretical min -1")
        plt.xlabel(x_label)
        plt.ylabel("Cost")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, out_filename))
        plt.close()

    def _plot_best_so_far(out_filename: str) -> None:
        plt.figure(figsize=(8, 5))
        for name in names:
            cost = history[name].get("cost", [])
            if not cost:
                continue
            best = np.minimum.accumulate(np.asarray(cost, dtype=float))
            x_vals = list(range(1, len(best) + 1))
            plt.plot(x_vals, best, label=optimizer_label(name), color=optimizer_color(name))
        plt.axhline(y=-1.0, color="r", linestyle="--", label="theoretical min -1")
        plt.xlabel("Optimizer iteration")
        plt.ylabel("Best cost so far")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, out_filename))
        plt.close()

    _plot("iteration", "Optimizer iteration", iteration_filename)
    _plot("budget", "Eval count", budget_filename)
    _plot_best_so_far(best_filename)

    grad_names = [
        name
        for name in names
        if history[name].get("grad_norm") and history[name].get("grad_checkpoint_evals")
    ]
    if grad_names:
        plt.figure(figsize=(8, 5))
        for name in grad_names:
            series = history[name]
            plt.plot(
                series["grad_checkpoint_evals"],
                series["grad_norm"],
                "o-",
                label=optimizer_label(name),
                color=optimizer_color(name),
                markersize=4,
            )
        plt.xlabel("Eval count (gradient checkpoint)")
        plt.ylabel("||grad C||_2")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{stem}_grad_norm{ext}"))
        plt.close()

    grad_var_names = [
        name
        for name in names
        if history[name].get("grad_var_components") and history[name].get("grad_checkpoint_evals")
    ]
    if grad_var_names:
        plt.figure(figsize=(8, 5))
        for name in grad_var_names:
            series = history[name]
            plt.plot(
                series["grad_checkpoint_evals"],
                series["grad_var_components"],
                "o-",
                label=optimizer_label(name),
                color=optimizer_color(name),
                markersize=4,
            )
        plt.xlabel("Eval count (gradient checkpoint)")
        plt.ylabel("Var_i(dC/dtheta_i)")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{stem}_grad_var{ext}"))
        plt.close()


def save_final_cost_boxplot(rows: list, agg: dict, outdir: str) -> list[str]:
    """Save boxplots of final costs per optimizer grouped by qubit counts.

    Writes two fixed-scale views for direct cross-run comparison:
    - ``final_cost_vs_qubits.png``: y-axis from 0 to -1 (convergence zoom)
    - ``final_cost_vs_qubits_full.png``: y-axis from 1 to -1 (full range)

    `rows` is a list of RunSummary objects (final costs keyed by optimizer).
    `agg` is the aggregated dictionary produced by `aggregate_rows`.

    Returns the list of saved plot paths.
    """
    qubits = sorted(agg)
    names = collect_optimizer_names(rows)
    if not names:
        return []

    # Per-optimizer samples grouped by qubit count.
    samples_by_name: dict[str, list[list[float]]] = {}
    for name in names:
        per_qubit: list[list[float]] = []
        for q in qubits:
            by_qubit = [row for row in rows if row.qubits == q]
            per_qubit.append(
                [row.final_cost.get(name) for row in by_qubit if row.final_cost.get(name) is not None]
            )
        samples_by_name[name] = per_qubit

    positions = np.arange(len(qubits), dtype=float)
    n_opt = len(names)
    group_span = 0.72
    slot = group_span / n_opt
    box_width = slot * 0.8
    common_box_style = dict(
        patch_artist=True,
        widths=box_width,
        showmeans=True,
        meanline=True,
        whis=1.5,
    )

    plot_specs = (
        ("final_cost_vs_qubits.png", -1.0, 0.0, "Final Cost vs Qubits (0 to -1)"),
        ("final_cost_vs_qubits_full.png", -1.0, 1.0, "Final Cost vs Qubits (1 to -1)"),
    )
    saved_paths: list[str] = []

    for filename, y_min, y_max, title in plot_specs:
        plt.figure(figsize=(9.5, 5.8))

        legend_handles = []
        legend_labels = []
        for i, name in enumerate(names):
            offset = (i - (n_opt - 1) / 2.0) * slot
            box = plt.boxplot(
                samples_by_name[name],
                positions=positions + offset,
                **common_box_style,
            )
            color = optimizer_color(name)
            for patch in box["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.55)
                patch.set_edgecolor(color)
            for element in ["whiskers", "caps", "medians", "means"]:
                for artist in box[element]:
                    artist.set_color(color)
            if box["boxes"]:
                legend_handles.append(box["boxes"][0])
                legend_labels.append(optimizer_label(name))

        plt.axhline(y=-1.0, color="r", linestyle="--", linewidth=1.2, label="Theoretical min")
        plt.xticks(positions, [str(q) for q in qubits])
        plt.xlabel("Qubits")
        plt.ylabel("Final cost")
        plt.title(title)
        plt.ylim(y_min, y_max)
        plt.grid(True, axis="y", linestyle="--", alpha=0.4)
        if legend_handles:
            plt.legend(legend_handles, legend_labels, loc="best")
        plt.tight_layout()
        out_path = os.path.join(outdir, filename)
        plt.savefig(out_path, dpi=180)
        plt.close()
        saved_paths.append(out_path)

    return saved_paths
