#!/usr/bin/env python3
"""Aggregate optimizer comparison outputs from run folders.

Expected run folder name:
  seed_<SEED>_qubits_<Q>

Expected payload file inside each run folder:
  optimizer_history.npz

Outputs:
  - summary_per_run.csv
  - summary_aggregated.csv
  - final_cost_vs_qubits.png
  - evals_vs_qubits.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_DIR_RE = re.compile(r"seed_(?P<seed>-?\d+)_qubits_(?P<qubits>\d+)$")


@dataclass
class RunSummary:
    seed: int
    qubits: int
    cobyla_final_cost: float
    qnspsa_final_cost: float
    cobyla_evals: int
    qnspsa_evals: int


def _default_input_dir() -> str:
    candidates = [
        "cluster_outputs/outputs",
        "outputs",
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return "cluster_outputs/outputs"


def _default_outdir(input_dir: str) -> str:
    return os.path.join(input_dir, "analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate QNSPSA-vs-COBYLA results")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=_default_input_dir(),
        help="Directory containing seed_<seed>_qubits_<q>/optimizer_history.npz",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory where summary files and plots will be written",
    )
    args = parser.parse_args()
    if args.outdir is None:
        args.outdir = _default_outdir(args.input_dir)
    return args


def _to_1d_float_array(values: object, name: str, npz_path: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"Empty {name} history found in {npz_path}")
    return arr


def load_single_run(npz_path: str) -> RunSummary:
    run_dir = os.path.basename(os.path.dirname(npz_path))
    match = RUN_DIR_RE.match(run_dir)
    if not match:
        raise ValueError(f"Unexpected run directory name: {run_dir}")

    seed = int(match.group("seed"))
    qubits = int(match.group("qubits"))

    payload = np.load(npz_path, allow_pickle=True)
    if "history" not in payload:
        raise ValueError(f"Missing 'history' key in {npz_path}")

    history = payload["history"].item()
    if "cobyla" not in history or "qnspsa" not in history:
        raise ValueError(f"Missing optimizer keys in history for {npz_path}")

    cobyla_cost = _to_1d_float_array(history["cobyla"]["cost"], "cobyla/cost", npz_path)
    qnspsa_cost = _to_1d_float_array(history["qnspsa"]["cost"], "qnspsa/cost", npz_path)

    return RunSummary(
        seed=seed,
        qubits=qubits,
        cobyla_final_cost=float(cobyla_cost[-1]),
        qnspsa_final_cost=float(qnspsa_cost[-1]),
        cobyla_evals=int(cobyla_cost.size),
        qnspsa_evals=int(qnspsa_cost.size),
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


def aggregate(rows: List[RunSummary]) -> Dict[int, Dict[str, float]]:
    by_qubits: Dict[int, List[RunSummary]] = {}
    for row in rows:
        by_qubits.setdefault(row.qubits, []).append(row)

    agg: Dict[int, Dict[str, float]] = {}
    for qubits, grp in by_qubits.items():
        c_final = np.array([r.cobyla_final_cost for r in grp], dtype=float)
        q_final = np.array([r.qnspsa_final_cost for r in grp], dtype=float)
        c_eval = np.array([r.cobyla_evals for r in grp], dtype=float)
        q_eval = np.array([r.qnspsa_evals for r in grp], dtype=float)

        qnspsa_wins = int(np.sum(q_final < c_final))
        cobyla_wins = int(np.sum(c_final < q_final))

        agg[qubits] = {
            "n_runs": float(len(grp)),
            "cobyla_final_mean": float(np.mean(c_final)),
            "cobyla_final_std": float(np.std(c_final)),
            "qnspsa_final_mean": float(np.mean(q_final)),
            "qnspsa_final_std": float(np.std(q_final)),
            "cobyla_evals_mean": float(np.mean(c_eval)),
            "qnspsa_evals_mean": float(np.mean(q_eval)),
            "qnspsa_wins": float(qnspsa_wins),
            "cobyla_wins": float(cobyla_wins),
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
                "qnspsa_wins",
                "cobyla_wins",
            ]
        )

        for qubits in sorted(agg):
            a = agg[qubits]
            writer.writerow(
                [
                    qubits,
                    int(a["n_runs"]),
                    a["cobyla_final_mean"],
                    a["cobyla_final_std"],
                    a["qnspsa_final_mean"],
                    a["qnspsa_final_std"],
                    a["cobyla_evals_mean"],
                    a["qnspsa_evals_mean"],
                    int(a["qnspsa_wins"]),
                    int(a["cobyla_wins"]),
                ]
            )


def make_plots(agg: Dict[int, Dict[str, float]], outdir: str) -> None:
    qubits = sorted(agg)
    if not qubits:
        return

    c_mean = np.array([agg[q]["cobyla_final_mean"] for q in qubits], dtype=float)
    c_std = np.array([agg[q]["cobyla_final_std"] for q in qubits], dtype=float)
    q_mean = np.array([agg[q]["qnspsa_final_mean"] for q in qubits], dtype=float)
    q_std = np.array([agg[q]["qnspsa_final_std"] for q in qubits], dtype=float)

    c_eval = np.array([agg[q]["cobyla_evals_mean"] for q in qubits], dtype=float)
    q_eval = np.array([agg[q]["qnspsa_evals_mean"] for q in qubits], dtype=float)

    plt.figure(figsize=(9, 5.5))
    plt.errorbar(qubits, c_mean, yerr=c_std, marker="o", capsize=4, label="COBYLA")
    plt.errorbar(qubits, q_mean, yerr=q_std, marker="s", capsize=4, label="QNSPSA")
    plt.axhline(y=-1.0, color="r", linestyle="--", linewidth=1.2, label="Theoretical min")
    plt.xlabel("Qubits")
    plt.ylabel("Final cost (mean +- std across seeds)")
    plt.title("Final Cost vs Qubits")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "final_cost_vs_qubits.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5.5))
    plt.plot(qubits, c_eval, marker="o", label="COBYLA evals")
    plt.plot(qubits, q_eval, marker="s", label="QNSPSA evals")
    plt.xlabel("Qubits")
    plt.ylabel("Mean function evaluations")
    plt.title("Mean Evaluations vs Qubits")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "evals_vs_qubits.png"), dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    pattern = os.path.join(args.input_dir, "seed_*_qubits_*", "optimizer_history.npz")
    files = sorted(glob.glob(pattern))

    if not files:
        raise SystemExit(f"No optimizer_history.npz files found under: {args.input_dir}")

    rows: List[RunSummary] = []
    for file_path in files:
        try:
            rows.append(load_single_run(file_path))
        except Exception as exc:
            print(f"Skipping {file_path}: {exc}")

    if not rows:
        raise SystemExit("No valid runs found after parsing input files.")

    per_run_csv = os.path.join(args.outdir, "summary_per_run.csv")
    agg_csv = os.path.join(args.outdir, "summary_aggregated.csv")

    write_per_run_csv(rows, per_run_csv)
    agg = aggregate(rows)
    write_aggregated_csv(agg, agg_csv)
    make_plots(agg, args.outdir)

    print(f"Parsed runs: {len(rows)}")
    print(f"Saved: {per_run_csv}")
    print(f"Saved: {agg_csv}")
    print(f"Saved: {os.path.join(args.outdir, 'final_cost_vs_qubits.png')}")
    print(f"Saved: {os.path.join(args.outdir, 'evals_vs_qubits.png')}")


if __name__ == "__main__":
    main()
