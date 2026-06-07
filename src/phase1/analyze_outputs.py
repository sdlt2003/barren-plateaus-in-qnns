#!/usr/bin/env python3
"""Aggregate optimizer comparison outputs from Slurm runs.

Expected input structure:
    outputs/<run_root>/seed_<SEED>/qubits_<Q>/optimizer_history.npz

This script builds inside the selected run folder:
  - summary_per_run.csv
  - summary_aggregated.csv
  - final_cost_vs_qubits.png
  - final_cost_vs_qubits_full.png

It supports both the flat layout:
        outputs/seed_<SEED>_qubits_<Q>/optimizer_history.npz

and the timestamped layout produced by the sbatch launchers:
        outputs/<timestamp>_<launcher_name>/seed_<SEED>/qubits_<Q>/optimizer_history.npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
from typing import List

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import (
    find_optimizer_history_files,
    parse_single_run,
    write_per_run_csv,
    aggregate_rows,
    write_aggregated_csv,
    save_final_cost_boxplot,
    RunSummary,
)

TOTAL_RUNTIME_RE = re.compile(r"Total runtime:\s*([0-9]+(?:\.[0-9]+)?)")
SEED_RE = re.compile(r"seed=(\d+)")
QUBITS_RE = re.compile(r"(?:qubits|n_qubits)=(\d+)")
ARCH_RE = re.compile(r"arch=([a-zA-Z0-9_\\-]+)")
EXCEPTION_TYPE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*Error):")
RUN_ROOT_RE = re.compile(r"Run root:\s*(\S+)")


def _classify_error(err_text: str) -> tuple[str, str | None]:
    text = err_text.strip()
    if not text:
        return "none", None
    if "DUE TO TIME LIMIT" in text:
        return "time_limit", "time_limit"
    exc_match = EXCEPTION_TYPE_RE.search(text)
    if exc_match:
        return "python_exception", exc_match.group(1)
    if "Traceback (most recent call last):" in text:
        return "python_exception", "Traceback"
    return "stderr_nonempty", None


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def _extract_run_root(out_text: str) -> str | None:
    match = RUN_ROOT_RE.search(out_text)
    if not match:
        return None
    raw = match.group(1).strip()
    if raw.startswith("outputs/"):
        return raw
    return f"outputs/{raw}"


def write_run_diagnostics_log(run_root: str, outdir: str) -> None:
    logs_dir = Path("outputs/logs")
    log_path = Path(outdir) / "run_diagnostics.md"

    if not logs_dir.is_dir():
        log_path.write_text("# Run diagnostics\n\nNo `outputs/logs` directory found.\n", encoding="utf-8")
        return

    run_root_norm = run_root.replace("\\", "/").rstrip("/")
    matched_entries: list[dict] = []

    for out_file in sorted(logs_dir.glob("*.out")):
        try:
            out_text = out_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        out_run_root = _extract_run_root(out_text)
        if out_run_root is None:
            continue
        if out_run_root.replace("\\", "/").rstrip("/") != run_root_norm:
            continue

        err_file = out_file.with_suffix(".err")
        try:
            err_text = err_file.read_text(encoding="utf-8", errors="replace") if err_file.exists() else ""
        except OSError:
            err_text = ""

        runtime_s_raw = _first_match(TOTAL_RUNTIME_RE, out_text)
        runtime_s = float(runtime_s_raw) if runtime_s_raw is not None else None
        error_kind, error_type = _classify_error(err_text)

        matched_entries.append(
            {
                "out_file": out_file.name,
                "err_file": err_file.name if err_file.exists() else "",
                "seed": _first_match(SEED_RE, out_text),
                "qubits": _first_match(QUBITS_RE, out_text),
                "arch": _first_match(ARCH_RE, out_text),
                "runtime_s": runtime_s,
                "error_kind": error_kind,
                "error_type": error_type,
                "err_preview": err_text.strip().splitlines()[0] if err_text.strip() else "",
            }
        )

    lines: list[str] = []
    lines.append("# Run diagnostics")
    lines.append("")
    lines.append(f"- Run root: `{run_root}`")
    lines.append(f"- Matched log tasks: **{len(matched_entries)}**")

    if not matched_entries:
        lines.append("- No matching log files found (searched `outputs/logs/*.out`).")
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    runtimes = [entry["runtime_s"] for entry in matched_entries if entry["runtime_s"] is not None]
    if runtimes:
        lines.append(
            f"- Runtime seconds stats: min={min(runtimes):.2f}, "
            f"max={max(runtimes):.2f}, avg={sum(runtimes)/len(runtimes):.2f}"
        )
    else:
        lines.append("- Runtime seconds stats: unavailable (no `Total runtime:` found).")

    error_counts: dict[str, int] = {}
    for entry in matched_entries:
        key = str(entry["error_kind"])
        error_counts[key] = error_counts.get(key, 0) + 1
    lines.append("")
    lines.append("## Error summary")
    lines.append("")
    lines.append("| error_kind | count |")
    lines.append("|---|---:|")
    for key in sorted(error_counts):
        lines.append(f"| {key} | {error_counts[key]} |")

    lines.append("")
    lines.append("## Per-task details")
    lines.append("")
    lines.append(
        "| seed | qubits | arch | runtime_s | status | out_file | err_file | err_preview |"
    )
    lines.append("|---:|---:|---|---:|---|---|---|---|")
    for entry in sorted(
        matched_entries,
        key=lambda e: (
            int(e["qubits"]) if e["qubits"] and str(e["qubits"]).isdigit() else 999999,
            int(e["seed"]) if e["seed"] and str(e["seed"]).isdigit() else 999999,
            str(e["out_file"]),
        ),
    ):
        runtime_disp = f"{entry['runtime_s']:.2f}" if entry["runtime_s"] is not None else "n/a"
        err_label = entry["error_kind"]
        if entry["error_type"]:
            err_label = f"{err_label}:{entry['error_type']}"
        err_preview = (entry["err_preview"] or "").replace("|", "\\|")
        lines.append(
            f"| {entry['seed'] or '?'} "
            f"| {entry['qubits'] or '?'} "
            f"| {entry['arch'] or '?'} "
            f"| {runtime_disp} "
            f"| {err_label} "
            f"| `{entry['out_file']}` "
            f"| `{entry['err_file'] or '-'}` "
            f"| {err_preview} |"
        )

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate QNSPSA-vs-COBYLA results")
    parser.add_argument(
        "run_dir",
        type=str,
        nargs="?",
        default=None,
        help=(
            "Folder to analyze; results are written into <run_dir>/analysis. "
            "If omitted, analyze all run folders under outputs/."
        ),
    )
    parser.add_argument(
        "--last",
        type=int,
        nargs="?",
        const=1,
        default=None,
        help=(
            "Analyze only the last X discovered run folders under outputs/. "
            "Using --last without value is equivalent to --last 1."
        ),
    )
    return parser.parse_args()


def analyze_single_run(run_root: str) -> int:
    outdir = os.path.join(run_root, "analysis")
    os.makedirs(outdir, exist_ok=True)

    files = find_optimizer_history_files(run_root)
    if not files:
        raise ValueError(f"No optimizer_history.npz files found under: {run_root}")

    rows: List[RunSummary] = []
    for fp in files:
        try:
            rows.append(parse_single_run(fp))
        except Exception as exc:
            print(f"Skipping {fp}: {exc}")

    if not rows:
        raise ValueError(f"No valid runs found after parsing input files under: {run_root}")

    per_run_csv = os.path.join(outdir, "summary_per_run.csv")
    agg_csv = os.path.join(outdir, "summary_aggregated.csv")

    write_per_run_csv(rows, per_run_csv)
    agg = aggregate_rows(rows)
    write_aggregated_csv(agg, agg_csv)
    boxplot_paths = save_final_cost_boxplot(rows, agg, outdir)
    write_run_diagnostics_log(run_root, outdir)

    print(f"[ok] {run_root}")
    print(f"  Parsed runs: {len(rows)}")
    print(f"  Saved: {per_run_csv}")
    print(f"  Saved: {agg_csv}")
    for boxplot_path in boxplot_paths:
        print(f"  Saved: {boxplot_path}")
    print(f"  Saved: {os.path.join(outdir, 'run_diagnostics.md')}")
    return len(rows)


def discover_run_dirs(outputs_root: str = "outputs") -> List[str]:
    if not os.path.isdir(outputs_root):
        return []
    candidates: list[tuple[float, str]] = []
    for name in sorted(os.listdir(outputs_root)):
        full = os.path.join(outputs_root, name)
        if not os.path.isdir(full):
            continue
        if name in {"logs", ".runmeta"}:
            continue
        if find_optimizer_history_files(full):
            try:
                mtime = os.path.getmtime(full)
            except OSError:
                mtime = 0.0
            candidates.append((mtime, full))
    # Oldest -> newest by modification time.
    return [path for _, path in sorted(candidates, key=lambda item: item[0])]


def resolve_single_run_dir(run_arg: str) -> str:
    # If argument is an existing directory, use it directly.
    if os.path.isdir(run_arg):
        return run_arg

    # Prefer explicit outputs/ prefix if user provided only the run name.
    candidate = os.path.join("outputs", run_arg)
    if os.path.isdir(candidate):
        return candidate

    # Try to find a single matching directory inside outputs/ that contains the run_arg
    matches = []
    if os.path.isdir("outputs"):
        for name in os.listdir("outputs"):
            if run_arg in name:
                full = os.path.join("outputs", name)
                if os.path.isdir(full):
                    matches.append(full)

    if len(matches) == 1:
        run_root = matches[0]
        print(f"Using matched run directory: {run_root}")
        return run_root
    if len(matches) > 1:
        print("Multiple matching run directories found under outputs/:")
        for match in matches:
            print(" -", match)
        raise SystemExit("Provide a more specific run name or the full path to the run directory.")

    raise SystemExit(
        f"No optimizer_history.npz files found under: {run_arg}\nTried: {run_arg} and outputs/{run_arg}"
    )


def main() -> None:
    args = parse_args()
    if args.last is not None and args.last <= 0:
        raise SystemExit("--last must be a positive integer.")
    if args.last is not None and args.run_dir is not None:
        raise SystemExit("Use either run_dir or --last, not both together.")

    if args.run_dir is None:
        run_roots = discover_run_dirs("outputs")
        if not run_roots:
            raise SystemExit("No run folders with optimizer_history.npz found under outputs/")

        if args.last is not None:
            take = min(args.last, len(run_roots))
            run_roots = run_roots[-take:]
            print(f"Analyzing last {take} run folder(s) under outputs/.")

        total_rows = 0
        failures: List[str] = []
        print(f"Discovered {len(run_roots)} run folder(s) under outputs/.")
        for run_root in run_roots:
            try:
                total_rows += analyze_single_run(run_root)
            except Exception as exc:
                failures.append(run_root)
                print(f"[fail] {run_root}: {exc}")

        print(f"Finished. Successful runs: {len(run_roots) - len(failures)}/{len(run_roots)}")
        print(f"Total parsed sub-runs: {total_rows}")
        if failures:
            print("Failed run folders:")
            for fail in failures:
                print(" -", fail)
        return

    # Allow passing either a full path or a run-name identifier.
    run_arg = args.run_dir.strip()
    run_root = resolve_single_run_dir(run_arg)
    analyze_single_run(run_root)


if __name__ == "__main__":
    main()
