#!/usr/bin/env python3
"""Visualize and summarize phase 2 circuit architectures."""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

from phase2.architectures import available_architectures, build_architecture, parse_depth_split


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _is_valid_combo(architecture: str, n_qubits: int) -> bool:
    if architecture == "qcnn":
        return _is_power_of_two(n_qubits)
    return True


def _write_text_diagram(path: Path, circuit) -> None:
    path.write_text(str(circuit.draw(output="text")), encoding="utf-8")


def _architecture_notes(architecture: str, metadata: dict) -> str:
    if architecture == "baseline_hea":
        reps = metadata.get("reps", "?")
        entanglement = metadata.get("entanglement", "linear")
        return (
            "Blocks:\n"
            "- Hardware-efficient SU2 layers\n"
            f"- Repetitions: {reps}\n"
            f"- Entanglement: {entanglement}"
        )

    if architecture == "qcnn":
        levels = metadata.get("levels", "?")
        conv_sublayers = metadata.get("conv_sublayers_per_level", "?")
        pooling_direction = metadata.get("pooling_direction", "A_to_B_keep_B")
        return (
            "Blocks:\n"
            f"- {conv_sublayers} convolution sublayers per level\n"
            "- Pairwise hierarchical pooling\n"
            f"- Pooling direction: {pooling_direction}\n"
            f"- Levels: {levels} (log2 hierarchy)"
        )

    if architecture == "resqnet":
        depth_split = metadata.get("depth_split", ["?", "?"])
        residual_mode = metadata.get("residual_mode", "structural")
        shared_reupload = metadata.get("shared_reupload", False)
        layer_design = metadata.get("layer_design", "rx-ry + linear_cnot")
        return (
            "Blocks:\n"
            "- Two-node residual quantum network\n"
            f"- Depth split (D1,D2): {depth_split[0]},{depth_split[1]}\n"
            f"- Residual mode: {residual_mode}\n"
            f"- Shared re-upload: {shared_reupload}\n"
            f"- Layer design: {layer_design}"
        )

    return "Blocks:\n- Custom architecture"


def _write_mpl_diagram(path: Path, circuit, *, architecture: str, summary: dict, metadata: dict) -> bool:
    try:
        figure = circuit.draw(output="mpl")
        info = (
            f"architecture={architecture} | n_qubits={summary['n_qubits']} | "
            f"num_params={summary['num_params']} | depth={summary['depth']} | "
            f"size={summary['size']} | readout_qubit={summary['readout_qubit']}"
        )
        notes = _architecture_notes(architecture=architecture, metadata=metadata)
        figure.suptitle("Architecture visualization", fontsize=12, y=0.98)
        figure.subplots_adjust(bottom=0.23)
        figure.text(
            0.01,
            0.01,
            f"{info}\n{notes}",
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        figure.savefig(path, dpi=180, bbox_inches="tight")
        figure.clf()
        return True
    except Exception:
        return False


def _render_one(
    architecture: str,
    n_qubits: int,
    output_dir: Path,
    *,
    resqnet_depth_split: tuple[int, int] = (5, 1),
    resqnet_residual_mode: str = "structural",
) -> dict[str, int | str]:
    spec = build_architecture(
        architecture=architecture,
        n_qubits=n_qubits,
        resqnet_depth_split=resqnet_depth_split,
        resqnet_residual_mode=resqnet_residual_mode,
    )
    circuit = spec.circuit

    if architecture == "resqnet":
        split_label = f"d{resqnet_depth_split[0]}-{resqnet_depth_split[1]}"
        run_dir = output_dir / architecture / split_label / f"q{n_qubits}"
    else:
        run_dir = output_dir / architecture / f"q{n_qubits}"
    run_dir.mkdir(parents=True, exist_ok=True)

    text_path = run_dir / "diagram.txt"
    _write_text_diagram(text_path, circuit)

    summary = {
        "architecture": architecture,
        "n_qubits": n_qubits,
        "num_params": spec.num_parameters,
        "depth": int(circuit.depth()),
        "size": int(circuit.size()),
        "readout_qubit": int(spec.readout_qubit),
    }

    png_path = run_dir / "diagram.png"
    png_ok = _write_mpl_diagram(
        png_path,
        circuit,
        architecture=architecture,
        summary=summary,
        metadata=spec.metadata,
    )

    row = {
        **summary,
        "resqnet_depth_split": ",".join(map(str, spec.metadata.get("depth_split", []))),
        "resqnet_residual_mode": str(spec.metadata.get("residual_mode", "")),
        "png_generated": "yes" if png_ok else "no",
        "text_diagram": str(text_path.relative_to(output_dir)),
        "png_diagram": str(png_path.relative_to(output_dir)) if png_ok else "",
    }

    run_summary_path = run_dir / "summary.csv"
    with run_summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    extra = ""
    if architecture == "resqnet":
        extra = (
            f" depth_split={resqnet_depth_split[0]},{resqnet_depth_split[1]}"
            f" residual_mode={resqnet_residual_mode}"
        )
    print(
        f"[{architecture}] q={n_qubits} params={spec.num_parameters} depth={circuit.depth()} "
        f"readout={spec.readout_qubit}{extra} dir={run_dir}"
    )
    return row


def visualize(
    architectures: list[str],
    qubit_values: list[int],
    output_dir: Path,
    *,
    resqnet_depth_splits: list[tuple[int, int]],
    resqnet_residual_mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, int | str]] = []

    for architecture, n_qubits in itertools.product(architectures, qubit_values):
        if not _is_valid_combo(architecture=architecture, n_qubits=n_qubits):
            print(
                f"[skip] architecture={architecture} only supports power-of-two qubits; "
                f"got n_qubits={n_qubits}"
            )
            continue
        if architecture == "resqnet":
            for depth_split in resqnet_depth_splits:
                summary_rows.append(
                    _render_one(
                        architecture=architecture,
                        n_qubits=n_qubits,
                        output_dir=output_dir,
                        resqnet_depth_split=depth_split,
                        resqnet_residual_mode=resqnet_residual_mode,
                    )
                )
        else:
            summary_rows.append(_render_one(architecture=architecture, n_qubits=n_qubits, output_dir=output_dir))

    summary_path = output_dir / "architecture_summary_all.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "architecture",
                "n_qubits",
                "num_params",
                "depth",
                "size",
                "readout_qubit",
                "resqnet_depth_split",
                "resqnet_residual_mode",
                "png_generated",
                "text_diagram",
                "png_diagram",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary written to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize phase 2 architectures")
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=list(available_architectures()),
        choices=available_architectures(),
        help="Architectures to render",
    )
    parser.add_argument("--n-qubits", type=int, default=8, help="Number of qubits")
    parser.add_argument(
        "--qubits",
        nargs="+",
        type=int,
        default=None,
        help="List of qubit counts to render (if omitted, uses --n-qubits)",
    )
    parser.add_argument(
        "--all-combinations",
        action="store_true",
        help="Render all combinations between available architectures and selected qubits",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/circuits"),
        help="Directory where diagrams and summary are written",
    )
    parser.add_argument(
        "--resqnet-depth-splits",
        nargs="+",
        default=["5,1"],
        help=(
            "One or more depth splits for resqnet in D1,D2 format. "
            "Example: --resqnet-depth-splits 5,1 4,2 3,3"
        ),
    )
    parser.add_argument(
        "--resqnet-residual-mode",
        type=str,
        default="structural",
        help="Residual mode for resqnet (current supported value: structural)",
    )
    args = parser.parse_args()

    architectures = list(available_architectures()) if args.all_combinations else args.architectures
    qubit_values = args.qubits if args.qubits else [args.n_qubits]
    resqnet_depth_splits = [parse_depth_split(raw) for raw in args.resqnet_depth_splits]

    visualize(
        architectures=architectures,
        qubit_values=qubit_values,
        output_dir=args.output_dir,
        resqnet_depth_splits=resqnet_depth_splits,
        resqnet_residual_mode=args.resqnet_residual_mode,
    )


if __name__ == "__main__":
    main()
