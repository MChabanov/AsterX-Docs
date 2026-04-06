#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from plot1d_common import (
    ENERGY_CU_TO_ERG,
    TIME_CU_TO_MS,
    add_shared_arguments,
    gather_output_data_dirs,
    plt,
    prepare_output_dir,
    style_1d_axis,
)


HEADER_COLUMN_RE = re.compile(r"^#\s*Col\.\s*(\d+)\s*:\s*([^.]+)")
DEFAULT_COLUMN_NAME = "magnetic_energy_total"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the stitched total magnetic energy from volume_integrals-GRMHDX.asc."
    )
    add_shared_arguments(parser)
    parser.add_argument(
        "--column-name",
        default=DEFAULT_COLUMN_NAME,
        help="Column name to extract from volume_integrals-GRMHDX.asc.",
    )
    return parser.parse_args()


def _time_key(value: float) -> float:
    return round(float(value), 10)


def find_column_index(file_path: Path, column_name: str) -> int:
    with file_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            match = HEADER_COLUMN_RE.match(line.strip())
            if not match:
                continue
            column_number = int(match.group(1))
            label = match.group(2).strip()
            if label == column_name:
                return column_number - 1
    raise ValueError(f"Column '{column_name}' not found in {file_path}")


def gather_integral_files(data_dirs: list[Path]) -> list[Path]:
    files = []
    relative_path = Path("volume_integration") / "volume_integrals-GRMHDX.asc"
    for data_dir in data_dirs:
        file_path = data_dir / relative_path
        if file_path.is_file():
            files.append(file_path)
    if not files:
        raise FileNotFoundError("No volume_integrals-GRMHDX.asc files found under the simulation path")
    return files


def stitch_integral_column(files: list[Path], column_name: str) -> tuple[np.ndarray, np.ndarray, int]:
    column_index = find_column_index(files[0], column_name)
    combined = {}

    for file_path in files:
        data = np.loadtxt(file_path, comments="#")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] <= column_index:
            raise ValueError(f"Column index {column_index} is out of range for {file_path}")

        for row in data:
            time_cu = float(row[0])
            value = float(row[column_index])
            combined[_time_key(time_cu)] = (time_cu, value)

    stitched = np.array([combined[key] for key in sorted(combined)], dtype=float)
    return stitched[:, 0], stitched[:, 1], column_index


def main() -> None:
    args = parse_args()
    _sim_root, sim_name, data_dirs = gather_output_data_dirs(args.sim_dir)
    _sim_root, _sim_name, out_dir = prepare_output_dir(args.sim_dir, "magnetic_energy", args.out_root)

    files = gather_integral_files(data_dirs)
    time_cu, energy_cu, column_index = stitch_integral_column(files, args.column_name)
    time_ms = time_cu * TIME_CU_TO_MS
    energy_erg = energy_cu * ENERGY_CU_TO_ERG
    line_color = "#b45309"

    linear_path = out_dir / f"{sim_name}_{args.column_name}_linear.png"
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.plot(time_ms, energy_erg, linewidth=1.7, color=line_color)
    ax.set_xlabel(r"$t$ [ms]")
    ax.set_ylabel(r"$E_{\rm mag}$ [erg]")
    style_1d_axis(ax)
    fig.tight_layout()
    fig.savefig(linear_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    positive_mask = energy_erg > 0.0
    if not np.any(positive_mask):
        raise ValueError(f"Column '{args.column_name}' has no positive samples for the log plot")

    log_path = out_dir / f"{sim_name}_{args.column_name}_log.png"
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.semilogy(time_ms[positive_mask], energy_erg[positive_mask], linewidth=1.7, color=line_color)
    ax.set_xlabel(r"$t$ [ms]")
    ax.set_ylabel(r"$E_{\rm mag}$ [erg]")
    style_1d_axis(ax)
    fig.tight_layout()
    fig.savefig(log_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    summary_path = out_dir / "README_summary.txt"
    with summary_path.open("w", encoding="utf-8") as stream:
        stream.write("Magnetic energy from volume_integrals-GRMHDX.asc\n")
        stream.write(f"Simulation: {sim_name}\n")
        stream.write(f"Column: {args.column_name} (0-based index {column_index})\n")
        stream.write(f"Files stitched: {len(files)}\n")
        stream.write(f"Samples stitched: {len(time_cu)}\n")
        stream.write(f"Time range: [{float(time_ms.min()):.6f}, {float(time_ms.max()):.6f}] ms\n")
        stream.write(
            f"Positive samples for log plot: {int(np.count_nonzero(positive_mask))}\n"
        )
        stream.write(f"Energy conversion: erg = code_energy * {ENERGY_CU_TO_ERG:.16e}\n")

    print(f"Simulation: {sim_name}")
    print(f"Column: {args.column_name} (0-based index {column_index})")
    print("Created files:")
    print(f" - {linear_path}")
    print(f" - {log_path}")
    print(f" - {summary_path}")


if __name__ == "__main__":
    main()
