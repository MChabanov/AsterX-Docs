#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
from cycler import cycler

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from plot2d_common import DEFAULT_SIM_DIR, infer_sim_name as infer_movie_sim_name
from unit_converter import CU_CGS


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "plots" / "1d"

TIME_CU_TO_MS = CU_CGS.time / 1.0e-3
ENERGY_CU_TO_ERG = CU_CGS.energy

_SPECIAL_SUBDIR_PREFIXES = ("checkpoint", "git_info")
USE_TEX = shutil.which("latex") is not None

plt.rcParams.update(
    {
        "text.usetex": bool(USE_TEX),
        "font.family": "serif" if USE_TEX else "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.prop_cycle": cycler(
            color=["#0f766e", "#2563eb", "#7c3aed", "#db2777", "#d97706", "#059669"]
        ),
    }
)


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "sim_dir",
        nargs="?",
        default=DEFAULT_SIM_DIR,
        help="Simulation root, output directory, or inner data directory.",
    )
    parser.add_argument(
        "--out-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory where the plot folder will be written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Saved figure resolution.",
    )


def infer_sim_root(sim_dir: str | Path) -> Path:
    path = Path(sim_dir).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Simulation path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Simulation path must be a directory: {path}")

    if any(child.is_dir() and child.name.startswith("output-") for child in path.iterdir()):
        return path

    if path.name.startswith("output-"):
        return path.parent

    if path.parent.name.startswith("output-"):
        return path.parent.parent

    return path


def infer_sim_name(sim_dir: str | Path) -> str:
    return infer_movie_sim_name(str(Path(sim_dir).expanduser().resolve()), [])


def _resolve_output_data_dir(output_dir: Path, preferred_name: str) -> Path:
    preferred_dir = output_dir / preferred_name
    if preferred_dir.is_dir():
        return preferred_dir

    candidates = [
        child
        for child in sorted(output_dir.iterdir())
        if child.is_dir() and not child.name.startswith(_SPECIAL_SUBDIR_PREFIXES)
    ]
    if len(candidates) == 1:
        return candidates[0]
    return output_dir


def gather_output_data_dirs(sim_dir: str | Path) -> tuple[Path, str, list[Path]]:
    sim_root = infer_sim_root(sim_dir)
    sim_name = infer_sim_name(sim_dir)
    output_dirs = sorted(
        child for child in sim_root.iterdir() if child.is_dir() and child.name.startswith("output-")
    )

    if output_dirs:
        data_dirs = [_resolve_output_data_dir(output_dir, sim_name) for output_dir in output_dirs]
    else:
        data_dirs = [Path(sim_dir).expanduser().resolve()]

    return sim_root, sim_name, data_dirs


def prepare_output_dir(sim_dir: str | Path, plot_name: str, out_root: str | Path) -> tuple[Path, str, Path]:
    sim_root, sim_name, _data_dirs = gather_output_data_dirs(sim_dir)
    out_dir = Path(out_root).expanduser().resolve() / f"{sim_name}_{plot_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return sim_root, sim_name, out_dir


def radius_to_tag(radius: float) -> str:
    rounded = round(float(radius))
    if abs(float(radius) - rounded) < 1.0e-9:
        return str(int(rounded))
    return f"{float(radius):.2f}".replace(".", "p")


def style_1d_axis(ax) -> None:
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.grid(True, which="major", alpha=0.20, linewidth=0.7)
    ax.grid(True, which="minor", alpha=0.08, linewidth=0.5)
    ax.tick_params(axis="both", which="both", direction="out")
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#374151")
        ax.spines[side].set_linewidth(0.9)
    if ax.legend_ is not None:
        ax.legend_.set_frame_on(False)
