#!/usr/bin/env python3
import argparse
import glob
import os
import re


DEFAULT_SIM_DIR = "/scratch/09228/jkalinan/simulations/AsterX_BNS_APRLDP_RPA_8lvl_sc_fixedGrid_dx025"
DEFAULT_OUTPUT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "plots",
    "movies",
)
DEFAULT_DOMAIN_HALF_WIDTH_CU = 150.0

_SERIES_RE = re.compile(r"^(?P<name>.+)\.it(?P<it>\d+)\.bp\d*$")


def parse_itnum(path):
    match = _SERIES_RE.match(os.path.basename(path.rstrip(os.sep)))
    return int(match.group("it")) if match else 0


def infer_sim_name(sim_dir, series_files):
    if series_files:
        match = _SERIES_RE.match(os.path.basename(series_files[0].rstrip(os.sep)))
        if match:
            return match.group("name")

    sim_dir = os.path.abspath(os.path.expanduser(sim_dir))
    base = os.path.basename(sim_dir.rstrip(os.sep))
    if base.startswith("output-"):
        try:
            subdirs = [
                name for name in sorted(os.listdir(sim_dir))
                if os.path.isdir(os.path.join(sim_dir, name))
                and not name.startswith("git_info")
                and not name.startswith("checkpoint")
            ]
        except OSError:
            subdirs = []
        if len(subdirs) == 1:
            return subdirs[0]
    return base


def _is_series_path(path):
    name = os.path.basename(path.rstrip(os.sep))
    if ".md." in name or name.endswith(".dir"):
        return False
    if not _SERIES_RE.match(name):
        return False

    parts = os.path.normpath(path).split(os.sep)
    return not any(part.startswith("checkpoint") for part in parts)


def gather_series_files(sim_dir):
    sim_dir = os.path.abspath(os.path.expanduser(sim_dir))
    if _is_series_path(sim_dir):
        return [sim_dir]

    patterns = [
        os.path.join(sim_dir, "*.bp*"),
        os.path.join(sim_dir, "*", "*.bp*"),
        os.path.join(sim_dir, "output-*", "*.bp*"),
        os.path.join(sim_dir, "output-*", "*", "*.bp*"),
    ]

    matches = {}
    for pattern in patterns:
        for path in glob.glob(pattern):
            if not _is_series_path(path):
                continue
            matches[os.path.abspath(path)] = parse_itnum(path)

    return [
        path for path, _it in
        sorted(matches.items(), key=lambda item: (item[1], item[0]))
    ]


def resolve_movie_paths(sim_dir, field_name, out_root=DEFAULT_OUTPUT_ROOT):
    sim_dir = os.path.abspath(os.path.expanduser(sim_dir))
    out_root = os.path.abspath(os.path.expanduser(out_root))
    series_files = gather_series_files(sim_dir)
    sim_name = infer_sim_name(sim_dir, series_files)
    out_dir = os.path.join(out_root, f"{sim_name}_{field_name}_frames")
    movie_file = os.path.join(out_dir, f"{sim_name}_{field_name}.mp4")
    return series_files, sim_name, out_dir, movie_file


def parse_movie_args(field_name, default_sim_dir=DEFAULT_SIM_DIR,
                     default_domain_half_width_cu=DEFAULT_DOMAIN_HALF_WIDTH_CU,
                     default_output_root=DEFAULT_OUTPUT_ROOT,
                     default_fps=1,
                     default_nxny=1024,
                     default_vmin=None,
                     default_vmax=None,
                     default_merger_time_ms=14.0,
                     default_final_after_ms=30.0,
                     default_level_mode="auto"):
    parser = argparse.ArgumentParser(
        description=f"Plot {field_name} slices and assemble a movie from openPMD outputs."
    )
    parser.add_argument(
        "sim_dir",
        nargs="?",
        default=default_sim_dir,
        help="Simulation directory, output directory, data directory, or single .bp* series.",
    )
    parser.add_argument(
        "--out-root",
        default=default_output_root,
        help="Root directory where frame folders and movies will be written.",
    )
    parser.add_argument(
        "--domain-half-width-cu",
        type=float,
        default=default_domain_half_width_cu,
        help="Half-width of the symmetric plotting domain in code units.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=default_fps,
        help="Movie frame rate.",
    )
    parser.add_argument(
        "--nxny",
        type=int,
        default=default_nxny,
        help="Canvas resolution along one axis for each slice panel.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=default_vmin,
        help="Override the lower color scale bound.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=default_vmax,
        help="Override the upper color scale bound.",
    )
    parser.add_argument(
        "--merger-time-ms",
        type=float,
        default=default_merger_time_ms,
        help="Subtract this time from the displayed iteration time.",
    )
    parser.add_argument(
        "--final-after-ms",
        type=float,
        default=default_final_after_ms,
        help="Stop when the displayed time exceeds this value. Use a negative value to disable.",
    )
    parser.add_argument(
        "--level-mode",
        choices=("auto", "composite", "finest", "coarsest"),
        default=default_level_mode,
        help="How to combine AMR/fixed-grid levels in the plotted window.",
    )
    return parser.parse_args()
