#!/usr/bin/env python3
"""Render an off-screen 3D rho contour movie from openPMD outputs with PyVista.

Typical use from this directory:
    module load adios2
    python3 plot3d_rho.py

Quick smoke test:
    module load adios2
    python3 plot3d_rho.py --max-frames 1 --grid-size 64

Higher-quality example:
    module load adios2
    python3 plot3d_rho.py --grid-size 224 --width 1440 --height 1088 --contour-count 8
"""

from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import numpy as np

try:
    import openpmd_api as io
except ImportError as exc:
    raise ImportError(
        "Failed to import openpmd_api. On Vista, load the ADIOS2 runtime first "
        "with `module load adios2`."
    ) from exc

from plot3d_common import (
    CU_TO_KM,
    DEFAULT_3D_THEME,
    DEFAULT_3D_SCALAR_BAR_HEIGHT,
    DEFAULT_3D_SCALAR_BAR_LABEL_SIZE,
    DEFAULT_3D_SCALAR_BAR_TITLE_SIZE,
    DEFAULT_3D_SCALAR_BAR_Y,
    DEFAULT_3D_RESOLUTION,
    DEFAULT_3D_SCALAR_BAR_WIDTH,
    DEFAULT_3D_SCALAR_BAR_X,
    DEFAULT_3D_TIME_LABEL_POSITION,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PYVISTA_ROOT,
    DEFAULT_SIM_DIR,
    RHO_CU_TO_CGS,
    TIME_CU_TO_MS,
    composite_scalar_volume,
    compute_mesh_bbox,
    get_3d_theme,
    get_time_code_units,
    load_pyvista,
    parse_itnum,
    resolve_3d_paths,
    sanitize_scalar_field,
    select_series_files,
)


RENDER_NAME = "rho3d_pyvista"
REC_COMP = "hydrobasex_rho"
REC_NAME_RE = re.compile(r"^hydrobasex_rho_patch(\d+)_lev(\d+)$")
SCALAR_NAME = "log10_rho"

# ----------------------- default run settings -----------------------
# Data sampling and composite-grid controls.
CADENCE = 1
GRID_SIZE = 256
EDGE_ERODE = 1
Z_CHUNK = 16
TILE_XY = 256
DOWNSAMPLE = 1

# The default manual box is centered on the origin and comfortably contains
# the stars at the earliest times for this simulation.
BBOX_HALF_WIDTH_CU = 140.0
BBOX_PAD_FRAC = 0.02
AUTO_BBOX = False

# Contour and transfer-function controls for the rho rendering.
# `CONTOUR_COUNT` sets how many iso-surfaces are drawn.
# `CONTOUR_LOW_FRAC` sets where the lowest contour sits between the log10(rho)
# window minimum and maximum; larger values emphasize only the densest matter.
PCT_LOW = 45.0
PCT_HIGH = 99.8
DYNAMIC_RANGE_DEX = 5.0
MAX_BRIGHT_GAP_DEX = 1.0
CONTOUR_COUNT = 8
CONTOUR_LOW_FRAC = 0.55
CONTOUR_OPACITY = 0.22
SHOW_COLORBAR = True
COLORBAR_TITLE = "log10 rho\n[g cm^-3]"
RHO_CMAP = "inferno"

# Movie and camera controls. The default frame is narrower along x so the
# centered render spends less space on empty side margins.
FPS = 6
RESOLUTION = DEFAULT_3D_RESOLUTION
SPIN_DEG = 0.0
FAN_DEG = 0.0
CAMERA_ZOOM = 1.2
SAVE_SNAPSHOTS = True
MERGER_TIME_MS = 15.0
FINAL_AFTER_MS = None
TIME_LABEL_FONT_SIZE = 22
AXES_LINE_WIDTH = 3.0
# ---------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a 3D rho contour movie from openPMD outputs using PyVista."
    )
    parser.add_argument(
        "sim_dir",
        nargs="?",
        default=DEFAULT_SIM_DIR,
        help="Simulation directory, output directory, data directory, or single .bp* series root.",
    )
    parser.add_argument(
        "--out-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory where frames and the mp4 movie will be written.",
    )
    parser.add_argument(
        "--cadence",
        type=int,
        default=CADENCE,
        help="Render every Nth openPMD file.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on rendered frames for test runs.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=GRID_SIZE,
        help="Uniform output grid size along each axis. Higher gives smoother stars but costs more time.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=RESOLUTION[0],
        help="Output image width in pixels for snapshots and the movie.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=RESOLUTION[1],
        help="Output image height in pixels for snapshots and the movie.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=FPS,
        help="Movie frame rate.",
    )
    parser.add_argument(
        "--spin-deg",
        type=float,
        default=SPIN_DEG,
        help="Camera azimuth increment per rendered frame.",
    )
    parser.add_argument(
        "--fan-deg",
        type=float,
        default=FAN_DEG,
        help="Camera elevation increment per rendered frame.",
    )
    parser.add_argument(
        "--camera-zoom",
        type=float,
        default=CAMERA_ZOOM,
        help="Additional camera zoom factor. Values above 1 zoom in.",
    )
    parser.add_argument(
        "--no-snapshots",
        action="store_true",
        help="Disable per-frame PNG snapshots alongside the mp4 movie.",
    )
    parser.add_argument(
        "--time-label-font-size",
        type=int,
        default=TIME_LABEL_FONT_SIZE,
        help="Font size of the on-frame time label.",
    )
    parser.add_argument(
        "--axes-line-width",
        type=float,
        default=AXES_LINE_WIDTH,
        help="Line width used for the orientation axes widget.",
    )
    parser.add_argument(
        "--theme",
        choices=("dark", "light"),
        default=DEFAULT_3D_THEME,
        help="Annotation theme: dark = black background with white labels, light = white background with black labels.",
    )
    parser.add_argument(
        "--rho-cmap",
        default=RHO_CMAP,
        help="Colormap used for the rho contours and rho colorbar, e.g. inferno, plasma, Blues.",
    )
    parser.add_argument(
        "--colorbar-x",
        type=float,
        default=DEFAULT_3D_SCALAR_BAR_X,
        help="Horizontal position of the rho colorbar in normalized viewport coordinates.",
    )
    parser.add_argument(
        "--colorbar-y",
        type=float,
        default=DEFAULT_3D_SCALAR_BAR_Y,
        help="Vertical position of the rho colorbar in normalized viewport coordinates.",
    )
    parser.add_argument(
        "--colorbar-width",
        type=float,
        default=DEFAULT_3D_SCALAR_BAR_WIDTH,
        help="Normalized viewport width of the rho colorbar.",
    )
    parser.add_argument(
        "--colorbar-height",
        type=float,
        default=DEFAULT_3D_SCALAR_BAR_HEIGHT,
        help="Normalized viewport height of the rho colorbar.",
    )
    parser.add_argument(
        "--colorbar-title-size",
        type=int,
        default=DEFAULT_3D_SCALAR_BAR_TITLE_SIZE,
        help="Font size of the rho colorbar title.",
    )
    parser.add_argument(
        "--colorbar-label-size",
        type=int,
        default=DEFAULT_3D_SCALAR_BAR_LABEL_SIZE,
        help="Font size of the rho colorbar tick labels.",
    )
    parser.add_argument(
        "--bbox-half-width-cu",
        type=float,
        default=BBOX_HALF_WIDTH_CU,
        help="Manual half-width in code units if auto-bbox is disabled.",
    )
    parser.add_argument(
        "--auto-bbox",
        action="store_true",
        help="Auto-fit the bbox from the first frame instead of using the symmetric manual box.",
    )
    parser.add_argument(
        "--pct-low",
        type=float,
        default=PCT_LOW,
        help="Lower percentile used to set the rho transfer-function window.",
    )
    parser.add_argument(
        "--pct-high",
        type=float,
        default=PCT_HIGH,
        help="Upper percentile used to set the rho transfer-function window.",
    )
    parser.add_argument(
        "--dynamic-range-dex",
        type=float,
        default=DYNAMIC_RANGE_DEX,
        help="Minimum log10 dynamic range kept below the bright-end rho cutoff.",
    )
    parser.add_argument(
        "--max-bright-gap-dex",
        type=float,
        default=MAX_BRIGHT_GAP_DEX,
        help="Clamp the bright-end cutoff to stay within this many dex of the true rho maximum.",
    )
    parser.add_argument(
        "--rho-log-min",
        type=float,
        default=None,
        help="Override the lower end of the log10(rho [g cm^-3]) display range.",
    )
    parser.add_argument(
        "--rho-log-max",
        type=float,
        default=None,
        help="Override the upper end of the log10(rho [g cm^-3]) display range.",
    )
    parser.add_argument(
        "--contour-count",
        type=int,
        default=CONTOUR_COUNT,
        help="Number of rho iso-surfaces to render.",
    )
    parser.add_argument(
        "--contour-low-frac",
        type=float,
        default=CONTOUR_LOW_FRAC,
        help="Lowest contour position as a fraction of the log10(rho) window span.",
    )
    parser.add_argument(
        "--contour-opacity",
        type=float,
        default=CONTOUR_OPACITY,
        help="Opacity applied to the rho contour surfaces.",
    )
    parser.add_argument(
        "--no-colorbar",
        action="store_true",
        help="Disable the rho colorbar in snapshots and the movie.",
    )
    parser.add_argument(
        "--pyvista-root",
        default=str(DEFAULT_PYVISTA_ROOT),
        help="Vendored PyVista checkout to try before falling back to the system package.",
    )
    parser.add_argument(
        "--final-after-ms",
        type=float,
        default=FINAL_AFTER_MS,
        help="Stop once displayed time exceeds this threshold. Omit to render all selected frames.",
    )
    parser.add_argument(
        "--merger-time-ms",
        type=float,
        default=MERGER_TIME_MS,
        help="Merger time subtracted when applying --final-after-ms.",
    )
    return parser.parse_args()


def make_manual_bbox(half_width_cu: float):
    half_width_km = float(half_width_cu) * CU_TO_KM
    return (
        (-half_width_km, half_width_km),
        (-half_width_km, half_width_km),
        (-half_width_km, half_width_km),
    )


def build_image_data(pv, data: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
    # PyVista expects point data flattened in Fortran order for image volumes.
    nx, ny, nz = data.shape
    grid = pv.ImageData(dimensions=(nx, ny, nz))
    dx = (xs[-1] - xs[0]) / (nx - 1) if nx > 1 else 1.0
    dy = (ys[-1] - ys[0]) / (ny - 1) if ny > 1 else 1.0
    dz = (zs[-1] - zs[0]) / (nz - 1) if nz > 1 else 1.0
    grid.spacing = (dx, dy, dz)
    grid.origin = (xs[0], ys[0], zs[0])
    grid.point_data[SCALAR_NAME] = data.ravel(order="F")
    return grid


def build_log_volume(volume_cgs: np.ndarray, floor_value: float) -> np.ndarray:
    clipped = np.clip(volume_cgs, floor_value, None)
    return np.log10(clipped).astype(np.float32, copy=False)


def compute_transfer_window(
    volume_cgs: np.ndarray,
    pct_low: float,
    pct_high: float,
    dynamic_range_dex: float,
    max_bright_gap_dex: float,
):
    # Anchor the bright end close to the true rho maximum so the initial stars
    # are not washed out by the atmosphere-dominated voxel distribution.
    positive = volume_cgs[volume_cgs > 0.0]
    if positive.size == 0:
        low = 1.0
        high = 10.0
    else:
        low_pct, high_pct = np.percentile(positive, [pct_low, pct_high])
        vmax = max(float(np.max(positive)), np.finfo(np.float32).tiny)
        high = max(float(high_pct), vmax / (10.0 ** float(max_bright_gap_dex)))
        low = max(float(low_pct), high / (10.0 ** float(dynamic_range_dex)))
        high = max(float(high), low * 10.0)

    floor_value = max(low * 0.5, np.finfo(np.float32).tiny)
    clim = (float(np.log10(floor_value)), float(np.log10(high)))
    return floor_value, clim


def build_contour_levels(
    clim,
    count: int,
    low_frac: float,
):
    lo, hi = clim
    start = lo + float(low_frac) * (hi - lo)
    if start >= hi:
        start = hi - 0.5
    return np.linspace(start, hi, max(2, int(count)))


def apply_log_range_override(
    floor_value: float,
    clim,
    rho_log_min: float | None,
    rho_log_max: float | None,
):
    log_min, log_max = clim
    if rho_log_min is not None:
        log_min = float(rho_log_min)
    if rho_log_max is not None:
        log_max = float(rho_log_max)
    if log_max <= log_min:
        raise ValueError(
            f"Invalid rho log range: min={log_min:.6f} must be smaller than max={log_max:.6f}"
        )
    floor_value = 10.0 ** log_min
    return floor_value, (log_min, log_max)


def set_camera(plotter, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> None:
    cx = 0.5 * (xs[0] + xs[-1])
    cy = 0.5 * (ys[0] + ys[-1])
    cz = 0.5 * (zs[0] + zs[-1])
    rx = xs[-1] - xs[0]
    ry = ys[-1] - ys[0]
    rz = zs[-1] - zs[0]
    plotter.camera.focal_point = (cx, cy, cz)
    plotter.camera.position = (cx + 0.85 * rx, cy + 0.45 * ry, cz + 0.40 * rz)
    plotter.camera.up = (0.0, 0.0, 1.0)


def update_frame_label(
    plotter,
    time_ms: float | None,
    time_shift_ms: float = 0.0,
    font_size: int = TIME_LABEL_FONT_SIZE,
    text_color: str = "white",
    text_shadow: bool = True,
) -> None:
    label = "t = n/a"
    if time_ms is not None:
        label = f"t = {time_ms - float(time_shift_ms):6.2f} ms"
    plotter.add_text(
        label,
        position=DEFAULT_3D_TIME_LABEL_POSITION,
        font_size=int(font_size),
        color=text_color,
        name="time_label",
        shadow=text_shadow,
        viewport=True,
    )


def should_stop(time_cu: float | None, merger_time_ms: float, final_after_ms: float | None) -> bool:
    if time_cu is None or final_after_ms is None:
        return False
    return (time_cu * TIME_CU_TO_MS - merger_time_ms) > final_after_ms


def write_summary(
    summary_path: Path,
    sim_name: str,
    movie_file: Path,
    frame_count: int,
    bbox,
    grid_size: int,
    cadence: int,
    pyvista_source: str,
    clim,
    snapshot_dir: Path | None,
    resolution,
    contour_count: int,
    contour_low_frac: float,
    contour_opacity: float,
) -> None:
    with summary_path.open("w", encoding="utf-8") as stream:
        stream.write("PyVista 3D rho contour movie\n")
        stream.write(f"Simulation: {sim_name}\n")
        stream.write(f"PyVista source: {pyvista_source}\n")
        stream.write(f"Movie: {movie_file}\n")
        if snapshot_dir is not None:
            stream.write(f"Snapshots: {snapshot_dir}\n")
        stream.write(f"Frames rendered: {frame_count}\n")
        stream.write(f"Cadence: every {cadence} file(s)\n")
        stream.write(f"Grid size: {grid_size}^3\n")
        stream.write(f"Resolution [px]: {resolution[0]} x {resolution[1]}\n")
        stream.write(f"BBox [km]: X{bbox[0]} Y{bbox[1]} Z{bbox[2]}\n")
        stream.write(
            "Contours: "
            f"count={contour_count}, low_frac={contour_low_frac:.3f}, opacity={contour_opacity:.3f}\n"
        )
        stream.write(f"log10(rho) clim: ({clim[0]:.6f}, {clim[1]:.6f})\n")


def main() -> None:
    args = parse_args()
    pv, pyvista_source = load_pyvista(args.pyvista_root)
    theme = get_3d_theme(args.theme)
    save_snapshots = SAVE_SNAPSHOTS and not args.no_snapshots
    show_colorbar = SHOW_COLORBAR and not args.no_colorbar
    resolution = (int(args.width), int(args.height))

    all_series_files, sim_name, out_dir, movie_file = resolve_3d_paths(
        args.sim_dir,
        render_name=RENDER_NAME,
        out_root=args.out_root,
    )
    snapshot_dir = out_dir / "snapshots" if save_snapshots else None
    if snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    selected_files = select_series_files(
        all_series_files,
        cadence=args.cadence,
        max_frames=args.max_frames,
        include_last=True,
    )

    if not selected_files:
        raise RuntimeError(f"No openPMD series files found under {args.sim_dir}")

    print(f"Using PyVista: {pyvista_source}")
    print(f"Found {len(all_series_files)} openPMD files, rendering {len(selected_files)} of them")

    first_file = selected_files[0]
    first_iteration = parse_itnum(first_file)
    series = io.Series(first_file, io.Access.read_only)
    try:
        # Set the movie framing from the first selected output so all later
        # frames are rendered on a consistent box and camera.
        bbox = (
            compute_mesh_bbox(
                series,
                first_iteration,
                record_component=REC_COMP,
                mesh_name_re=REC_NAME_RE,
                pad_frac=BBOX_PAD_FRAC,
            )
            if args.auto_bbox or AUTO_BBOX
            else make_manual_bbox(args.bbox_half_width_cu)
        )
        print(f"Bounding box [km]: X{bbox[0]} Y{bbox[1]} Z{bbox[2]}")

        first_time_cu = get_time_code_units(series, first_iteration, record_component=REC_COMP)
        first_volume_cu, (xs, ys, zs) = composite_scalar_volume(
            series,
            first_iteration,
            bbox=bbox,
            shape=(args.grid_size, args.grid_size, args.grid_size),
            record_component=REC_COMP,
            mesh_name_re=REC_NAME_RE,
            edge_erode=EDGE_ERODE,
            chunk_z=Z_CHUNK,
            tile_xy=TILE_XY,
            downsample=DOWNSAMPLE,
        )
    finally:
        series.close()

    first_volume_cgs = sanitize_scalar_field(first_volume_cu, scale=RHO_CU_TO_CGS)
    floor_value, clim = compute_transfer_window(
        first_volume_cgs,
        args.pct_low,
        args.pct_high,
        args.dynamic_range_dex,
        args.max_bright_gap_dex,
    )
    floor_value, clim = apply_log_range_override(
        floor_value,
        clim,
        args.rho_log_min,
        args.rho_log_max,
    )
    grid = build_image_data(pv, build_log_volume(first_volume_cgs, floor_value), xs, ys, zs)
    contour_levels = build_contour_levels(
        clim,
        count=args.contour_count,
        low_frac=args.contour_low_frac,
    )

    plotter = pv.Plotter(off_screen=True, window_size=resolution)
    plotter.set_background(theme["background"])
    set_camera(plotter, xs, ys, zs)
    if args.camera_zoom != 1.0:
        plotter.camera.zoom(float(args.camera_zoom))
    plotter.add_axes(line_width=float(args.axes_line_width), color=theme["foreground"])
    contour = grid.contour(isosurfaces=contour_levels, scalars=SCALAR_NAME)
    contour_actor = plotter.add_mesh(
        contour,
        scalars=SCALAR_NAME,
        clim=clim,
        cmap=args.rho_cmap,
        opacity=args.contour_opacity,
        smooth_shading=True,
        ambient=0.25,
        diffuse=0.65,
        specular=0.10,
        show_scalar_bar=False,
        name="rho_contours",
    )
    if show_colorbar:
        plotter.add_scalar_bar(
            title=COLORBAR_TITLE,
            n_labels=5,
            fmt="%.1f",
            color=theme["foreground"],
            vertical=True,
            position_x=float(args.colorbar_x),
            position_y=float(args.colorbar_y),
            width=float(args.colorbar_width),
            height=float(args.colorbar_height),
            title_font_size=int(args.colorbar_title_size),
            label_font_size=int(args.colorbar_label_size),
            shadow=False,
        )

    rendered_frames = 0
    first_iteration_written = None
    last_iteration_written = None

    try:
        plotter.open_movie(str(movie_file), framerate=args.fps)

        for frame_index, series_file in enumerate(selected_files):
            iteration = parse_itnum(series_file)

            if frame_index == 0:
                time_cu = first_time_cu
                current_volume_cgs = first_volume_cgs
            else:
                series = io.Series(series_file, io.Access.read_only)
                try:
                    time_cu = get_time_code_units(series, iteration, record_component=REC_COMP)
                    if should_stop(time_cu, args.merger_time_ms, args.final_after_ms):
                        print(f"Stopping before iteration {iteration}: reached final-after-ms threshold")
                        break

                    volume_cu, _axes = composite_scalar_volume(
                        series,
                        iteration,
                        bbox=bbox,
                        shape=(args.grid_size, args.grid_size, args.grid_size),
                        record_component=REC_COMP,
                        mesh_name_re=REC_NAME_RE,
                        edge_erode=EDGE_ERODE,
                        chunk_z=Z_CHUNK,
                        tile_xy=TILE_XY,
                        downsample=DOWNSAMPLE,
                        out=current_volume_cgs,
                    )
                finally:
                    series.close()

                current_volume_cgs = sanitize_scalar_field(volume_cu, scale=RHO_CU_TO_CGS)
                # Reuse the same grid object, refresh the scalar field, and
                # rebuild only the contour mesh for the next frame.
                grid.point_data[SCALAR_NAME][:] = build_log_volume(
                    current_volume_cgs,
                    floor_value,
                ).ravel(order="F")
                contour = grid.contour(isosurfaces=contour_levels, scalars=SCALAR_NAME)
                plotter.remove_actor(contour_actor, render=False)
                contour_actor = plotter.add_mesh(
                    contour,
                    scalars=SCALAR_NAME,
                    clim=clim,
                    cmap=args.rho_cmap,
                    opacity=args.contour_opacity,
                    smooth_shading=True,
                    ambient=0.25,
                    diffuse=0.65,
                    specular=0.10,
                    show_scalar_bar=False,
                    name="rho_contours",
                    render=False,
                )

            time_ms = None if time_cu is None else time_cu * TIME_CU_TO_MS
            update_frame_label(
                plotter,
                time_ms,
                time_shift_ms=args.merger_time_ms,
                font_size=args.time_label_font_size,
                text_color=str(theme["foreground"]),
                text_shadow=bool(theme["text_shadow"]),
            )

            if args.spin_deg:
                plotter.camera.azimuth(args.spin_deg)
            if args.fan_deg:
                plotter.camera.elevation(args.fan_deg)
            plotter.render()
            if snapshot_dir is not None:
                snapshot_path = snapshot_dir / f"frame_{rendered_frames + 1:04d}_it{iteration:08d}.png"
                plotter.screenshot(str(snapshot_path))
            plotter.write_frame()
            gc.collect()
            rendered_frames += 1
            if first_iteration_written is None:
                first_iteration_written = iteration
            last_iteration_written = iteration
            print(f"[Frame {rendered_frames:03d}] it={iteration:08d}")
    finally:
        plotter.close()

    summary_path = out_dir / "README_summary.txt"
    write_summary(
        summary_path=summary_path,
        sim_name=sim_name,
        movie_file=movie_file,
        frame_count=rendered_frames,
        bbox=bbox,
        grid_size=args.grid_size,
        cadence=args.cadence,
        pyvista_source=pyvista_source,
        clim=clim,
        snapshot_dir=snapshot_dir,
        resolution=resolution,
        contour_count=args.contour_count,
        contour_low_frac=args.contour_low_frac,
        contour_opacity=args.contour_opacity,
    )

    print(f"Saved movie:   {movie_file}")
    if snapshot_dir is not None:
        print(f"Saved snaps:   {snapshot_dir}")
    print(f"Saved summary: {summary_path}")
    if rendered_frames:
        print(
            "First/last rendered iterations: "
            f"{first_iteration_written:08d} ... {last_iteration_written:08d}"
        )


if __name__ == "__main__":
    main()
