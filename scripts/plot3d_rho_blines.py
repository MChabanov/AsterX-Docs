#!/usr/bin/env python3
"""Render an off-screen 3D magnetic-field-line movie with PyVista.

This script uses a BNS-safe seeding strategy: it picks seeds from the strongest
magnetic-field voxels inside matter, instead of using older center-at-zero
angular weights that only make sense for a single star. It can also split the
strongest-voxel search across remnant and funnel regions so one movie can show
both the interior loops and any polar-field structure that is present.

Typical use from this directory:
    module load adios2
    python3 plot3d_blines.py

Quick smoke test:
    module load adios2
    python3 plot3d_blines.py --max-frames 1 --grid-size 96
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
    B_CU_TO_GAUSS,
    CU_TO_KM,
    DEFAULT_3D_THEME,
    DEFAULT_3D_B_SCALAR_BAR_X,
    DEFAULT_3D_RHO_SCALAR_BAR_X,
    DEFAULT_3D_SCALAR_BAR_HEIGHT,
    DEFAULT_3D_SCALAR_BAR_LABEL_SIZE,
    DEFAULT_3D_SCALAR_BAR_TITLE_SIZE,
    DEFAULT_3D_SCALAR_BAR_Y,
    DEFAULT_3D_RESOLUTION,
    DEFAULT_3D_SCALAR_BAR_WIDTH,
    DEFAULT_3D_SCALAR_BAR_X,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PYVISTA_ROOT,
    DEFAULT_SIM_DIR,
    RHO_CU_TO_CGS,
    TIME_CU_TO_MS,
    composite_scalar_volume,
    composite_vector_volume,
    compute_mesh_bbox,
    get_3d_theme,
    get_time_code_units,
    load_pyvista,
    parse_itnum,
    resolve_3d_paths,
    sanitize_scalar_field,
    sanitize_vector_field,
    select_series_files,
    vector_magnitude_scaled,
)
from plot3d_rho import (
    apply_log_range_override,
    build_contour_levels,
    build_log_volume,
    compute_transfer_window,
    set_camera,
    update_frame_label,
)


RENDER_NAME = "blines3d_pyvista"

B_RECORD = "hydrobasex_bvec"
B_COMPONENTS = ("hydrobasex_bvecx", "hydrobasex_bvecy", "hydrobasex_bvecz")
B_MESH_RE = re.compile(r"^hydrobasex_bvec_patch(\d+)_lev(\d+)$")

RHO_RECORD = "hydrobasex_rho"
RHO_MESH_RE = re.compile(r"^hydrobasex_rho_patch(\d+)_lev(\d+)$")
RHO_SCALAR_NAME = "log10_rho"
B_SCALAR_NAME = "log10_B"
B_VECTOR_NAME = "B"

# ----------------------- default run settings -----------------------
CADENCE = 4
GRID_SIZE = 256
EDGE_ERODE = 1
DOWNSAMPLE = 1
BBOX_HALF_WIDTH_CU = 140.0
BBOX_PAD_FRAC = 0.02
AUTO_BBOX = False

# Context rho contours.
RHO_PCT_LOW = 45.0
RHO_PCT_HIGH = 99.8
RHO_DYNAMIC_RANGE_DEX = 5.0
RHO_MAX_BRIGHT_GAP_DEX = 1.0
RHO_CONTOUR_COUNT = 6
RHO_CONTOUR_LOW_FRAC = 0.55
RHO_CONTOUR_OPACITY = 0.2
RHO_COLOR = "#b37d3f"

# Magnetic field-line coloring.
B_PCT_LOW = 50.0
B_PCT_HIGH = 99.9
B_DYNAMIC_RANGE_DEX = 3.0
B_MAX_BRIGHT_GAP_DEX = 1.0

# Seed selection: choose the strongest |B| voxels first, optionally enforcing
# a minimum spatial separation between accepted seeds.
SEED_MODE = "remnant"
SEED_STRENGTH_FRAC = 3.0e-3
SEED_RHO_MIN_CGS = 1.0e8
MAX_SEEDS = 256
MIN_SEED_SEPARATION_KM = 0.0
FUNNEL_THETA_MAX_DEG = 40.0
FUNNEL_RHO_MAX_CGS = 1.0e12
FUNNEL_MIN_ABS_Z_KM = 6.0
FUNNEL_AZIMUTH_BINS = 0

# Streamline integration and appearance.
STREAMLINE_MAX_LENGTH_KM = 220.0
STREAMLINE_INITIAL_STEP = 0.5
STREAMLINE_TERMINAL_SPEED = 1.0e-12
STREAMLINE_MAX_STEPS = 2000
STREAMLINE_BATCH_SIZE = 64
STREAMLINE_STYLE = "tube"
STREAMLINE_COLOR_MODE = "scalar"
STREAMLINE_SOLID_COLOR = "white"
STREAMLINE_LINE_WIDTH = 3.0
STREAMLINE_OPACITY = 1.0
STREAMLINE_TUBE_RADIUS_KM = 0.14 #0.35
STREAMLINE_TUBE_SIDES = 8
POST_MERGER_MODE = "jetlike"
JETLIKE_KEEP_CURVES = 96
JETLIKE_THETA_MAX_DEG = 20.0
JETLIKE_MIN_RADIUS_KM = 80.0
JETLIKE_MIN_ABS_Z_KM = 20.0
JETLIKE_START_DELAY_MS = 12.0
JETLIKE_MIN_KEEP_CURVES = 24
JETLIKE_MIN_CURVE_LENGTH_KM = 40.0
SHOW_COLORBAR = True
SHOW_RHO_COLORBAR = False
RHO_COLORBAR_TITLE = "log10 rho\n[g cm^-3]"
B_COLORBAR_TITLE = "log10 |B|\n[G]"
RHO_CMAP = "inferno"
B_CMAP = "viridis"

# Movie and camera controls. The default frame is narrower along x so the
# centered render spends less space on empty side margins.
FPS = 1
RESOLUTION = DEFAULT_3D_RESOLUTION
SPIN_DEG = 0.0
FAN_DEG = 0.0
CAMERA_ZOOM = 1.3
SAVE_SNAPSHOTS = True
MERGER_TIME_MS = 15.0
FINAL_AFTER_MS = None
TIME_LABEL_FONT_SIZE = 22
AXES_LINE_WIDTH = 3.0
# ---------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a 3D magnetic-field-line movie from openPMD outputs using PyVista."
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
        help="Uniform output grid size for both rho context and B lines.",
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
        "--show-rho-colorbar",
        action="store_true",
        help="Also show a rho colorbar alongside the magnetic-field colorbar.",
    )
    parser.add_argument(
        "--rho-cmap",
        default=RHO_CMAP,
        help="Colormap used for the rho context when scalar-colored, e.g. inferno, plasma, Blues.",
    )
    parser.add_argument(
        "--b-cmap",
        default=B_CMAP,
        help="Colormap used for the magnetic-field lines and B colorbar, e.g. viridis, plasma, Blues.",
    )
    parser.add_argument(
        "--no-b-colorbar",
        action="store_true",
        help="Hide the magnetic-field colorbar while keeping other colorbars enabled.",
    )
    parser.add_argument(
        "--rho-colorbar-x",
        type=float,
        default=DEFAULT_3D_RHO_SCALAR_BAR_X,
        help="Horizontal position of the rho colorbar in normalized viewport coordinates.",
    )
    parser.add_argument(
        "--b-colorbar-x",
        type=float,
        default=DEFAULT_3D_B_SCALAR_BAR_X,
        help="Horizontal position of the magnetic-field colorbar in normalized viewport coordinates.",
    )
    parser.add_argument(
        "--colorbar-y",
        type=float,
        default=DEFAULT_3D_SCALAR_BAR_Y,
        help="Shared fallback vertical position of the colorbars in normalized viewport coordinates.",
    )
    parser.add_argument(
        "--rho-colorbar-y",
        type=float,
        default=None,
        help="Vertical position of the rho colorbar in normalized viewport coordinates. Defaults to --colorbar-y.",
    )
    parser.add_argument(
        "--b-colorbar-y",
        type=float,
        default=None,
        help="Vertical position of the magnetic-field colorbar in normalized viewport coordinates. Defaults to --colorbar-y.",
    )
    parser.add_argument(
        "--colorbar-width",
        type=float,
        default=DEFAULT_3D_SCALAR_BAR_WIDTH,
        help="Normalized viewport width of each colorbar.",
    )
    parser.add_argument(
        "--colorbar-height",
        type=float,
        default=DEFAULT_3D_SCALAR_BAR_HEIGHT,
        help="Normalized viewport height of each colorbar.",
    )
    parser.add_argument(
        "--colorbar-title-size",
        type=int,
        default=DEFAULT_3D_SCALAR_BAR_TITLE_SIZE,
        help="Font size of the colorbar titles.",
    )
    parser.add_argument(
        "--colorbar-label-size",
        type=int,
        default=DEFAULT_3D_SCALAR_BAR_LABEL_SIZE,
        help="Font size of the colorbar tick labels.",
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
        help="Auto-fit the bbox from the first rho frame instead of using the symmetric manual box.",
    )
    parser.add_argument(
        "--rho-log-min",
        type=float,
        default=None,
        help="Override the lower end of the log10(rho [g cm^-3]) context range.",
    )
    parser.add_argument(
        "--rho-log-max",
        type=float,
        default=None,
        help="Override the upper end of the log10(rho [g cm^-3]) context range.",
    )
    parser.add_argument(
        "--contour-count",
        type=int,
        default=RHO_CONTOUR_COUNT,
        help="Number of rho context contours drawn behind the field lines.",
    )
    parser.add_argument(
        "--contour-low-frac",
        type=float,
        default=RHO_CONTOUR_LOW_FRAC,
        help="Lowest rho contour position inside the log10(rho) window.",
    )
    parser.add_argument(
        "--contour-opacity",
        type=float,
        default=RHO_CONTOUR_OPACITY,
        help="Opacity of the rho context contours.",
    )
    parser.add_argument(
        "--b-log-min",
        type=float,
        default=None,
        help="Override the lower end of the log10(|B| [G]) line-color range.",
    )
    parser.add_argument(
        "--b-log-max",
        type=float,
        default=None,
        help="Override the upper end of the log10(|B| [G]) line-color range.",
    )
    parser.add_argument(
        "--seed-mode",
        choices=("remnant", "funnel", "both"),
        default=SEED_MODE,
        help="Seed from strongest voxels in the dense remnant, the polar funnel, or both.",
    )
    parser.add_argument(
        "--seed-strength-frac",
        type=float,
        default=SEED_STRENGTH_FRAC,
        help="Keep only seed candidates with |B| above this fraction of the strongest seed.",
    )
    parser.add_argument(
        "--seed-rho-min-cgs",
        type=float,
        default=SEED_RHO_MIN_CGS,
        help="Minimum rho [g cm^-3] required for remnant-seed selection.",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=MAX_SEEDS,
        help="Maximum total number of seed points kept after ranking by |B|.",
    )
    parser.add_argument(
        "--remnant-seeds",
        type=int,
        default=None,
        help="Optional remnant-seed quota. In --seed-mode both, the remainder goes to funnel seeds.",
    )
    parser.add_argument(
        "--funnel-seeds",
        type=int,
        default=None,
        help="Optional funnel-seed quota. In --seed-mode both, the remainder goes to remnant seeds.",
    )
    parser.add_argument(
        "--min-seed-separation-km",
        type=float,
        default=MIN_SEED_SEPARATION_KM,
        help="Minimum allowed distance between accepted seed points in km.",
    )
    parser.add_argument(
        "--funnel-theta-max-deg",
        type=float,
        default=FUNNEL_THETA_MAX_DEG,
        help="Half-opening angle of the polar funnel cone measured from the +/-z axis.",
    )
    parser.add_argument(
        "--funnel-rho-max-cgs",
        type=float,
        default=FUNNEL_RHO_MAX_CGS,
        help="Maximum rho [g cm^-3] allowed for funnel-seed selection.",
    )
    parser.add_argument(
        "--funnel-min-abs-z-km",
        type=float,
        default=FUNNEL_MIN_ABS_Z_KM,
        help="Minimum |z| in km required for funnel-seed selection.",
    )
    parser.add_argument(
        "--funnel-azimuth-bins",
        type=int,
        default=FUNNEL_AZIMUTH_BINS,
        help="If positive, distribute funnel seeds across this many azimuth bins per hemisphere while still ranking by |B| within each bin.",
    )
    parser.add_argument(
        "--streamline-max-length-km",
        type=float,
        default=STREAMLINE_MAX_LENGTH_KM,
        help="Maximum streamline length in km.",
    )
    parser.add_argument(
        "--streamline-initial-step",
        type=float,
        default=STREAMLINE_INITIAL_STEP,
        help="Initial streamline integration step length.",
    )
    parser.add_argument(
        "--streamline-terminal-speed",
        type=float,
        default=STREAMLINE_TERMINAL_SPEED,
        help="Terminal speed threshold used by the streamline integrator.",
    )
    parser.add_argument(
        "--streamline-max-steps",
        type=int,
        default=STREAMLINE_MAX_STEPS,
        help="Maximum number of integration steps per streamline.",
    )
    parser.add_argument(
        "--streamline-batch-size",
        type=int,
        default=STREAMLINE_BATCH_SIZE,
        help="Number of seeds traced per VTK streamline batch. Lower values reduce peak memory.",
    )
    parser.add_argument(
        "--streamline-style",
        choices=("tube", "line"),
        default=STREAMLINE_STYLE,
        help="Render streamlines as shaded tubes or lightweight colored lines.",
    )
    parser.add_argument(
        "--streamline-color-mode",
        choices=("scalar", "solid"),
        default=STREAMLINE_COLOR_MODE,
        help="Color streamlines by log10|B| or draw them with one solid color.",
    )
    parser.add_argument(
        "--streamline-solid-color",
        default=STREAMLINE_SOLID_COLOR,
        help="Solid streamline color used when --streamline-color-mode solid is selected.",
    )
    parser.add_argument(
        "--streamline-line-width",
        type=float,
        default=STREAMLINE_LINE_WIDTH,
        help="Line width used when --streamline-style line is selected.",
    )
    parser.add_argument(
        "--streamline-opacity",
        type=float,
        default=STREAMLINE_OPACITY,
        help="Opacity of the magnetic-field streamlines for both line and tube rendering.",
    )
    parser.add_argument(
        "--streamline-tube-radius-km",
        type=float,
        default=STREAMLINE_TUBE_RADIUS_KM,
        help="Tube radius in km used when rendering the magnetic field lines.",
    )
    parser.add_argument(
        "--streamline-tube-sides",
        type=int,
        default=STREAMLINE_TUBE_SIDES,
        help="Number of polygon sides used for streamline tubes. Lower values save memory.",
    )
    parser.add_argument(
        "--no-colorbar",
        action="store_true",
        help="Disable the magnetic-field colorbar in snapshots and the movie.",
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
    parser.add_argument(
        "--post-merger-mode",
        choices=("current", "jetlike"),
        default=POST_MERGER_MODE,
        help="Rendering policy after merger time: keep current lines or filter to the most jet-like outer curves.",
    )
    parser.add_argument(
        "--jetlike-keep-curves",
        type=int,
        default=JETLIKE_KEEP_CURVES,
        help="Maximum number of post-merger jet-like curves kept after geometric ranking.",
    )
    parser.add_argument(
        "--jetlike-theta-max-deg",
        type=float,
        default=JETLIKE_THETA_MAX_DEG,
        help="Post-merger jet filter cone half-angle measured from the +/-z axis.",
    )
    parser.add_argument(
        "--jetlike-min-radius-km",
        type=float,
        default=JETLIKE_MIN_RADIUS_KM,
        help="Minimum spherical radius a curve must reach to qualify as jet-like post merger.",
    )
    parser.add_argument(
        "--jetlike-min-abs-z-km",
        type=float,
        default=JETLIKE_MIN_ABS_Z_KM,
        help="Minimum |z| a curve must reach to qualify as jet-like post merger.",
    )
    parser.add_argument(
        "--jetlike-start-delay-ms",
        type=float,
        default=JETLIKE_START_DELAY_MS,
        help="Delay after merger time before enabling the jetlike post-merger filter.",
    )
    parser.add_argument(
        "--jetlike-min-keep-curves",
        type=int,
        default=JETLIKE_MIN_KEEP_CURVES,
        help="If fewer than this many curves survive the jetlike filter, fall back to the current render for that frame.",
    )
    parser.add_argument(
        "--jetlike-min-curve-length-km",
        type=float,
        default=JETLIKE_MIN_CURVE_LENGTH_KM,
        help="Minimum total curve length required for a post-merger jetlike streamline to be kept.",
    )
    return parser.parse_args()


def make_manual_bbox(half_width_cu: float):
    half_width_km = float(half_width_cu) * CU_TO_KM
    return (
        (-half_width_km, half_width_km),
        (-half_width_km, half_width_km),
        (-half_width_km, half_width_km),
    )


def build_vector_image_data(
    pv,
    vectors: np.ndarray,
    logb: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
):
    nx, ny, nz = logb.shape
    grid = pv.ImageData(dimensions=(nx, ny, nz))
    dx = (xs[-1] - xs[0]) / (nx - 1) if nx > 1 else 1.0
    dy = (ys[-1] - ys[0]) / (ny - 1) if ny > 1 else 1.0
    dz = (zs[-1] - zs[0]) / (nz - 1) if nz > 1 else 1.0
    grid.spacing = (dx, dy, dz)
    grid.origin = (xs[0], ys[0], zs[0])
    point_vectors = np.empty((nx * ny * nz, 3), dtype=np.float32)
    point_vectors[:, 0] = vectors[..., 0].ravel(order="F")
    point_vectors[:, 1] = vectors[..., 1].ravel(order="F")
    point_vectors[:, 2] = vectors[..., 2].ravel(order="F")
    grid.point_data[B_VECTOR_NAME] = point_vectors
    grid.point_data[B_SCALAR_NAME] = logb.ravel(order="F")
    grid.set_active_vectors(B_VECTOR_NAME)
    return grid


def build_scalar_image_data(
    pv,
    data: np.ndarray,
    scalar_name: str,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
):
    nx, ny, nz = data.shape
    grid = pv.ImageData(dimensions=(nx, ny, nz))
    dx = (xs[-1] - xs[0]) / (nx - 1) if nx > 1 else 1.0
    dy = (ys[-1] - ys[0]) / (ny - 1) if ny > 1 else 1.0
    dz = (zs[-1] - zs[0]) / (nz - 1) if nz > 1 else 1.0
    grid.spacing = (dx, dy, dz)
    grid.origin = (xs[0], ys[0], zs[0])
    grid.point_data[scalar_name] = data.ravel(order="F")
    return grid


def build_streamlines_from_seed_points(
    pv,
    b_grid,
    seed_points: np.ndarray,
    batch_size: int,
    max_length: float,
    initial_step_length: float,
    terminal_speed: float,
    max_steps: int,
):
    """Trace streamlines in bounded batches to reduce late-frame VTK peak memory."""
    if seed_points.size == 0:
        return pv.PolyData()

    batch = max(1, int(batch_size))
    pieces = []

    for start in range(0, seed_points.shape[0], batch):
        seed_mesh = pv.PolyData(seed_points[start : start + batch])
        stream_piece = b_grid.streamlines_from_source(
            seed_mesh,
            vectors=B_VECTOR_NAME,
            integration_direction="both",
            max_length=max_length,
            initial_step_length=initial_step_length,
            terminal_speed=terminal_speed,
            max_steps=max_steps,
        )
        if stream_piece.n_cells > 0:
            pieces.append(stream_piece)

    if not pieces:
        return pv.PolyData()

    combined = pieces[0]
    for piece in pieces[1:]:
        combined = combined.merge(piece, merge_points=False)
    return combined


def add_streamline_actor(
    plotter,
    streamlines,
    style: str,
    color_mode: str,
    b_cmap: str,
    b_clim,
    solid_color: str,
    tube_radius_km: float,
    tube_sides: int,
    line_width: float,
    opacity: float,
    render: bool = False,
):
    """Add the current streamline geometry using either tubes or plain lines."""
    if style == "line":
        mesh_kwargs = dict(
            line_width=float(line_width),
            render_lines_as_tubes=False,
            lighting=False,
            show_scalar_bar=False,
            opacity=float(opacity),
            name="streamlines",
            render=render,
        )
        if color_mode == "solid":
            return plotter.add_mesh(
                streamlines,
                color=solid_color,
                **mesh_kwargs,
            )
        return plotter.add_mesh(
            streamlines,
            scalars=B_SCALAR_NAME,
            cmap=b_cmap,
            clim=b_clim,
            **mesh_kwargs,
        )

    stream_actor_mesh = streamlines.tube(
        radius=tube_radius_km,
        n_sides=max(3, int(tube_sides)),
    )
    mesh_kwargs = dict(
        smooth_shading=True,
        ambient=0.20,
        diffuse=0.60,
        specular=0.15,
        show_scalar_bar=False,
        opacity=float(opacity),
        name="streamlines",
        render=render,
    )
    if color_mode == "solid":
        actor = plotter.add_mesh(
            stream_actor_mesh,
            color=solid_color,
            **mesh_kwargs,
        )
        del stream_actor_mesh
        return actor
    actor = plotter.add_mesh(
        stream_actor_mesh,
        scalars=B_SCALAR_NAME,
        cmap=b_cmap,
        clim=b_clim,
        **mesh_kwargs,
    )
    del stream_actor_mesh
    return actor


def subset_polyline_cells(pv, polydata, selected_cell_ids):
    """Build a PolyData subset from selected polyline cell ids while preserving data arrays."""
    if not selected_cell_ids:
        return pv.PolyData()

    selected = set(int(cell_id) for cell_id in selected_cell_ids)
    lines = np.asarray(polydata.lines)
    points = np.asarray(polydata.points, dtype=np.float32)

    new_points = []
    new_lines = []
    new_point_ids = []
    kept_cell_ids = []
    old_to_new = {}

    cursor = 0
    cell_id = 0
    while cursor < lines.size:
        npts = int(lines[cursor])
        ids = lines[cursor + 1 : cursor + 1 + npts]
        cursor += npts + 1
        if cell_id not in selected:
            cell_id += 1
            continue

        kept_cell_ids.append(cell_id)
        remapped = []
        for old_id in ids:
            old_id = int(old_id)
            new_id = old_to_new.get(old_id)
            if new_id is None:
                new_id = len(new_points)
                old_to_new[old_id] = new_id
                new_points.append(points[old_id])
                new_point_ids.append(old_id)
            remapped.append(new_id)
        new_lines.extend([len(remapped), *remapped])
        cell_id += 1

    if not new_points:
        return pv.PolyData()

    subset = pv.PolyData(
        np.asarray(new_points, dtype=np.float32),
        lines=np.asarray(new_lines, dtype=np.int64),
    )

    point_ids = np.asarray(new_point_ids, dtype=np.int64)
    for name in polydata.point_data:
        subset.point_data[name] = np.asarray(polydata.point_data[name])[point_ids]

    cell_ids = np.asarray(kept_cell_ids, dtype=np.int64)
    for name in polydata.cell_data:
        subset.cell_data[name] = np.asarray(polydata.cell_data[name])[cell_ids]

    return subset


def filter_streamlines_post_merger(
    pv,
    streamlines,
    time_ms: float | None,
    merger_time_ms: float,
    mode: str,
    keep_curves: int,
    theta_max_deg: float,
    min_radius_km: float,
    min_abs_z_km: float,
    start_delay_ms: float,
    min_keep_curves: int,
    min_curve_length_km: float,
):
    """After merger, keep only the most outer-polar streamlines, inspired by the old jet plots."""
    total_curves = int(streamlines.n_cells)
    if (
        mode != "jetlike"
        or time_ms is None
        or time_ms < (merger_time_ms + max(0.0, float(start_delay_ms)))
        or total_curves <= 0
        or keep_curves <= 0
    ):
        return streamlines, {"mode": "current", "kept": total_curves, "total": total_curves}

    lines = np.asarray(streamlines.lines)
    points = np.asarray(streamlines.points, dtype=np.float32)
    if lines.size == 0 or points.size == 0:
        return streamlines, {"mode": "current", "kept": total_curves, "total": total_curves}

    theta_limit = None
    if float(theta_max_deg) < 90.0:
        theta_limit = float(np.tan(np.deg2rad(max(0.0, float(theta_max_deg)))))

    cursor = 0
    cell_id = 0
    ranked_pos = []
    ranked_neg = []
    while cursor < lines.size:
        npts = int(lines[cursor])
        ids = lines[cursor + 1 : cursor + 1 + npts]
        cursor += npts + 1
        if npts < 2:
            cell_id += 1
            continue

        pts = points[ids]
        rcyl = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        abs_z = np.abs(pts[:, 2])
        rsph = np.sqrt(rcyl ** 2 + pts[:, 2] ** 2)
        seg = np.diff(pts, axis=0)
        curve_length = float(np.sum(np.sqrt(np.sum(seg * seg, axis=1))))
        if curve_length < float(min_curve_length_km):
            cell_id += 1
            continue

        mask = (rsph >= float(min_radius_km)) & (abs_z >= float(min_abs_z_km))
        if theta_limit is not None:
            mask &= rcyl <= theta_limit * np.maximum(abs_z, 1.0e-6)
        if not np.any(mask):
            cell_id += 1
            continue

        cos_theta = abs_z[mask] / np.maximum(rsph[mask], 1.0e-6)
        score = float(np.max(abs_z[mask] * cos_theta)) * max(curve_length, 1.0)
        sign_ref = float(np.mean(pts[mask, 2]))
        if sign_ref >= 0.0:
            ranked_pos.append((score, cell_id))
        else:
            ranked_neg.append((score, cell_id))
        cell_id += 1

    ranked_all = ranked_pos + ranked_neg
    if not ranked_all:
        return streamlines, {"mode": "jetlike-fallback", "kept": total_curves, "total": total_curves}

    ranked_pos.sort(reverse=True)
    ranked_neg.sort(reverse=True)

    keep_total = max(1, int(keep_curves))
    keep_pos = min(len(ranked_pos), max(1, keep_total // 2))
    keep_neg = min(len(ranked_neg), max(1, keep_total - keep_pos))

    selected_ids = [cell_id for _score, cell_id in ranked_pos[:keep_pos]]
    selected_ids.extend(cell_id for _score, cell_id in ranked_neg[:keep_neg])

    if len(selected_ids) < keep_total:
        leftovers = ranked_pos[keep_pos:] + ranked_neg[keep_neg:]
        leftovers.sort(reverse=True)
        need = keep_total - len(selected_ids)
        selected_ids.extend(cell_id for _score, cell_id in leftovers[:need])

    filtered = subset_polyline_cells(pv, streamlines, selected_ids)
    kept_curves = int(filtered.n_cells) if filtered.n_cells > 0 else total_curves
    if filtered.n_cells <= 0:
        return streamlines, {"mode": "jetlike-fallback", "kept": total_curves, "total": total_curves}
    if kept_curves < max(1, int(min_keep_curves)):
        return streamlines, {"mode": "jetlike-sparse-fallback", "kept": total_curves, "total": total_curves}
    return filtered, {"mode": "jetlike", "kept": kept_curves, "total": total_curves}


def select_strong_seed_points(
    interest: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    seed_strength_frac: float,
    max_seeds: int,
    rho_cgs: np.ndarray | None = None,
    rho_min_cgs: float | None = None,
    rho_max_cgs: float | None = None,
    funnel_theta_max_deg: float | None = None,
    funnel_min_abs_z_km: float = 0.0,
    azimuth_bins: int = 0,
    min_seed_separation_km: float = 0.0,
    allow_empty: bool = False,
    region_name: str = "selected region",
):
    """Pick the strongest magnetic-field voxels as streamline seeds.

    The older VTK scripts used angular weighting centered at the origin after
    extracting many lines. That biases BNS data badly at early times because
    the field is anchored in two stars, not one object at x=y=z=0. Here we
    seed directly from the strongest local field inside matter and optionally
    enforce a minimum spatial separation between accepted seeds.
    """
    data = np.asarray(interest, dtype=np.float32)
    valid = np.isfinite(data)
    valid &= data > 0.0

    if rho_cgs is not None:
        rho_data = np.asarray(rho_cgs, dtype=np.float32)
        if rho_min_cgs is not None:
            valid &= rho_data >= np.float32(rho_min_cgs)
        if rho_max_cgs is not None:
            valid &= rho_data <= np.float32(rho_max_cgs)

    if funnel_theta_max_deg is not None or funnel_min_abs_z_km > 0.0:
        xs32 = np.asarray(xs, dtype=np.float32)
        ys32 = np.asarray(ys, dtype=np.float32)
        zs32 = np.asarray(zs, dtype=np.float32)
        xy_sq = xs32[:, None] * xs32[:, None] + ys32[None, :] * ys32[None, :]
        if funnel_theta_max_deg is None or float(funnel_theta_max_deg) >= 90.0:
            tan_sq = None
        else:
            theta_rad = np.deg2rad(max(0.0, float(funnel_theta_max_deg)))
            tan_sq = float(np.tan(theta_rad) ** 2)
        for k, z_val in enumerate(zs32):
            valid_slice = valid[:, :, k]
            if not np.any(valid_slice):
                continue
            abs_z = abs(float(z_val))
            if abs_z < float(funnel_min_abs_z_km):
                valid_slice.fill(False)
                continue
            if tan_sq is not None:
                valid_slice &= xy_sq <= (tan_sq * abs_z * abs_z)

    if not np.any(valid):
        if allow_empty:
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
        raise RuntimeError(f"No valid magnetic-field seeds found in the {region_name}.")

    peak = None
    for k in range(valid.shape[2]):
        valid_slice = valid[:, :, k]
        if not np.any(valid_slice):
            continue
        local_peak = float(np.max(data[:, :, k][valid_slice]))
        peak = local_peak if peak is None else max(peak, local_peak)

    if peak is None or peak <= 0.0:
        if allow_empty:
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
        raise RuntimeError(f"No positive magnetic-field seeds found in the {region_name}.")

    threshold = peak * max(0.0, float(seed_strength_frac))
    limit = max(1, int(max_seeds))
    # A very low seed-strength threshold can admit millions of voxels. Keep a
    # bounded strongest-candidate pool per slice and globally before sorting so
    # the selector stays memory-safe on login nodes.
    slice_pool_limit = min(data.shape[0] * data.shape[1], max(limit * 64, 1024))
    global_pool_limit = max(limit * 256, 4096)
    xs32 = np.asarray(xs, dtype=np.float32)
    ys32 = np.asarray(ys, dtype=np.float32)
    zs32 = np.asarray(zs, dtype=np.float32)
    candidate_points = []
    candidate_strengths = []

    for k, z_val in enumerate(zs32):
        valid_slice = valid[:, :, k]
        if not np.any(valid_slice):
            continue
        strong_slice = valid_slice & (data[:, :, k] >= threshold)
        if not np.any(strong_slice):
            continue
        strengths_slice = data[:, :, k][strong_slice].astype(np.float32, copy=False)
        ij = np.argwhere(strong_slice)
        if strengths_slice.size > slice_pool_limit:
            top_idx = np.argpartition(strengths_slice, -slice_pool_limit)[-slice_pool_limit:]
            strengths_slice = strengths_slice[top_idx]
            ij = ij[top_idx]
        points = np.empty((ij.shape[0], 3), dtype=np.float32)
        points[:, 0] = xs32[ij[:, 0]]
        points[:, 1] = ys32[ij[:, 1]]
        points[:, 2] = z_val
        candidate_points.append(points)
        candidate_strengths.append(strengths_slice)

    if not candidate_points:
        if allow_empty:
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
        raise RuntimeError(f"No thresholded magnetic-field seeds survived in the {region_name}.")

    points = np.concatenate(candidate_points, axis=0)
    strengths = np.concatenate(candidate_strengths, axis=0)
    if strengths.size > global_pool_limit:
        top_idx = np.argpartition(strengths, -global_pool_limit)[-global_pool_limit:]
        points = points[top_idx]
        strengths = strengths[top_idx]
    if int(azimuth_bins) > 1 and points.shape[0] > int(azimuth_bins):
        phi = np.arctan2(points[:, 1], points[:, 0])
        phi_bin = np.floor((phi + np.pi) * (float(azimuth_bins) / (2.0 * np.pi))).astype(np.int32)
        phi_bin = np.clip(phi_bin, 0, int(azimuth_bins) - 1)
        hemi = (points[:, 2] >= 0.0).astype(np.int32)
        grouped = {}
        for idx in range(points.shape[0]):
            key = (int(hemi[idx]), int(phi_bin[idx]))
            grouped.setdefault(key, []).append(idx)
        for key in grouped:
            bucket = np.asarray(grouped[key], dtype=np.int32)
            bucket_strengths = strengths[bucket]
            grouped[key] = bucket[np.argsort(bucket_strengths)[::-1]]

        order_chunks = []
        max_bucket = max((bucket.size for bucket in grouped.values()), default=0)
        for depth in range(max_bucket):
            layer = []
            for hemi_id in (1, 0):
                for bin_id in range(int(azimuth_bins)):
                    bucket = grouped.get((hemi_id, bin_id))
                    if bucket is not None and depth < bucket.size:
                        layer.append(int(bucket[depth]))
            if layer:
                layer = np.asarray(layer, dtype=np.int32)
                layer_strengths = strengths[layer]
                layer = layer[np.argsort(layer_strengths)[::-1]]
                order_chunks.append(layer)
        order = np.concatenate(order_chunks) if order_chunks else np.argsort(strengths)[::-1]
    else:
        order = np.argsort(strengths)[::-1]

    if min_seed_separation_km <= 0.0:
        order = order[:limit]
        return points[order], strengths[order]

    selected_points = []
    selected_strengths = []
    min_sep_sq = float(min_seed_separation_km) ** 2

    for index in order:
        point = points[index]
        if all(np.sum((point - other) ** 2) >= min_sep_sq for other in selected_points):
            selected_points.append(point)
            selected_strengths.append(strengths[index])
        if len(selected_points) >= limit:
            break

    if not selected_points:
        best = int(order[0])
        selected_points.append(points[best])
        selected_strengths.append(strengths[best])

    return np.asarray(selected_points, dtype=np.float32), np.asarray(selected_strengths, dtype=np.float32)


def resolve_seed_pool_quotas(
    seed_mode: str,
    max_seeds: int,
    remnant_seeds: int | None,
    funnel_seeds: int | None,
) -> tuple[int, int]:
    total = max(1, int(max_seeds))

    if seed_mode == "remnant":
        return total, 0
    if seed_mode == "funnel":
        return 0, total

    remnant = None if remnant_seeds is None else max(0, int(remnant_seeds))
    funnel = None if funnel_seeds is None else max(0, int(funnel_seeds))

    if remnant is None and funnel is None:
        remnant = total // 2
        funnel = total - remnant
    elif remnant is None:
        funnel = min(total, funnel)
        remnant = total - funnel
    elif funnel is None:
        remnant = min(total, remnant)
        funnel = total - remnant
    elif remnant + funnel > total:
        raise ValueError(
            f"Requested remnant_seeds + funnel_seeds = {remnant + funnel}, "
            f"which exceeds --max-seeds = {total}."
        )

    if remnant <= 0 and funnel <= 0:
        raise ValueError("At least one of remnant or funnel seed quotas must be positive.")
    return remnant, funnel


def merge_seed_groups(seed_groups):
    merged = {}

    for label, points, strengths in seed_groups:
        for point, strength in zip(points, strengths):
            key = (float(point[0]), float(point[1]), float(point[2]))
            previous = merged.get(key)
            if previous is None or float(strength) > previous[1]:
                merged[key] = (np.asarray(point, dtype=np.float32), float(strength), label)

    if not merged:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            {"remnant": 0, "funnel": 0},
        )

    ranked = sorted(merged.values(), key=lambda item: item[1], reverse=True)
    points = np.asarray([item[0] for item in ranked], dtype=np.float32)
    strengths = np.asarray([item[1] for item in ranked], dtype=np.float32)
    counts = {"remnant": 0, "funnel": 0}
    for _point, _strength, label in ranked:
        counts[label] += 1
    return points, strengths, counts


def select_seed_points_by_mode(
    interest: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    seed_mode: str,
    seed_strength_frac: float,
    max_seeds: int,
    rho_cgs: np.ndarray,
    remnant_rho_min_cgs: float,
    remnant_seeds: int | None,
    funnel_seeds: int | None,
    funnel_theta_max_deg: float,
    funnel_rho_max_cgs: float | None,
    funnel_min_abs_z_km: float,
    funnel_azimuth_bins: int,
    min_seed_separation_km: float,
):
    remnant_quota, funnel_quota = resolve_seed_pool_quotas(
        seed_mode,
        max_seeds=max_seeds,
        remnant_seeds=remnant_seeds,
        funnel_seeds=funnel_seeds,
    )

    seed_groups = []

    if remnant_quota > 0:
        remnant_points, remnant_strengths = select_strong_seed_points(
            interest,
            xs,
            ys,
            zs,
            seed_strength_frac=seed_strength_frac,
            max_seeds=remnant_quota,
            rho_cgs=rho_cgs,
            rho_min_cgs=remnant_rho_min_cgs,
            min_seed_separation_km=min_seed_separation_km,
            allow_empty=(seed_mode == "both"),
            region_name="remnant",
        )
        if remnant_points.size:
            seed_groups.append(("remnant", remnant_points, remnant_strengths))

    if funnel_quota > 0:
        funnel_points, funnel_strengths = select_strong_seed_points(
            interest,
            xs,
            ys,
            zs,
            seed_strength_frac=seed_strength_frac,
            max_seeds=funnel_quota,
            rho_cgs=rho_cgs,
            rho_max_cgs=funnel_rho_max_cgs,
            funnel_theta_max_deg=funnel_theta_max_deg,
            funnel_min_abs_z_km=funnel_min_abs_z_km,
            azimuth_bins=funnel_azimuth_bins,
            min_seed_separation_km=min_seed_separation_km,
            allow_empty=(seed_mode == "both"),
            region_name="funnel",
        )
        if funnel_points.size:
            seed_groups.append(("funnel", funnel_points, funnel_strengths))

    if not seed_groups:
        raise RuntimeError(
            "No seed points survived the requested seed mode and thresholds. "
            "Relax the remnant/funnel masks or lower the seed thresholds."
        )

    return merge_seed_groups(seed_groups)


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
    rho_clim,
    b_clim,
    snapshot_dir: Path | None,
    resolution,
    seed_mode: str,
    seed_strength_frac: float,
    max_seeds: int,
    remnant_quota: int,
    funnel_quota: int,
    funnel_theta_max_deg: float,
    funnel_rho_max_cgs: float | None,
    funnel_min_abs_z_km: float,
    post_merger_mode: str,
    jetlike_keep_curves: int,
    jetlike_theta_max_deg: float,
    jetlike_min_radius_km: float,
    jetlike_min_abs_z_km: float,
    jetlike_start_delay_ms: float,
    jetlike_min_keep_curves: int,
    jetlike_min_curve_length_km: float,
    min_seed_separation_km: float,
    streamline_style: str,
    streamline_batch_size: int,
    streamline_opacity: float,
    streamline_tube_radius_km: float,
    streamline_tube_sides: int,
    contour_count: int,
    contour_low_frac: float,
    contour_opacity: float,
    seed_count_range,
    remnant_seed_count_range,
    funnel_seed_count_range,
    streamline_count_range,
) -> None:
    with summary_path.open("w", encoding="utf-8") as stream:
        stream.write("PyVista 3D magnetic-field-line movie\n")
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
            "Seeds: "
            f"mode={seed_mode}, strength_frac={seed_strength_frac:.4f}, max_seeds={max_seeds}, "
            f"min_separation_km={min_seed_separation_km:.3f}\n"
        )
        stream.write(f"Seed quotas: remnant={remnant_quota}, funnel={funnel_quota}\n")
        stream.write(
            "Funnel mask: "
            f"theta_max_deg={funnel_theta_max_deg:.2f}, "
            f"rho_max_cgs={funnel_rho_max_cgs}, "
            f"min_abs_z_km={funnel_min_abs_z_km:.3f}\n"
        )
        stream.write(
            "Post-merger mode: "
            f"{post_merger_mode}, keep_curves={jetlike_keep_curves}, "
            f"theta_max_deg={jetlike_theta_max_deg:.2f}, "
            f"min_radius_km={jetlike_min_radius_km:.3f}, "
            f"min_abs_z_km={jetlike_min_abs_z_km:.3f}, "
            f"start_delay_ms={jetlike_start_delay_ms:.3f}, "
            f"min_keep_curves={jetlike_min_keep_curves}, "
            f"min_curve_length_km={jetlike_min_curve_length_km:.3f}\n"
        )
        stream.write(
            "Rho context contours: "
            f"count={contour_count}, low_frac={contour_low_frac:.3f}, "
            f"opacity={contour_opacity:.3f}\n"
        )
        stream.write(
            "Accepted seeds per frame: "
            f"{seed_count_range[0]} .. {seed_count_range[1]}\n"
        )
        stream.write(
            "Accepted remnant seeds per frame: "
            f"{remnant_seed_count_range[0]} .. {remnant_seed_count_range[1]}\n"
        )
        stream.write(
            "Accepted funnel seeds per frame: "
            f"{funnel_seed_count_range[0]} .. {funnel_seed_count_range[1]}\n"
        )
        stream.write(
            "Streamlines per frame: "
            f"{streamline_count_range[0]} .. {streamline_count_range[1]}\n"
        )
        stream.write(
            f"Streamline style: {streamline_style}, batch_size={streamline_batch_size}, "
            f"opacity={streamline_opacity:.3f}\n"
        )
        stream.write(f"Tube radius [km]: {streamline_tube_radius_km:.3f}\n")
        stream.write(f"Tube sides: {streamline_tube_sides}\n")
        stream.write(f"log10(rho) context clim: ({rho_clim[0]:.6f}, {rho_clim[1]:.6f})\n")
        stream.write(f"log10(|B|) line clim: ({b_clim[0]:.6f}, {b_clim[1]:.6f})\n")


def main() -> None:
    args = parse_args()
    pv, pyvista_source = load_pyvista(args.pyvista_root)
    theme = get_3d_theme(args.theme)
    save_snapshots = SAVE_SNAPSHOTS and not args.no_snapshots
    show_b_colorbar = (
        SHOW_COLORBAR
        and not args.no_colorbar
        and not args.no_b_colorbar
        and args.streamline_color_mode == "scalar"
    )
    show_rho_colorbar = (SHOW_RHO_COLORBAR or args.show_rho_colorbar) and not args.no_colorbar
    resolution = (int(args.width), int(args.height))

    if float(args.streamline_terminal_speed) > 1.0e-8:
        print(
            "Warning: --streamline-terminal-speed is high for weak-field tails; "
            "values like 1e-10 to 1e-12 keep long outer lines more reliably."
        )

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
    remnant_quota, funnel_quota = resolve_seed_pool_quotas(
        args.seed_mode,
        max_seeds=args.max_seeds,
        remnant_seeds=args.remnant_seeds,
        funnel_seeds=args.funnel_seeds,
    )

    first_file = selected_files[0]
    first_iteration = parse_itnum(first_file)
    series = io.Series(first_file, io.Access.read_only)
    try:
        bbox = (
            compute_mesh_bbox(
                series,
                first_iteration,
                record_component=RHO_RECORD,
                mesh_name_re=RHO_MESH_RE,
                pad_frac=BBOX_PAD_FRAC,
            )
            if args.auto_bbox or AUTO_BBOX
            else make_manual_bbox(args.bbox_half_width_cu)
        )
        print(f"Bounding box [km]: X{bbox[0]} Y{bbox[1]} Z{bbox[2]}")

        first_time_cu = get_time_code_units(series, first_iteration, record_component=RHO_RECORD)
        rho_cu, (xs, ys, zs) = composite_scalar_volume(
            series,
            first_iteration,
            bbox=bbox,
            shape=(args.grid_size, args.grid_size, args.grid_size),
            record_component=RHO_RECORD,
            mesh_name_re=RHO_MESH_RE,
            edge_erode=EDGE_ERODE,
            downsample=DOWNSAMPLE,
        )
        bvec_cu, _axes = composite_vector_volume(
            series,
            first_iteration,
            bbox=bbox,
            shape=(args.grid_size, args.grid_size, args.grid_size),
            record_component=B_RECORD,
            component_names=B_COMPONENTS,
            mesh_name_re=B_MESH_RE,
            edge_erode=EDGE_ERODE,
            downsample=DOWNSAMPLE,
        )
    finally:
        series.close()

    rho_cgs = sanitize_scalar_field(rho_cu, scale=RHO_CU_TO_CGS)
    bvec_cu = sanitize_vector_field(bvec_cu)
    bmag_gauss = vector_magnitude_scaled(bvec_cu, scale=B_CU_TO_GAUSS)

    rho_floor, rho_clim = compute_transfer_window(
        rho_cgs,
        RHO_PCT_LOW,
        RHO_PCT_HIGH,
        RHO_DYNAMIC_RANGE_DEX,
        RHO_MAX_BRIGHT_GAP_DEX,
    )
    rho_floor, rho_clim = apply_log_range_override(
        rho_floor,
        rho_clim,
        args.rho_log_min,
        args.rho_log_max,
    )

    b_floor, b_clim = compute_transfer_window(
        bmag_gauss,
        B_PCT_LOW,
        B_PCT_HIGH,
        B_DYNAMIC_RANGE_DEX,
        B_MAX_BRIGHT_GAP_DEX,
    )
    b_floor, b_clim = apply_log_range_override(
        b_floor,
        b_clim,
        args.b_log_min,
        args.b_log_max,
    )

    rho_grid = build_scalar_image_data(
        pv,
        build_log_volume(rho_cgs, rho_floor),
        RHO_SCALAR_NAME,
        xs,
        ys,
        zs,
    )
    rho_contour = rho_grid.contour(
        isosurfaces=build_contour_levels(
            rho_clim,
            count=args.contour_count,
            low_frac=args.contour_low_frac,
        ),
        scalars=RHO_SCALAR_NAME,
    )

    b_log = np.log10(np.clip(bmag_gauss, b_floor, None)).astype(np.float32)
    b_grid = build_vector_image_data(pv, bvec_cu.astype(np.float32), b_log, xs, ys, zs)

    seed_points, _seed_strengths, seed_breakdown = select_seed_points_by_mode(
        bmag_gauss,
        xs,
        ys,
        zs,
        seed_mode=args.seed_mode,
        seed_strength_frac=args.seed_strength_frac,
        max_seeds=args.max_seeds,
        rho_cgs=rho_cgs,
        remnant_rho_min_cgs=args.seed_rho_min_cgs,
        remnant_seeds=args.remnant_seeds,
        funnel_seeds=args.funnel_seeds,
        funnel_theta_max_deg=args.funnel_theta_max_deg,
        funnel_rho_max_cgs=args.funnel_rho_max_cgs,
        funnel_min_abs_z_km=args.funnel_min_abs_z_km,
        funnel_azimuth_bins=args.funnel_azimuth_bins,
        min_seed_separation_km=args.min_seed_separation_km,
    )
    first_time_ms = None if first_time_cu is None else first_time_cu * TIME_CU_TO_MS
    streamlines = build_streamlines_from_seed_points(
        pv,
        b_grid,
        seed_points,
        batch_size=args.streamline_batch_size,
        max_length=args.streamline_max_length_km,
        initial_step_length=args.streamline_initial_step,
        terminal_speed=args.streamline_terminal_speed,
        max_steps=args.streamline_max_steps,
    )
    streamlines, streamline_filter_info = filter_streamlines_post_merger(
        pv,
        streamlines,
        time_ms=first_time_ms,
        merger_time_ms=args.merger_time_ms,
        mode=args.post_merger_mode,
        keep_curves=args.jetlike_keep_curves,
        theta_max_deg=args.jetlike_theta_max_deg,
        min_radius_km=args.jetlike_min_radius_km,
        min_abs_z_km=args.jetlike_min_abs_z_km,
        start_delay_ms=args.jetlike_start_delay_ms,
        min_keep_curves=args.jetlike_min_keep_curves,
        min_curve_length_km=args.jetlike_min_curve_length_km,
    )

    plotter = pv.Plotter(off_screen=True, window_size=resolution)
    plotter.set_background(theme["background"])
    set_camera(plotter, xs, ys, zs)
    if args.camera_zoom != 1.0:
        plotter.camera.zoom(float(args.camera_zoom))
    plotter.add_axes(line_width=float(args.axes_line_width), color=theme["foreground"])
    rho_mesh_kwargs = dict(
        opacity=args.contour_opacity,
        smooth_shading=True,
        ambient=0.20,
        diffuse=0.60,
        specular=0.08,
        show_scalar_bar=False,
        name="rho_context",
    )
    if show_rho_colorbar:
        contour_actor = plotter.add_mesh(
            rho_contour,
            scalars=RHO_SCALAR_NAME,
            cmap=args.rho_cmap,
            clim=rho_clim,
            **rho_mesh_kwargs,
        )
    else:
        contour_actor = plotter.add_mesh(
            rho_contour,
            color=RHO_COLOR,
            **rho_mesh_kwargs,
        )
    streamline_actor = add_streamline_actor(
        plotter,
        streamlines,
        style=args.streamline_style,
        color_mode=args.streamline_color_mode,
        b_cmap=args.b_cmap,
        b_clim=b_clim,
        solid_color=args.streamline_solid_color,
        tube_radius_km=args.streamline_tube_radius_km,
        tube_sides=args.streamline_tube_sides,
        line_width=args.streamline_line_width,
        opacity=args.streamline_opacity,
        render=False,
    )
    if show_rho_colorbar:
        plotter.add_scalar_bar(
            title=RHO_COLORBAR_TITLE,
            mapper=contour_actor.mapper,
            n_labels=5,
            fmt="%.1f",
            color=theme["foreground"],
            vertical=True,
            position_x=float(args.rho_colorbar_x),
            position_y=float(args.rho_colorbar_y if args.rho_colorbar_y is not None else args.colorbar_y),
            width=float(args.colorbar_width),
            height=float(args.colorbar_height),
            title_font_size=int(args.colorbar_title_size),
            label_font_size=int(args.colorbar_label_size),
            shadow=False,
        )
    if show_b_colorbar:
        plotter.add_scalar_bar(
            title=B_COLORBAR_TITLE,
            mapper=streamline_actor.mapper,
            n_labels=5,
            fmt="%.1f",
            color=theme["foreground"],
            vertical=True,
            position_x=float(args.b_colorbar_x),
            position_y=float(args.b_colorbar_y if args.b_colorbar_y is not None else args.colorbar_y),
            width=float(args.colorbar_width),
            height=float(args.colorbar_height),
            title_font_size=int(args.colorbar_title_size),
            label_font_size=int(args.colorbar_label_size),
            shadow=False,
        )

    rendered_frames = 0
    first_iteration_written = None
    last_iteration_written = None
    seed_counts = []
    remnant_seed_counts = []
    funnel_seed_counts = []
    streamline_counts = []

    try:
        plotter.open_movie(str(movie_file), framerate=args.fps)

        for frame_index, series_file in enumerate(selected_files):
            iteration = parse_itnum(series_file)

            if frame_index == 0:
                time_cu = first_time_cu
                current_rho_cgs = rho_cgs
                current_bvec_cu = bvec_cu
                current_bmag_gauss = bmag_gauss
            else:
                series = io.Series(series_file, io.Access.read_only)
                try:
                    time_cu = get_time_code_units(series, iteration, record_component=RHO_RECORD)
                    if should_stop(time_cu, args.merger_time_ms, args.final_after_ms):
                        print(f"Stopping before iteration {iteration}: reached final-after-ms threshold")
                        break

                    rho_cu_frame, _axes = composite_scalar_volume(
                        series,
                        iteration,
                        bbox=bbox,
                        shape=(args.grid_size, args.grid_size, args.grid_size),
                        record_component=RHO_RECORD,
                        mesh_name_re=RHO_MESH_RE,
                        edge_erode=EDGE_ERODE,
                        downsample=DOWNSAMPLE,
                        out=current_rho_cgs,
                    )
                    bvec_cu_frame, _axes = composite_vector_volume(
                        series,
                        iteration,
                        bbox=bbox,
                        shape=(args.grid_size, args.grid_size, args.grid_size),
                        record_component=B_RECORD,
                        component_names=B_COMPONENTS,
                        mesh_name_re=B_MESH_RE,
                        edge_erode=EDGE_ERODE,
                        downsample=DOWNSAMPLE,
                        out=current_bvec_cu,
                    )
                finally:
                    series.close()

                current_rho_cgs = sanitize_scalar_field(rho_cu_frame, scale=RHO_CU_TO_CGS)
                current_bvec_cu = sanitize_vector_field(bvec_cu_frame)
                current_bmag_gauss = vector_magnitude_scaled(
                    current_bvec_cu,
                    scale=B_CU_TO_GAUSS,
                    out=current_bmag_gauss,
                )

                rho_grid.point_data[RHO_SCALAR_NAME][:] = build_log_volume(
                    current_rho_cgs,
                    rho_floor,
                ).ravel(order="F")
                rho_contour = rho_grid.contour(
                    isosurfaces=build_contour_levels(
                        rho_clim,
                        count=args.contour_count,
                        low_frac=args.contour_low_frac,
                    ),
                    scalars=RHO_SCALAR_NAME,
                )
                plotter.remove_actor(contour_actor, render=False)
                if show_rho_colorbar:
                    contour_actor = plotter.add_mesh(
                        rho_contour,
                        scalars=RHO_SCALAR_NAME,
                        cmap=args.rho_cmap,
                        clim=rho_clim,
                        opacity=args.contour_opacity,
                        smooth_shading=True,
                        ambient=0.20,
                        diffuse=0.60,
                        specular=0.08,
                        show_scalar_bar=False,
                        name="rho_context",
                        render=False,
                    )
                else:
                    contour_actor = plotter.add_mesh(
                        rho_contour,
                        color=RHO_COLOR,
                        opacity=args.contour_opacity,
                        smooth_shading=True,
                        ambient=0.20,
                        diffuse=0.60,
                        specular=0.08,
                        show_scalar_bar=False,
                        name="rho_context",
                        render=False,
                    )

                b_point_vectors = b_grid.point_data[B_VECTOR_NAME]
                b_point_vectors[:, 0] = current_bvec_cu[..., 0].ravel(order="F")
                b_point_vectors[:, 1] = current_bvec_cu[..., 1].ravel(order="F")
                b_point_vectors[:, 2] = current_bvec_cu[..., 2].ravel(order="F")
                b_grid.point_data[B_SCALAR_NAME][:] = np.log10(
                    np.clip(current_bmag_gauss, b_floor, None)
                ).astype(np.float32).ravel(order="F")

                if streamline_actor is not None:
                    plotter.remove_actor(streamline_actor, render=False)
                    streamline_actor = None

                seed_points, _seed_strengths, seed_breakdown = select_seed_points_by_mode(
                    current_bmag_gauss,
                    xs,
                    ys,
                    zs,
                    seed_mode=args.seed_mode,
                    seed_strength_frac=args.seed_strength_frac,
                    max_seeds=args.max_seeds,
                    rho_cgs=current_rho_cgs,
                    remnant_rho_min_cgs=args.seed_rho_min_cgs,
                    remnant_seeds=args.remnant_seeds,
                    funnel_seeds=args.funnel_seeds,
                    funnel_theta_max_deg=args.funnel_theta_max_deg,
                    funnel_rho_max_cgs=args.funnel_rho_max_cgs,
                    funnel_min_abs_z_km=args.funnel_min_abs_z_km,
                    funnel_azimuth_bins=args.funnel_azimuth_bins,
                    min_seed_separation_km=args.min_seed_separation_km,
                )
                time_ms = None if time_cu is None else time_cu * TIME_CU_TO_MS
                streamlines = build_streamlines_from_seed_points(
                    pv,
                    b_grid,
                    seed_points,
                    batch_size=args.streamline_batch_size,
                    max_length=args.streamline_max_length_km,
                    initial_step_length=args.streamline_initial_step,
                    terminal_speed=args.streamline_terminal_speed,
                    max_steps=args.streamline_max_steps,
                )
                streamlines, streamline_filter_info = filter_streamlines_post_merger(
                    pv,
                    streamlines,
                    time_ms=time_ms,
                    merger_time_ms=args.merger_time_ms,
                    mode=args.post_merger_mode,
                    keep_curves=args.jetlike_keep_curves,
                    theta_max_deg=args.jetlike_theta_max_deg,
                    min_radius_km=args.jetlike_min_radius_km,
                    min_abs_z_km=args.jetlike_min_abs_z_km,
                    start_delay_ms=args.jetlike_start_delay_ms,
                    min_keep_curves=args.jetlike_min_keep_curves,
                    min_curve_length_km=args.jetlike_min_curve_length_km,
                )
                streamline_actor = add_streamline_actor(
                    plotter,
                    streamlines,
                    style=args.streamline_style,
                    color_mode=args.streamline_color_mode,
                    b_cmap=args.b_cmap,
                    b_clim=b_clim,
                    solid_color=args.streamline_solid_color,
                    tube_radius_km=args.streamline_tube_radius_km,
                    tube_sides=args.streamline_tube_sides,
                    line_width=args.streamline_line_width,
                    opacity=args.streamline_opacity,
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
            current_seed_count = int(seed_points.shape[0])
            current_streamline_count = int(streamlines.n_cells)
            seed_counts.append(current_seed_count)
            remnant_seed_counts.append(int(seed_breakdown["remnant"]))
            funnel_seed_counts.append(int(seed_breakdown["funnel"]))
            streamline_counts.append(current_streamline_count)

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
            print(
                f"[Frame {rendered_frames:03d}] it={iteration:08d} "
                f"seeds={current_seed_count} "
                f"(remnant={seed_breakdown['remnant']}, funnel={seed_breakdown['funnel']}) "
                f"lines={current_streamline_count}"
                f"/{streamline_filter_info['total']} mode={streamline_filter_info['mode']}"
            )
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
        rho_clim=rho_clim,
        b_clim=b_clim,
        snapshot_dir=snapshot_dir,
        resolution=resolution,
        seed_mode=args.seed_mode,
        seed_strength_frac=args.seed_strength_frac,
        max_seeds=args.max_seeds,
        remnant_quota=remnant_quota,
        funnel_quota=funnel_quota,
        funnel_theta_max_deg=args.funnel_theta_max_deg,
        funnel_rho_max_cgs=args.funnel_rho_max_cgs,
        funnel_min_abs_z_km=args.funnel_min_abs_z_km,
        post_merger_mode=args.post_merger_mode,
        jetlike_keep_curves=args.jetlike_keep_curves,
        jetlike_theta_max_deg=args.jetlike_theta_max_deg,
        jetlike_min_radius_km=args.jetlike_min_radius_km,
        jetlike_min_abs_z_km=args.jetlike_min_abs_z_km,
        jetlike_start_delay_ms=args.jetlike_start_delay_ms,
        jetlike_min_keep_curves=args.jetlike_min_keep_curves,
        jetlike_min_curve_length_km=args.jetlike_min_curve_length_km,
        min_seed_separation_km=args.min_seed_separation_km,
        streamline_style=args.streamline_style,
        streamline_batch_size=args.streamline_batch_size,
        streamline_opacity=args.streamline_opacity,
        streamline_tube_radius_km=args.streamline_tube_radius_km,
        streamline_tube_sides=max(3, int(args.streamline_tube_sides)),
        contour_count=args.contour_count,
        contour_low_frac=args.contour_low_frac,
        contour_opacity=args.contour_opacity,
        seed_count_range=(min(seed_counts), max(seed_counts)) if seed_counts else (0, 0),
        remnant_seed_count_range=(
            min(remnant_seed_counts),
            max(remnant_seed_counts),
        )
        if remnant_seed_counts
        else (0, 0),
        funnel_seed_count_range=(
            min(funnel_seed_counts),
            max(funnel_seed_counts),
        )
        if funnel_seed_counts
        else (0, 0),
        streamline_count_range=(
            min(streamline_counts),
            max(streamline_counts),
        )
        if streamline_counts
        else (0, 0),
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
