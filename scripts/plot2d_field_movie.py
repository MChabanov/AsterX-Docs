#!/usr/bin/env python3
from dataclasses import dataclass
import os
import re
import shutil
from typing import Optional, Tuple

import imageio.v2 as imageio
import numpy as np

try:
    import openpmd_api as io
except ImportError as exc:
    raise ImportError(
        "Failed to import openpmd_api. On Vista, load the ADIOS2 runtime first "
        "with `module load adios2`."
    ) from exc

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib import colors as mcolors
from mpl_toolkits.axes_grid1 import ImageGrid

from plot2d_common import (
    DEFAULT_DOMAIN_HALF_WIDTH_CU,
    DEFAULT_SIM_DIR,
    parse_movie_args,
    resolve_movie_paths,
)
from unit_converter import CU_CGS, GAUSS_CU


# ------------------ styling: LaTeX only if available ------------------
USE_TEX = shutil.which("latex") is not None
plt.rcParams.update({
    "text.usetex": bool(USE_TEX),
    "font.family": "serif" if USE_TEX else "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.labelpad": 0.5,
    "ytick.major.pad": 0.5,
    "ytick.minor.pad": 0.5,
})


# ------------------ conversions ------------------
CU_TO_KM = CU_CGS.length / 1.0e5
LENGTH_CU_TO_CM = CU_CGS.length
TIME_CU_TO_MS = CU_CGS.time / 1.0e-3
MASS_CU_TO_G = CU_CGS.mass
ENERGY_CU_TO_ERG = CU_CGS.energy
RHO_CU_TO_CGS = CU_CGS.density
PRESSURE_CU_TO_CGS = CU_CGS.pressure
B_CU_TO_GAUSS = 1.0 / GAUSS_CU
SPEED_CU_TO_CGS = CU_CGS.velocity
EPS_CU_TO_ERG_PER_G = ENERGY_CU_TO_ERG / MASS_CU_TO_G
HC_CU_TO_CM2_INV = 1.0 / (LENGTH_CU_TO_CM ** 2)

FPS = 1
NXNY = 1024
EDGE_FILL_PIX = 1
TIME_TEXT_COLOR = "w"

RHO_REC_COMP = "hydrobasex_rho"
RHO_REC_NAME_RE = re.compile(r"^hydrobasex_rho_patch(\d+)_lev(\d+)$")


try:
    from scipy.interpolate import RegularGridInterpolator
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


@dataclass(frozen=True)
class FieldConfig:
    field_name: str
    record_component: str
    mesh_name_re: str
    colorbar_label: str
    cmap_name: str
    scale_to_cgs: float = 1.0
    norm: str = "log"
    vmin: float = 1.0
    vmax: float = 10.0
    linthresh: float = 1e-12
    boundaries: Optional[Tuple[float, ...]] = None
    bad_color: str = "#0b0e2c"
    under_color: Optional[str] = "#0b0e2c"
    merger_time_ms: float = 14.0
    final_after_ms: Optional[float] = 30.0
    default_domain_half_width_cu: float = DEFAULT_DOMAIN_HALF_WIDTH_CU
    rho_mask_cgs: Optional[float] = None
    valid_min: Optional[float] = None
    valid_max: Optional[float] = None
    abs_value: bool = False
    level_mode: str = "auto"

    @property
    def mesh_pattern(self):
        return re.compile(self.mesh_name_re)


def _clean_var(label):
    return re.sub(r"^\$+|\$+$", "", str(label).strip())


def _make_cmap(name, bad_color, under_color):
    cmap = matplotlib.colormaps.get_cmap(name).copy()
    if bad_color is not None:
        cmap.set_bad(color=bad_color, alpha=1.0)
    if under_color is not None:
        cmap.set_under(color=under_color, alpha=1.0)
    return cmap


def _get_position(component):
    try:
        return np.array(component.get_attribute("position"), dtype=float)
    except Exception:
        return np.array([0.5, 0.5, 0.5], dtype=float)


def _index_for_coordinate(offset, spacing, position, size, coordinate):
    idx = int(round((coordinate - offset) / spacing - position))
    return max(0, min(size - 1, idx))


def _index_range(offset, spacing, position, size, lower, upper, pad=1):
    if lower > upper:
        lower, upper = upper, lower
    start = int(np.floor((lower - offset) / spacing - position)) - pad
    stop = int(np.ceil((upper - offset) / spacing - position)) + pad + 1
    return max(0, start), min(size, stop)


def _erode(mask, n=1):
    m = mask.astype(bool).copy()
    for _ in range(max(0, int(n))):
        up = np.zeros_like(m)
        up[1:, :] = m[:-1, :]
        down = np.zeros_like(m)
        down[:-1, :] = m[1:, :]
        left = np.zeros_like(m)
        left[:, 1:] = m[:, :-1]
        right = np.zeros_like(m)
        right[:, :-1] = m[:, 1:]
        m = m & up & down & left & right
    return m


def list_level_keys(series, iteration, mesh_pattern, record_component):
    out = []
    itobj = series.iterations[iteration]
    for name in itobj.meshes:
        match = mesh_pattern.match(name)
        if not match:
            continue
        mesh = itobj.meshes[name]
        if record_component not in mesh:
            continue
        out.append((int(match.group(2)), int(match.group(1)), name))
    out.sort(key=lambda item: (item[0], item[1]))
    return out


def _same_extent_layout(series, iteration, keys, record_component):
    if len(keys) <= 1:
        return False

    reference_offset = None
    reference_extent = None
    for _level, _patch, mesh_key in keys:
        mesh = series.iterations[iteration].meshes[mesh_key]
        comp = mesh[record_component]
        shape = np.array([int(s) for s in comp.shape], dtype=float)
        spacing = np.array(mesh.get_attribute("gridSpacing"), dtype=float)
        offset = np.array(mesh.get_attribute("gridGlobalOffset"), dtype=float)
        extent = spacing * (shape - 1.0)
        if reference_offset is None:
            reference_offset = offset
            reference_extent = extent
            continue
        if not np.allclose(offset, reference_offset, rtol=0.0, atol=1e-12):
            return False
        if not np.allclose(extent, reference_extent, rtol=1e-12, atol=1e-12):
            return False
    return True


def _mesh_has_single_full_chunk(series, iteration, mesh_key, record_component):
    mesh = series.iterations[iteration].meshes[mesh_key]
    comp = mesh[record_component]
    shape = tuple(int(s) for s in comp.shape)
    chunks = list(comp.available_chunks())
    if len(chunks) != 1:
        return False
    chunk = chunks[0]
    return tuple(int(v) for v in chunk.offset) == tuple(0 for _ in shape) and tuple(
        int(v) for v in chunk.extent
    ) == shape


def select_level_keys(series, iteration, config):
    keys = list_level_keys(series, iteration, config.mesh_pattern, config.record_component)
    if not keys:
        return keys

    if config.level_mode == "finest":
        max_level = max(level for level, _patch, _mesh_key in keys)
        return [item for item in keys if item[0] == max_level]

    if config.level_mode == "coarsest":
        min_level = min(level for level, _patch, _mesh_key in keys)
        return [item for item in keys if item[0] == min_level]

    if config.level_mode == "auto":
        level_counts = {}
        for level, _patch, _mesh_key in keys:
            level_counts[level] = level_counts.get(level, 0) + 1
        one_patch_per_level = max(level_counts.values()) == 1
        dense_full_meshes = all(
            _mesh_has_single_full_chunk(series, iteration, mesh_key, config.record_component)
            for _level, _patch, mesh_key in keys
        )
        if (
            one_patch_per_level
            and dense_full_meshes
            and _same_extent_layout(series, iteration, keys, config.record_component)
        ):
            max_level = max(level for level, _patch, _mesh_key in keys)
            return [item for item in keys if item[0] == max_level]

    return keys


def read_plane_window(series, iteration, mesh_key, record_component, axis,
                      domain_half_width_cu):
    itobj = series.iterations[iteration]
    mesh = itobj.meshes[mesh_key]
    comp = mesh[record_component]
    position = _get_position(comp)
    shape = tuple(int(s) for s in comp.shape)
    spacing = np.array(mesh.get_attribute("gridSpacing"), dtype=float)
    offset = np.array(mesh.get_attribute("gridGlobalOffset"), dtype=float)

    x0, x1 = _index_range(offset[2], spacing[2], position[2], shape[2],
                          -domain_half_width_cu, domain_half_width_cu)
    chunks = list(comp.available_chunks())
    if axis == "z":
        a0, a1 = _index_range(offset[1], spacing[1], position[1], shape[1],
                              -domain_half_width_cu, domain_half_width_cu)
        slab_idx = _index_for_coordinate(offset[0], spacing[0], position[0], shape[0], 0.0)
        arr = np.full((a1 - a0, x1 - x0), np.nan, dtype=np.float64)
        for chunk in chunks:
            chunk_offset = np.array(chunk.offset, dtype=int)
            chunk_extent = np.array(chunk.extent, dtype=int)
            if not (chunk_offset[0] <= slab_idx < chunk_offset[0] + chunk_extent[0]):
                continue

            yy0 = max(a0, int(chunk_offset[1]))
            yy1 = min(a1, int(chunk_offset[1] + chunk_extent[1]))
            xx0 = max(x0, int(chunk_offset[2]))
            xx1 = min(x1, int(chunk_offset[2] + chunk_extent[2]))
            if yy1 <= yy0 or xx1 <= xx0:
                continue

            slab = comp.load_chunk(
                [slab_idx, yy0, xx0],
                [1, yy1 - yy0, xx1 - xx0],
            )
            series.flush()
            sub = np.asarray(slab, dtype=np.float64)[0, :, :]
            arr[yy0 - a0:yy1 - a0, xx0 - x0:xx1 - x0] = sub
        axis1 = offset[1] + (np.arange(a0, a1) + position[1]) * spacing[1]
        axis2 = offset[2] + (np.arange(x0, x1) + position[2]) * spacing[2]
        return arr, (axis1 * CU_TO_KM, axis2 * CU_TO_KM)

    if axis == "y":
        a0, a1 = _index_range(offset[0], spacing[0], position[0], shape[0],
                              -domain_half_width_cu, domain_half_width_cu)
        slab_idx = _index_for_coordinate(offset[1], spacing[1], position[1], shape[1], 0.0)
        arr = np.full((a1 - a0, x1 - x0), np.nan, dtype=np.float64)
        for chunk in chunks:
            chunk_offset = np.array(chunk.offset, dtype=int)
            chunk_extent = np.array(chunk.extent, dtype=int)
            if not (chunk_offset[1] <= slab_idx < chunk_offset[1] + chunk_extent[1]):
                continue

            zz0 = max(a0, int(chunk_offset[0]))
            zz1 = min(a1, int(chunk_offset[0] + chunk_extent[0]))
            xx0 = max(x0, int(chunk_offset[2]))
            xx1 = min(x1, int(chunk_offset[2] + chunk_extent[2]))
            if zz1 <= zz0 or xx1 <= xx0:
                continue

            slab = comp.load_chunk(
                [zz0, slab_idx, xx0],
                [zz1 - zz0, 1, xx1 - xx0],
            )
            series.flush()
            sub = np.asarray(slab, dtype=np.float64)[:, 0, :]
            arr[zz0 - a0:zz1 - a0, xx0 - x0:xx1 - x0] = sub
        axis1 = offset[0] + (np.arange(a0, a1) + position[0]) * spacing[0]
        axis2 = offset[2] + (np.arange(x0, x1) + position[2]) * spacing[2]
        return arr, (axis1 * CU_TO_KM, axis2 * CU_TO_KM)

    raise ValueError("axis must be 'z' or 'y'")


def composite_axis(series, iteration, config, axis="z", ncanvas=NXNY):
    half_width_km = config.default_domain_half_width_cu * CU_TO_KM
    gy = np.linspace(-half_width_km, half_width_km, ncanvas)
    gx = np.linspace(-half_width_km, half_width_km, ncanvas)
    canvas = np.full((gy.size, gx.size), np.nan, dtype=np.float64)

    keys = select_level_keys(series, iteration, config)
    level_counts = {}
    for level, _patch, _mesh_key in keys:
        level_counts[level] = level_counts.get(level, 0) + 1
    edge_pixels = 0 if keys and max(level_counts.values()) == 1 else EDGE_FILL_PIX

    for _level, _patch, mesh_key in keys:
        try:
            slab, (a1, a2) = read_plane_window(
                series,
                iteration,
                mesh_key,
                config.record_component,
                axis=axis,
                domain_half_width_cu=config.default_domain_half_width_cu,
            )
        except Exception as exc:
            print(f"  Skip {mesh_key}: {exc}")
            continue

        if a2.max() < gx.min() or a2.min() > gx.max():
            continue
        if a1.max() < gy.min() or a1.min() > gy.max():
            continue

        j0 = np.searchsorted(gy, max(a1.min(), gy.min()), side="left")
        j1 = np.searchsorted(gy, min(a1.max(), gy.max()), side="right")
        i0 = np.searchsorted(gx, max(a2.min(), gx.min()), side="left")
        i1 = np.searchsorted(gx, min(a2.max(), gx.max()), side="right")
        if j1 <= j0 or i1 <= i0:
            continue

        sub_gy = gy[j0:j1]
        sub_gx = gx[i0:i1]
        values = np.where(np.isfinite(slab), slab, np.nan)
        if HAVE_SCIPY:
            try:
                interpolator = RegularGridInterpolator(
                    (a1, a2),
                    values,
                    bounds_error=False,
                    fill_value=np.nan,
                )
                xx, yy = np.meshgrid(sub_gx, sub_gy, indexing="xy")
                sub = interpolator((yy, xx))
            except Exception:
                jj = np.clip(np.searchsorted(a1, sub_gy), 0, len(a1) - 1)
                ii = np.clip(np.searchsorted(a2, sub_gx), 0, len(a2) - 1)
                sub = values[jj[:, None], ii[None, :]]
        else:
            jj = np.clip(np.searchsorted(a1, sub_gy), 0, len(a1) - 1)
            ii = np.clip(np.searchsorted(a2, sub_gx), 0, len(a2) - 1)
            sub = values[jj[:, None], ii[None, :]]

        core = _erode(np.isfinite(sub), n=edge_pixels)
        if np.any(core):
            block = canvas[j0:j1, i0:i1]
            block[core] = sub[core]
            canvas[j0:j1, i0:i1] = block

    return canvas, (gy, gx)


def get_time_code_units(series, iteration, record_component):
    itobj = series.iterations[iteration]
    for getter in (
        lambda: float(getattr(itobj, "time")),
        lambda: float(itobj.get_attribute("time")),
    ):
        try:
            return getter()
        except Exception:
            pass

    try:
        for mesh_name in itobj.meshes:
            mesh = itobj.meshes[mesh_name]
            if record_component not in mesh:
                continue
            for getter in (
                lambda: float(mesh.get_attribute("time")),
                lambda: float(mesh[record_component].get_attribute("time")),
            ):
                try:
                    return getter()
                except Exception:
                    pass
    except Exception:
        pass
    return None


def make_grid():
    fig = plt.figure(figsize=(7.8, 3.5), constrained_layout=True)
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(1, 2),
        axes_pad=(0.50, 0.25),
        share_all=False,
        cbar_mode="single",
        cbar_location="right",
        cbar_size="3%",
        cbar_pad=0.05,
        label_mode="all",
    )
    for ax in grid:
        ax.tick_params(labelleft=True)
    return fig, grid


def make_norm(config, cmap):
    if config.norm == "log":
        return mcolors.LogNorm(vmin=config.vmin, vmax=config.vmax), cmap
    if config.norm == "linear":
        return mcolors.Normalize(vmin=config.vmin, vmax=config.vmax), cmap
    if config.norm == "symlog":
        return mcolors.SymLogNorm(
            linthresh=config.linthresh,
            vmin=config.vmin,
            vmax=config.vmax,
            base=10.0,
        ), cmap
    if config.norm == "discrete":
        boundaries = np.array(config.boundaries, dtype=float)
        ncolors = len(boundaries) - 1
        cmap = matplotlib.colormaps.get_cmap(config.cmap_name).resampled(ncolors).copy()
        cmap.set_bad(color=config.bad_color, alpha=1.0)
        return mcolors.BoundaryNorm(boundaries, cmap.N, clip=True), cmap
    raise ValueError(f"Unsupported norm type: {config.norm}")


def plot_panel(ax, data2d, extent, y_label, norm, cmap):
    var = _clean_var(y_label)
    image = ax.imshow(
        data2d,
        origin="lower",
        extent=extent,
        norm=norm,
        cmap=cmap,
        interpolation="none",
    )
    ax.set_xlabel(r"$x$ [km]")
    ax.set_ylabel(rf"${var}$ [km]")
    minor = ticker.MultipleLocator(20)
    major = ticker.MultipleLocator(100)
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_minor_locator(minor)
    ax.xaxis.set_major_locator(major)
    ax.yaxis.set_major_locator(major)
    return image


def _apply_physical_filters(data, config):
    out = np.array(data, copy=True, dtype=np.float64)
    if config.scale_to_cgs != 1.0:
        out *= config.scale_to_cgs
    if config.abs_value:
        out = np.abs(out)
    out[~np.isfinite(out)] = np.nan
    if config.valid_min is not None:
        out[out < config.valid_min] = np.nan
    if config.valid_max is not None:
        out[out > config.valid_max] = np.nan
    if config.norm == "log":
        out[out <= 0.0] = np.nan
    return out


def _apply_rho_mask(data, rho_data_cu, config):
    if config.rho_mask_cgs is None:
        return data
    rho_cgs = rho_data_cu * RHO_CU_TO_CGS
    masked = np.array(data, copy=True)
    masked[~np.isfinite(rho_cgs) | (rho_cgs <= config.rho_mask_cgs)] = np.nan
    return masked


def composite_axis_from_file(file_path, iteration, config, axis, ncanvas):
    series = io.Series(file_path, io.Access.read_only)
    try:
        return composite_axis(series, iteration, config, axis=axis, ncanvas=ncanvas)
    finally:
        series.close()


def get_time_from_file(file_path, iteration, record_component):
    series = io.Series(file_path, io.Access.read_only)
    try:
        return get_time_code_units(series, iteration, record_component)
    finally:
        series.close()


def run_field_movie(config):
    global FPS

    args = parse_movie_args(
        config.field_name,
        default_sim_dir=DEFAULT_SIM_DIR,
        default_domain_half_width_cu=config.default_domain_half_width_cu,
        default_fps=FPS,
        default_nxny=NXNY,
        default_vmin=config.vmin,
        default_vmax=config.vmax,
        default_merger_time_ms=config.merger_time_ms,
        default_final_after_ms=(
            config.final_after_ms if config.final_after_ms is not None else -1.0
        ),
        default_level_mode=config.level_mode,
    )
    sim_dir = os.path.abspath(os.path.expanduser(args.sim_dir))
    FPS = args.fps
    ncanvas = args.nxny
    final_after_ms = None if args.final_after_ms is not None and args.final_after_ms < 0 else args.final_after_ms

    base_config = FieldConfig(**{
        **config.__dict__,
        "default_domain_half_width_cu": float(args.domain_half_width_cu),
        "vmin": args.vmin,
        "vmax": args.vmax,
        "merger_time_ms": args.merger_time_ms,
        "final_after_ms": final_after_ms,
        "level_mode": args.level_mode,
    })

    files, sim_name, out_dir, movie_file = resolve_movie_paths(
        sim_dir, field_name=base_config.field_name, out_root=args.out_root
    )
    os.makedirs(out_dir, exist_ok=True)

    if not files:
        print("No .bp* series found under", sim_dir)
        return

    print(f"Found {len(files)} {base_config.field_name} series for {sim_name}")
    cmap = _make_cmap(base_config.cmap_name, base_config.bad_color, base_config.under_color)
    norm, cmap = make_norm(base_config, cmap)

    frames = []
    extent_xy = (
        -base_config.default_domain_half_width_cu * CU_TO_KM,
        base_config.default_domain_half_width_cu * CU_TO_KM,
        -base_config.default_domain_half_width_cu * CU_TO_KM,
        base_config.default_domain_half_width_cu * CU_TO_KM,
    )
    extent_xz = extent_xy

    rho_mask_needed = base_config.rho_mask_cgs is not None
    rho_config = FieldConfig(
        field_name="rho_mask",
        record_component=RHO_REC_COMP,
        mesh_name_re=RHO_REC_NAME_RE.pattern,
        colorbar_label="",
        cmap_name="plasma",
    )

    for file_path in files:
        iteration = 0
        match = re.search(r"\.it(\d+)\.bp\d*$", os.path.basename(file_path))
        if match:
            iteration = int(match.group(1))

        print(f"Loading iteration {iteration}")

        data_xy_cu, _ = composite_axis_from_file(
            file_path, iteration, base_config, axis="z", ncanvas=ncanvas
        )
        data_xz_cu, _ = composite_axis_from_file(
            file_path, iteration, base_config, axis="y", ncanvas=ncanvas
        )

        if rho_mask_needed:
            rho_xy_cu, _ = composite_axis_from_file(
                file_path, iteration, rho_config, axis="z", ncanvas=ncanvas
            )
            rho_xz_cu, _ = composite_axis_from_file(
                file_path, iteration, rho_config, axis="y", ncanvas=ncanvas
            )
        else:
            rho_xy_cu = None
            rho_xz_cu = None

        data_xy = _apply_physical_filters(data_xy_cu, base_config)
        data_xz = _apply_physical_filters(data_xz_cu, base_config)
        if rho_mask_needed:
            data_xy = _apply_rho_mask(data_xy, rho_xy_cu, base_config)
            data_xz = _apply_rho_mask(data_xz, rho_xz_cu, base_config)

        if not np.isfinite(data_xy).any() and not np.isfinite(data_xz).any():
            print(f"  Skipping iteration {iteration}: no finite values in plotting window.")
            continue

        fig, grid = make_grid()
        image_xy = plot_panel(grid[0], data_xy, extent_xy, y_label="y", norm=norm, cmap=cmap)
        plot_panel(grid[1], data_xz, extent_xz, y_label="z", norm=norm, cmap=cmap)

        colorbar = grid.cbar_axes[0].colorbar(image_xy)
        colorbar.ax.set_ylabel(base_config.colorbar_label)

        t_cu = get_time_from_file(file_path, iteration, base_config.record_component)
        if t_cu is not None:
            t_ms = t_cu * TIME_CU_TO_MS
            t_rel = t_ms - base_config.merger_time_ms
            if (base_config.final_after_ms is not None) and (t_rel > base_config.final_after_ms):
                plt.close(fig)
                break
            grid[0].text(
                0.98,
                0.98,
                rf"$t = {t_rel:.1f}\,\mathrm{{ms}}$",
                color=TIME_TEXT_COLOR,
                transform=grid[0].transAxes,
                ha="right",
                va="top",
            )

        frame = os.path.join(out_dir, f"frame_{iteration:08d}.png")
        plt.savefig(frame, bbox_inches="tight", pad_inches=0.05, dpi=300)
        plt.close(fig)
        frames.append(frame)

    if not frames:
        print(f"No {base_config.field_name} frames were created.")
        return

    print("Combining frames…")
    images = [imageio.imread(frame) for frame in frames]

    have_ffmpeg = shutil.which("ffmpeg") is not None
    mp4_ok = False
    try:
        imageio.mimsave(movie_file, images, fps=FPS)
        mp4_ok = True
    except Exception as exc:
        print(f"MP4 writer not available ({exc}). Falling back to GIF…")

    if not mp4_ok:
        gif_file = os.path.splitext(movie_file)[0] + ".gif"
        imageio.mimsave(gif_file, images, fps=FPS)
        print(f"GIF saved to: {gif_file}")
        if have_ffmpeg:
            print("Tip: convert GIF -> MP4 with:")
            print(
                f"  ffmpeg -y -r {FPS} -i {os.path.join(out_dir, 'frame_%08d.png')} "
                f"-c:v libx264 -pix_fmt yuv420p -crf 20 {movie_file}"
            )
        else:
            print("Tip: install FFmpeg for direct MP4 output: pip install --user 'imageio[ffmpeg]'")

    print(f"Done. Wrote {'MP4' if mp4_ok else 'GIF'}.")
