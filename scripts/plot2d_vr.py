#!/usr/bin/env python3
import os
import re
import shutil

import imageio.v2 as imageio
import numpy as np

from plot2d_common import DEFAULT_SIM_DIR, parse_movie_args, resolve_movie_paths
from plot2d_field_movie import (
    CU_TO_KM,
    EDGE_FILL_PIX,
    FPS,
    NXNY,
    TIME_CU_TO_MS,
    FieldConfig,
    _apply_physical_filters,
    _erode,
    _make_cmap,
    composite_axis_from_file,
    get_time_from_file,
    make_grid,
    make_norm,
    plot_panel,
    plt,
    read_plane_window,
    select_level_keys,
)

try:
    from scipy.interpolate import RegularGridInterpolator
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


VEL_MESH_RE = r"^hydrobasex_vel_patch(\d+)_lev(\d+)$"
VELX = "hydrobasex_velx"
VELY = "hydrobasex_vely"
VELZ = "hydrobasex_velz"
VR_MAX_C = 0.5


CONFIG = FieldConfig(
    field_name="vr",
    record_component=VELX,
    mesh_name_re=VEL_MESH_RE,
    colorbar_label=r"$v_r$ [c]",
    cmap_name="seismic",
    scale_to_cgs=1.0,
    norm="linear",
    vmin=-VR_MAX_C,
    vmax=VR_MAX_C,
    valid_min=-1.0,
    valid_max=1.0,
    under_color=None,
)


def _compute_radial_velocity_native(component_a, component_b, axis1_km, axis2_km):
    xx, aa = np.meshgrid(axis2_km, axis1_km, indexing="xy")
    rr = np.hypot(xx, aa)
    vr = np.full_like(component_b, np.nan, dtype=np.float64)
    mask = np.isfinite(component_a) & np.isfinite(component_b) & (rr > 0.0)
    vr[mask] = (component_b[mask] * xx[mask] + component_a[mask] * aa[mask]) / rr[mask]
    return vr


def composite_vr_axis(series, iteration, config, axis="z", ncanvas=NXNY):
    half_width_km = config.default_domain_half_width_cu * CU_TO_KM
    gy = np.linspace(-half_width_km, half_width_km, ncanvas)
    gx = np.linspace(-half_width_km, half_width_km, ncanvas)
    canvas = np.full((gy.size, gx.size), np.nan, dtype=np.float64)

    component_a = VELY if axis == "z" else VELZ
    keys = select_level_keys(series, iteration, config)
    level_counts = {}
    for level, _patch, _mesh_key in keys:
        level_counts[level] = level_counts.get(level, 0) + 1
    edge_pixels = 0 if keys and max(level_counts.values()) == 1 else EDGE_FILL_PIX

    for _level, _patch, mesh_key in keys:
        try:
            component_a_data, (a1, a2) = read_plane_window(
                series,
                iteration,
                mesh_key,
                component_a,
                axis=axis,
                domain_half_width_cu=config.default_domain_half_width_cu,
            )
            component_b_data, _ = read_plane_window(
                series,
                iteration,
                mesh_key,
                VELX,
                axis=axis,
                domain_half_width_cu=config.default_domain_half_width_cu,
            )
        except Exception as exc:
            print(f"  Skip {mesh_key}: {exc}")
            continue

        vr_native = _compute_radial_velocity_native(component_a_data, component_b_data, a1, a2)
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
        values = np.where(np.isfinite(vr_native), vr_native, np.nan)
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


def composite_vr_axis_from_file(file_path, iteration, config, axis, ncanvas):
    import openpmd_api as io

    series = io.Series(file_path, io.Access.read_only)
    try:
        return composite_vr_axis(series, iteration, config, axis=axis, ncanvas=ncanvas)
    finally:
        series.close()


def main():
    global FPS

    args = parse_movie_args(
        CONFIG.field_name,
        default_sim_dir=DEFAULT_SIM_DIR,
        default_domain_half_width_cu=CONFIG.default_domain_half_width_cu,
        default_fps=FPS,
        default_nxny=NXNY,
        default_vmin=CONFIG.vmin,
        default_vmax=CONFIG.vmax,
        default_merger_time_ms=CONFIG.merger_time_ms,
        default_final_after_ms=(
            CONFIG.final_after_ms if CONFIG.final_after_ms is not None else -1.0
        ),
        default_level_mode=CONFIG.level_mode,
    )
    sim_dir = os.path.abspath(os.path.expanduser(args.sim_dir))
    FPS = args.fps
    ncanvas = args.nxny
    final_after_ms = None if args.final_after_ms is not None and args.final_after_ms < 0 else args.final_after_ms

    config = FieldConfig(
        **{
            **CONFIG.__dict__,
            "default_domain_half_width_cu": float(args.domain_half_width_cu),
            "vmin": args.vmin,
            "vmax": args.vmax,
            "merger_time_ms": args.merger_time_ms,
            "final_after_ms": final_after_ms,
            "level_mode": args.level_mode,
        }
    )

    files, sim_name, out_dir, movie_file = resolve_movie_paths(
        sim_dir, field_name=config.field_name, out_root=args.out_root
    )
    os.makedirs(out_dir, exist_ok=True)

    if not files:
        print("No .bp* series found under", sim_dir)
        return

    print(f"Found {len(files)} {config.field_name} series for {sim_name}")
    cmap = _make_cmap(config.cmap_name, config.bad_color, config.under_color)
    norm, cmap = make_norm(config, cmap)
    extent = (
        -config.default_domain_half_width_cu * CU_TO_KM,
        config.default_domain_half_width_cu * CU_TO_KM,
        -config.default_domain_half_width_cu * CU_TO_KM,
        config.default_domain_half_width_cu * CU_TO_KM,
    )

    frames = []
    for file_path in files:
        iteration = 0
        match = re.search(r"\.it(\d+)\.bp\d*$", os.path.basename(file_path))
        if match:
            iteration = int(match.group(1))

        print(f"Loading iteration {iteration}")
        data_xy_cu, _ = composite_vr_axis_from_file(
            file_path, iteration, config, axis="z", ncanvas=ncanvas
        )
        data_xz_cu, _ = composite_vr_axis_from_file(
            file_path, iteration, config, axis="y", ncanvas=ncanvas
        )

        data_xy = _apply_physical_filters(data_xy_cu, config)
        data_xz = _apply_physical_filters(data_xz_cu, config)

        if not np.isfinite(data_xy).any() and not np.isfinite(data_xz).any():
            print(f"  Skipping iteration {iteration}: no finite values in plotting window.")
            continue

        fig, grid = make_grid()
        image_xy = plot_panel(grid[0], data_xy, extent, y_label="y", norm=norm, cmap=cmap)
        plot_panel(grid[1], data_xz, extent, y_label="z", norm=norm, cmap=cmap)

        colorbar = grid.cbar_axes[0].colorbar(image_xy)
        colorbar.ax.set_ylabel(config.colorbar_label)

        t_cu = get_time_from_file(file_path, iteration, VELX)
        if t_cu is not None:
            t_ms = t_cu * TIME_CU_TO_MS
            t_rel = t_ms - config.merger_time_ms
            if (config.final_after_ms is not None) and (t_rel > config.final_after_ms):
                plt.close(fig)
                break
            grid[0].text(
                0.98,
                0.98,
                rf"$t = {t_rel:.1f}\,\mathrm{{ms}}$",
                color="k",
                transform=grid[0].transAxes,
                ha="right",
                va="top",
            )

        frame = os.path.join(out_dir, f"frame_{iteration:08d}.png")
        plt.savefig(frame, bbox_inches="tight", pad_inches=0.05, dpi=300)
        plt.close(fig)
        frames.append(frame)

    if not frames:
        print(f"No {config.field_name} frames were created.")
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


if __name__ == "__main__":
    main()
