#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

from plot2d_common import (
    DEFAULT_SIM_DIR,
    gather_series_files,
    infer_sim_name,
    parse_itnum,
)
from unit_converter import CU_CGS, GAUSS_CU


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "plots" / "3d"
DEFAULT_PYVISTA_ROOT = Path(__file__).resolve().parent / "pyvista"
DEFAULT_3D_RESOLUTION = (1440, 1088)
DEFAULT_3D_RHO_SCALAR_BAR_X = 0.81
DEFAULT_3D_B_SCALAR_BAR_X = 0.89
DEFAULT_3D_SCALAR_BAR_X = DEFAULT_3D_B_SCALAR_BAR_X
DEFAULT_3D_SCALAR_BAR_Y = 0.36
DEFAULT_3D_SCALAR_BAR_WIDTH = 0.024
DEFAULT_3D_SCALAR_BAR_HEIGHT = 0.28
DEFAULT_3D_SCALAR_BAR_TITLE_SIZE = 22
DEFAULT_3D_SCALAR_BAR_LABEL_SIZE = 22
DEFAULT_3D_TIME_LABEL_POSITION = (0.045, 0.94)
DEFAULT_3D_THEME = "dark"

CU_TO_KM = CU_CGS.length / 1.0e5
RHO_CU_TO_CGS = CU_CGS.density
TIME_CU_TO_MS = CU_CGS.time / 1.0e-3
B_CU_TO_GAUSS = 1.0 / GAUSS_CU


def get_3d_theme(theme_name: str = DEFAULT_3D_THEME) -> dict[str, object]:
    themes = {
        "dark": {
            "background": "#040816",
            "foreground": "white",
            "text_shadow": True,
        },
        "light": {
            "background": "white",
            "foreground": "black",
            "text_shadow": False,
        },
    }
    try:
        return themes[str(theme_name).lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown 3D theme: {theme_name!r}") from exc


def configure_offscreen_environment() -> None:
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "1")
    os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")
    os.environ.setdefault("VTK_LOG_DEFAULT_LEVEL", "ERROR")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    shader_cache = Path("/tmp/mesa_shader_cache")
    shader_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MESA_SHADER_CACHE_DIR", str(shader_cache))


def sanitize_scalar_field(data: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Scale a scalar field in-place and keep it in float32 when possible."""
    arr = np.asarray(data, dtype=np.float32)
    if scale != 1.0:
        arr *= np.float32(scale)
    return arr


def sanitize_vector_field(data: np.ndarray) -> np.ndarray:
    """Return a float32 view of a vector field without extra full-array work."""
    arr = np.asarray(data, dtype=np.float32)
    return arr


def vector_magnitude_scaled(
    data: np.ndarray,
    scale: float = 1.0,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Return |v| as float32, optionally reusing a caller-provided output buffer."""
    if out is not None and out.shape == data.shape[:-1] and out.dtype == np.float32:
        mag = out
    else:
        mag = np.empty(data.shape[:-1], dtype=np.float32)
    np.einsum("...i,...i->...", data, data, dtype=np.float32, optimize=True, out=mag)
    np.sqrt(mag, out=mag)
    if scale != 1.0:
        mag *= np.float32(scale)
    return mag


def _purge_modules(prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            sys.modules.pop(name, None)


def load_pyvista(preferred_root: str | Path = DEFAULT_PYVISTA_ROOT):
    configure_offscreen_environment()
    errors = []
    preferred_path = Path(preferred_root).expanduser().resolve() if preferred_root else None

    if preferred_path and preferred_path.is_dir():
        if sys.version_info >= (3, 10):
            sys.path.insert(0, str(preferred_path))
            try:
                import pyvista as pv

                return pv, f"vendored:{preferred_path}"
            except Exception as exc:
                errors.append(f"Vendored PyVista at {preferred_path} failed: {exc}")
                _purge_modules("pyvista")
                try:
                    sys.path.remove(str(preferred_path))
                except ValueError:
                    pass
        else:
            errors.append(
                f"Vendored PyVista at {preferred_path} requires Python >= 3.10; active Python is "
                f"{sys.version_info.major}.{sys.version_info.minor}"
            )

    try:
        import pyvista as pv

        return pv, f"system:{Path(pv.__file__).resolve()}"
    except Exception as exc:
        errors.append(f"System PyVista import failed: {exc}")
        raise ImportError("\n".join(errors))


def resolve_3d_paths(
    sim_dir: str | Path,
    render_name: str,
    out_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[list[str], str, Path, Path]:
    sim_dir = str(Path(sim_dir).expanduser().resolve())
    series_files = gather_series_files(sim_dir)
    sim_name = infer_sim_name(sim_dir, series_files)
    out_dir = Path(out_root).expanduser().resolve() / f"{sim_name}_{render_name}_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    movie_file = out_dir / f"{sim_name}_{render_name}.mp4"
    return series_files, sim_name, out_dir, movie_file


def select_series_files(
    series_files: list[str],
    cadence: int = 1,
    max_frames: int | None = None,
    include_last: bool = True,
) -> list[str]:
    if not series_files:
        return []

    stride = max(1, int(cadence))
    selected = list(series_files[::stride])
    if include_last and selected[-1] != series_files[-1]:
        selected.append(series_files[-1])
    if max_frames is not None:
        selected = selected[: max(1, int(max_frames))]
    return selected


def get_record_component(mesh, record_component: str):
    record = mesh[record_component]
    try:
        return record[""]
    except Exception:
        return record


def _compile_mesh_pattern(mesh_name_re):
    if hasattr(mesh_name_re, "match"):
        return mesh_name_re
    return re.compile(str(mesh_name_re))


def list_level_keys(series, iteration: int, mesh_name_re, record_component: str):
    mesh_pattern = _compile_mesh_pattern(mesh_name_re)
    out = []
    itobj = series.iterations[iteration]
    for name in itobj.meshes:
        match = mesh_pattern.match(name)
        if not match:
            continue
        mesh = itobj.meshes[name]
        try:
            _component = get_record_component(mesh, record_component)
        except Exception:
            continue
        out.append((int(match.group(2)), int(match.group(1)), name))
    out.sort(key=lambda item: (item[0], item[1]))
    return out


def list_mesh_keys(series, iteration: int, mesh_name_re):
    mesh_pattern = _compile_mesh_pattern(mesh_name_re)
    out = []
    itobj = series.iterations[iteration]
    for name in itobj.meshes:
        match = mesh_pattern.match(name)
        if not match:
            continue
        out.append((int(match.group(2)), int(match.group(1)), name))
    out.sort(key=lambda item: (item[0], item[1]))
    return out


def get_spacing_offset_km(mesh, cu_to_km: float = CU_TO_KM):
    spacing_cu = np.array(mesh.get_attribute("gridSpacing"), dtype=float)
    offset_cu = np.array(mesh.get_attribute("gridGlobalOffset"), dtype=float)
    dz, dy, dx = spacing_cu * cu_to_km
    z0, y0, x0 = offset_cu * cu_to_km
    return np.array([dx, dy, dz]), np.array([x0, y0, z0])


def compute_mesh_bbox(
    series,
    iteration: int,
    record_component: str,
    mesh_name_re,
    pad_frac: float = 0.02,
    cu_to_km: float = CU_TO_KM,
):
    keys = list_level_keys(series, iteration, mesh_name_re, record_component)
    if not keys:
        raise RuntimeError(f"No meshes matched {mesh_name_re!r} for component {record_component!r}")

    xmin = ymin = zmin = +1.0e99
    xmax = ymax = zmax = -1.0e99
    itobj = series.iterations[iteration]

    for _level, _patch, mesh_key in keys:
        mesh = itobj.meshes[mesh_key]
        component = get_record_component(mesh, record_component)
        nz, ny, nx = [int(size) for size in component.shape]
        dxyz, xyz0 = get_spacing_offset_km(mesh, cu_to_km=cu_to_km)
        dx, dy, dz = dxyz
        x0, y0, z0 = xyz0

        xl = x0 + 0.5 * dx
        xr = x0 + (nx - 0.5) * dx
        yl = y0 + 0.5 * dy
        yr = y0 + (ny - 0.5) * dy
        zl = z0 + 0.5 * dz
        zr = z0 + (nz - 0.5) * dz

        xmin = min(xmin, xl)
        xmax = max(xmax, xr)
        ymin = min(ymin, yl)
        ymax = max(ymax, yr)
        zmin = min(zmin, zl)
        zmax = max(zmax, zr)

    px = (xmax - xmin) * pad_frac
    py = (ymax - ymin) * pad_frac
    pz = (zmax - zmin) * pad_frac
    return ((xmin - px, xmax + px), (ymin - py, ymax + py), (zmin - pz, zmax + pz))


def get_time_code_units(series, iteration: int, record_component: str | None = None):
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
        for name in itobj.meshes:
            mesh = itobj.meshes[name]
            if record_component is not None and record_component not in mesh:
                continue
            for getter in (mesh.get_attribute, get_record_component(mesh, record_component).get_attribute):
                try:
                    return float(getter("time"))
                except Exception:
                    pass
    except Exception:
        pass

    return None


def composite_scalar_volume(
    series,
    iteration: int,
    bbox,
    shape,
    record_component: str,
    mesh_name_re,
    edge_erode: int = 0,
    chunk_z: int = 16,
    tile_xy: int = 256,
    downsample: int = 1,
    cu_to_km: float = CU_TO_KM,
    out: np.ndarray | None = None,
):
    """
    Composite AMR patches onto a uniform nearest-neighbor canvas in code units.

    The output array shape is (Nx, Ny, Nz) so it can be handed directly to
    `pyvista.ImageData` after flattening with Fortran order.
    """

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bbox
    nx_out, ny_out, nz_out = shape
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    zs = np.linspace(zmin, zmax, nz_out)

    if out is not None and out.shape == (nx_out, ny_out, nz_out) and out.dtype == np.float32:
        volume = out
        volume.fill(0.0)
    else:
        volume = np.zeros((nx_out, ny_out, nz_out), dtype=np.float32)
    written = 0
    keys = list_level_keys(series, iteration, mesh_name_re, record_component)

    for _level, _patch, mesh_key in keys:
        itobj = series.iterations[iteration]
        mesh = itobj.meshes[mesh_key]
        component = get_record_component(mesh, record_component)
        nz, ny, nx = [int(size) for size in component.shape]

        dxyz, xyz0 = get_spacing_offset_km(mesh, cu_to_km=cu_to_km)
        dx, dy, dz = dxyz
        x0, y0, z0 = xyz0
        x = x0 + (np.arange(nx) + 0.5) * dx
        y = y0 + (np.arange(ny) + 0.5) * dy
        z = z0 + (np.arange(nz) + 0.5) * dz

        if (
            x.max() < xmin
            or x.min() > xmax
            or y.max() < ymin
            or y.min() > ymax
            or z.max() < zmin
            or z.min() > zmax
        ):
            continue

        mesh_ix0 = np.searchsorted(xs, max(x.min(), xmin), side="left")
        mesh_ix1 = np.searchsorted(xs, min(x.max(), xmax), side="right")
        mesh_iy0 = np.searchsorted(ys, max(y.min(), ymin), side="left")
        mesh_iy1 = np.searchsorted(ys, min(y.max(), ymax), side="right")
        mesh_iz0 = np.searchsorted(zs, max(z.min(), zmin), side="left")
        mesh_iz1 = np.searchsorted(zs, min(z.max(), zmax), side="right")
        if mesh_ix1 <= mesh_ix0 or mesh_iy1 <= mesh_iy0 or mesh_iz1 <= mesh_iz0:
            continue

        mesh_ix0w = min(max(mesh_ix0 + edge_erode, mesh_ix0), mesh_ix1)
        mesh_iy0w = min(max(mesh_iy0 + edge_erode, mesh_iy0), mesh_iy1)
        mesh_iz0w = min(max(mesh_iz0 + edge_erode, mesh_iz0), mesh_iz1)
        mesh_ix1w = max(min(mesh_ix1 - edge_erode, mesh_ix1), mesh_ix0w)
        mesh_iy1w = max(min(mesh_iy1 - edge_erode, mesh_iy1), mesh_iy0w)
        mesh_iz1w = max(min(mesh_iz1 - edge_erode, mesh_iz1), mesh_iz0w)
        if mesh_ix1w <= mesh_ix0w or mesh_iy1w <= mesh_iy0w or mesh_iz1w <= mesh_iz0w:
            continue

        for chunk in component.available_chunks():
            chunk_offset = np.array(chunk.offset, dtype=int)
            chunk_extent = np.array(chunk.extent, dtype=int)
            z0c, y0c, x0c = chunk_offset
            z1c, y1c, x1c = chunk_offset + chunk_extent

            x_chunk = x[x0c:x1c]
            y_chunk = y[y0c:y1c]
            z_chunk = z[z0c:z1c]

            ix0 = max(mesh_ix0w, np.searchsorted(xs, max(x_chunk.min(), xmin), side="left"))
            ix1 = min(mesh_ix1w, np.searchsorted(xs, min(x_chunk.max(), xmax), side="right"))
            iy0 = max(mesh_iy0w, np.searchsorted(ys, max(y_chunk.min(), ymin), side="left"))
            iy1 = min(mesh_iy1w, np.searchsorted(ys, min(y_chunk.max(), ymax), side="right"))
            iz0 = max(mesh_iz0w, np.searchsorted(zs, max(z_chunk.min(), zmin), side="left"))
            iz1 = min(mesh_iz1w, np.searchsorted(zs, min(z_chunk.max(), zmax), side="right"))
            if ix1 <= ix0 or iy1 <= iy0 or iz1 <= iz0:
                continue

            data = component.load_chunk(chunk_offset.tolist(), chunk_extent.tolist())
            series.flush()
            data = np.asarray(data, dtype=np.float32)

            xi_local = np.clip(np.searchsorted(x_chunk, xs[ix0:ix1]), 0, len(x_chunk) - 1)
            yi_local = np.clip(np.searchsorted(y_chunk, ys[iy0:iy1]), 0, len(y_chunk) - 1)
            zi_local = np.clip(np.searchsorted(z_chunk, zs[iz0:iz1]), 0, len(z_chunk) - 1)

            uniq_z, inv = np.unique(zi_local, return_inverse=True)
            plane_cache = {
                index: data[int(z_local), :, :]
                for index, z_local in enumerate(uniq_z)
            }

            for local_k, iz_target in enumerate(range(iz0, iz1)):
                plane = plane_cache[inv[local_k]]
                slab = plane[np.ix_(yi_local, xi_local)]
                volume[ix0:ix1, iy0:iy1, iz_target] = slab.T
                written += slab.size

    stride = max(1, int(downsample))
    if stride > 1:
        volume = volume[::stride, ::stride, ::stride]
        xs = xs[::stride]
        ys = ys[::stride]
        zs = zs[::stride]

    print(f"  [Composite] wrote {written:,} voxels into canvas for it={iteration}")
    return volume, (xs, ys, zs)


def composite_vector_volume(
    series,
    iteration: int,
    bbox,
    shape,
    record_component: str,
    component_names,
    mesh_name_re,
    edge_erode: int = 0,
    chunk_z: int = 16,
    tile_xy: int = 256,
    downsample: int = 1,
    cu_to_km: float = CU_TO_KM,
    out: np.ndarray | None = None,
):
    """
    Composite a 3-component AMR vector field onto a uniform nearest-neighbor
    canvas in code units.

    The output array shape is (Nx, Ny, Nz, 3), matching the point ordering used
    by `pyvista.ImageData` once flattened in Fortran order.
    """

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bbox
    nx_out, ny_out, nz_out = shape
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    zs = np.linspace(zmin, zmax, nz_out)

    if out is not None and out.shape == (nx_out, ny_out, nz_out, 3) and out.dtype == np.float32:
        volume = out
        volume.fill(0.0)
    else:
        volume = np.zeros((nx_out, ny_out, nz_out, 3), dtype=np.float32)
    written = 0
    keys = list_mesh_keys(series, iteration, mesh_name_re)

    for _level, _patch, mesh_key in keys:
        itobj = series.iterations[iteration]
        mesh = itobj.meshes[mesh_key]
        components = [mesh[name] for name in component_names]
        nz, ny, nx = [int(size) for size in components[0].shape]

        dxyz, xyz0 = get_spacing_offset_km(mesh, cu_to_km=cu_to_km)
        dx, dy, dz = dxyz
        x0, y0, z0 = xyz0
        x = x0 + (np.arange(nx) + 0.5) * dx
        y = y0 + (np.arange(ny) + 0.5) * dy
        z = z0 + (np.arange(nz) + 0.5) * dz

        if (
            x.max() < xmin
            or x.min() > xmax
            or y.max() < ymin
            or y.min() > ymax
            or z.max() < zmin
            or z.min() > zmax
        ):
            continue

        mesh_ix0 = np.searchsorted(xs, max(x.min(), xmin), side="left")
        mesh_ix1 = np.searchsorted(xs, min(x.max(), xmax), side="right")
        mesh_iy0 = np.searchsorted(ys, max(y.min(), ymin), side="left")
        mesh_iy1 = np.searchsorted(ys, min(y.max(), ymax), side="right")
        mesh_iz0 = np.searchsorted(zs, max(z.min(), zmin), side="left")
        mesh_iz1 = np.searchsorted(zs, min(z.max(), zmax), side="right")
        if mesh_ix1 <= mesh_ix0 or mesh_iy1 <= mesh_iy0 or mesh_iz1 <= mesh_iz0:
            continue

        mesh_ix0w = min(max(mesh_ix0 + edge_erode, mesh_ix0), mesh_ix1)
        mesh_iy0w = min(max(mesh_iy0 + edge_erode, mesh_iy0), mesh_iy1)
        mesh_iz0w = min(max(mesh_iz0 + edge_erode, mesh_iz0), mesh_iz1)
        mesh_ix1w = max(min(mesh_ix1 - edge_erode, mesh_ix1), mesh_ix0w)
        mesh_iy1w = max(min(mesh_iy1 - edge_erode, mesh_iy1), mesh_iy0w)
        mesh_iz1w = max(min(mesh_iz1 - edge_erode, mesh_iz1), mesh_iz0w)
        if mesh_ix1w <= mesh_ix0w or mesh_iy1w <= mesh_iy0w or mesh_iz1w <= mesh_iz0w:
            continue

        for chunk in components[0].available_chunks():
            chunk_offset = np.array(chunk.offset, dtype=int)
            chunk_extent = np.array(chunk.extent, dtype=int)
            z0c, y0c, x0c = chunk_offset
            z1c, y1c, x1c = chunk_offset + chunk_extent

            x_chunk = x[x0c:x1c]
            y_chunk = y[y0c:y1c]
            z_chunk = z[z0c:z1c]

            ix0 = max(mesh_ix0w, np.searchsorted(xs, max(x_chunk.min(), xmin), side="left"))
            ix1 = min(mesh_ix1w, np.searchsorted(xs, min(x_chunk.max(), xmax), side="right"))
            iy0 = max(mesh_iy0w, np.searchsorted(ys, max(y_chunk.min(), ymin), side="left"))
            iy1 = min(mesh_iy1w, np.searchsorted(ys, min(y_chunk.max(), ymax), side="right"))
            iz0 = max(mesh_iz0w, np.searchsorted(zs, max(z_chunk.min(), zmin), side="left"))
            iz1 = min(mesh_iz1w, np.searchsorted(zs, min(z_chunk.max(), zmax), side="right"))
            if ix1 <= ix0 or iy1 <= iy0 or iz1 <= iz0:
                continue

            chunk_arrays = []
            for component in components:
                data = component.load_chunk(chunk_offset.tolist(), chunk_extent.tolist())
                series.flush()
                data = np.asarray(data, dtype=np.float32)
                chunk_arrays.append(data)

            xi_local = np.clip(np.searchsorted(x_chunk, xs[ix0:ix1]), 0, len(x_chunk) - 1)
            yi_local = np.clip(np.searchsorted(y_chunk, ys[iy0:iy1]), 0, len(y_chunk) - 1)
            zi_local = np.clip(np.searchsorted(z_chunk, zs[iz0:iz1]), 0, len(z_chunk) - 1)

            uniq_z, inv = np.unique(zi_local, return_inverse=True)
            plane_cache = []
            for data in chunk_arrays:
                plane_cache.append(
                    {
                        index: data[int(z_local), :, :]
                        for index, z_local in enumerate(uniq_z)
                    }
                )

            for local_k, iz_target in enumerate(range(iz0, iz1)):
                slabs = []
                for comp_planes in plane_cache:
                    plane = comp_planes[inv[local_k]]
                    slabs.append(plane[np.ix_(yi_local, xi_local)].T)
                volume[ix0:ix1, iy0:iy1, iz_target, 0] = slabs[0]
                volume[ix0:ix1, iy0:iy1, iz_target, 1] = slabs[1]
                volume[ix0:ix1, iy0:iy1, iz_target, 2] = slabs[2]
                written += slabs[0].size

    stride = max(1, int(downsample))
    if stride > 1:
        volume = volume[::stride, ::stride, ::stride, :]
        xs = xs[::stride]
        ys = ys[::stride]
        zs = zs[::stride]

    print(f"  [Composite] wrote {written:,} vector voxels into canvas for it={iteration}")
    return volume, (xs, ys, zs)
