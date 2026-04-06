#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from types import SimpleNamespace
import os
import re
from typing import List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

try:
    import openpmd_api as io
except ImportError as exc:
    raise ImportError(
        "Failed to import openpmd_api. Load the ADIOS2 runtime first."
    ) from exc

try:
    import h5py
except ImportError as exc:
    raise ImportError(
        "Failed to import h5py. THC HDF5 reading needs h5py."
    ) from exc

try:
    from scipy.interpolate import RegularGridInterpolator
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

from plot2d_common import gather_series_files


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "plots", "compare_thc_nux_m1")
DEFAULT_THC_DIR = "/scratch/09228/jkalinan/simulations/m1tests/kerrschildLR"

THC_DATASET_RE = re.compile(
    r"^THC_M1::(?P<field>.+) it=(?P<it>\d+) tl=0(?: m=0)? rl=(?P<rl>\d+)$"
)


@dataclass(frozen=True)
class FieldSpec:
    cli_name: str
    label: str
    nux_mesh_re: str
    nux_component: str
    thc_component: str
    default_cmap: str = "RdBu_r"
    default_vmin: Optional[float] = None
    default_vmax: Optional[float] = None


@dataclass(frozen=True)
class SnapshotRef:
    it: int
    time: float
    path: str
    dataset_key: Optional[str] = None


FIELD_SPECS = {
    "re": FieldSpec(
        cli_name="rE",
        label="rE",
        nux_mesh_re=r"^nux_m1_re_patch(\d+)_lev(\d+)$",
        nux_component="nux_m1_re[{species}]",
        thc_component="rE[{species}]",
        default_cmap="RdBu_r",
        default_vmin=0.0,
        default_vmax=1.27,
    ),
    "rn": FieldSpec(
        cli_name="rN",
        label="rN",
        nux_mesh_re=r"^nux_m1_rn_patch(\d+)_lev(\d+)$",
        nux_component="nux_m1_rn[{species}]",
        thc_component="rN[{species}]",
    ),
    "rfx": FieldSpec(
        cli_name="rFx",
        label="rFx",
        nux_mesh_re=r"^nux_m1_rf_patch(\d+)_lev(\d+)$",
        nux_component="nux_m1_rfx[{species}]",
        thc_component="rFx[{species}]",
    ),
    "rfy": FieldSpec(
        cli_name="rFy",
        label="rFy",
        nux_mesh_re=r"^nux_m1_rf_patch(\d+)_lev(\d+)$",
        nux_component="nux_m1_rfy[{species}]",
        thc_component="rFy[{species}]",
    ),
    "rfz": FieldSpec(
        cli_name="rFz",
        label="rFz",
        nux_mesh_re=r"^nux_m1_rf_patch(\d+)_lev(\d+)$",
        nux_component="nux_m1_rfz[{species}]",
        thc_component="rFz[{species}]",
    ),
    "rpxx": FieldSpec(
        cli_name="rPxx",
        label="rPxx",
        nux_mesh_re=r"^nux_m1_rp_patch(\d+)_lev(\d+)$",
        nux_component="nux_m1_rpxx[{species}]",
        thc_component="rPxx[{species}]",
    ),
    "rpxy": FieldSpec(
        cli_name="rPxy",
        label="rPxy",
        nux_mesh_re=r"^nux_m1_rp_patch(\d+)_lev(\d+)$",
        nux_component="nux_m1_rpxy[{species}]",
        thc_component="rPxy[{species}]",
    ),
    "rpxz": FieldSpec(
        cli_name="rPxz",
        label="rPxz",
        nux_mesh_re=r"^nux_m1_rp_patch(\d+)_lev(\d+)$",
        nux_component="nux_m1_rpxz[{species}]",
        thc_component="rPxz[{species}]",
    ),
    "rpyy": FieldSpec(
        cli_name="rPyy",
        label="rPyy",
        nux_mesh_re=r"^nux_m1_rp_patch(\d+)_lev(\d+)$",
        nux_component="nux_m1_rpyy[{species}]",
        thc_component="rPyy[{species}]",
    ),
    "rpyz": FieldSpec(
        cli_name="rPyz",
        label="rPyz",
        nux_mesh_re=r"^nux_m1_rp_patch(\d+)_lev(\d+)$",
        nux_component="nux_m1_rpyz[{species}]",
        thc_component="rPyz[{species}]",
    ),
    "rpzz": FieldSpec(
        cli_name="rPzz",
        label="rPzz",
        nux_mesh_re=r"^nux_m1_rp_patch(\d+)_lev(\d+)$",
        nux_component="nux_m1_rpzz[{species}]",
        thc_component="rPzz[{species}]",
    ),
}



FIELD_SETS = {
    "scalars": ["rE", "rN"],
    "fluxes": ["rFx", "rFy", "rFz"],
    "pressures": ["rPxx", "rPxy", "rPxz", "rPyy", "rPyz", "rPzz"],
    "all": ["rE", "rN", "rFx", "rFy", "rFz", "rPxx", "rPxy", "rPxz", "rPyy", "rPyz", "rPzz"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Make THC-vs-nuX xz-plane comparison movies with a third panel "
            "showing the relative difference."
        )
    )
    parser.add_argument(
        "nux_path",
        help="nuX output directory or a single BP5 series file.",
    )
    parser.add_argument(
        "--thc-dir",
        default=DEFAULT_THC_DIR,
        help="THC directory containing the *.xz.h5 files.",
    )
    parser.add_argument(
        "--field",
        default="rE",
        choices=[spec.cli_name for spec in FIELD_SPECS.values()],
        help="Field/component to compare.",
    )
    parser.add_argument(
        "--field-set",
        choices=sorted(FIELD_SETS.keys()),
        default=None,
        help=(
            "Batch-generate one movie or PNG per field in the selected set. "
            "If given, this overrides --field."
        ),
    )
    parser.add_argument(
        "--species",
        type=int,
        default=0,
        help="Species index inside [...].",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help="If given, render one nearest-time PNG instead of a movie.",
    )
    parser.add_argument(
        "--time-min",
        type=float,
        default=None,
        help="Earliest nuX frame time to include in movie mode.",
    )
    parser.add_argument(
        "--time-max",
        type=float,
        default=None,
        help="Latest nuX frame time to include in movie mode.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Use every Nth nuX frame in movie mode.",
    )
    parser.add_argument(
        "--xmin",
        type=float,
        default=None,
        help="Override x minimum. Default: keep full union extent.",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=None,
        help="Override x maximum. Default: keep full union extent.",
    )
    parser.add_argument(
        "--zmin",
        type=float,
        default=None,
        help="Override z minimum. Default: keep full union extent.",
    )
    parser.add_argument(
        "--zmax",
        type=float,
        default=None,
        help="Override z maximum. Default: keep full union extent.",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=None,
        help="Output x resolution. Default: inferred from finest spacing.",
    )
    parser.add_argument(
        "--nz",
        type=int,
        default=None,
        help="Output z resolution. Default: inferred from finest spacing.",
    )
    parser.add_argument(
        "--nux-y",
        type=float,
        default=None,
        help="Fixed y coordinate for the nuX xz slice. Default: first y cell center.",
    )
    parser.add_argument(
        "--out-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output directory for frames and movies.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Movie frame rate.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Frame DPI.",
    )
    parser.add_argument(
        "--cmap",
        default=None,
        help="Main field colormap override.",
    )
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument(
        "--diff-cmap",
        default="viridis",
        help="Relative-difference colormap in unsigned mode.",
    )
    parser.add_argument(
        "--diff-vmax",
        type=float,
        default=None,
        help="Relative-difference upper bound. Auto if omitted.",
    )
    parser.add_argument(
        "--diff-floor",
        type=float,
        default=1.0e-12,
        help="Denominator floor in the relative difference.",
    )
    parser.add_argument(
        "--signed-diff",
        action="store_true",
        help="Use signed relative difference instead of absolute relative difference.",
    )
    return parser.parse_args()


def resolve_field_spec(field_name: str) -> FieldSpec:
    key = field_name.strip().lower()
    for short_key, spec in FIELD_SPECS.items():
        if key == short_key or key == spec.cli_name.lower():
            return spec
    raise KeyError(f"Unsupported field '{field_name}'")


def resolve_requested_fields(args) -> List[FieldSpec]:
    if args.field_set is None:
        return [resolve_field_spec(args.field)]

    return [resolve_field_spec(name) for name in FIELD_SETS[args.field_set]]


def args_for_field(args, field_name: str):
    updated = vars(args).copy()
    updated["field"] = field_name
    updated["field_set"] = None
    return SimpleNamespace(**updated)


def make_output_stem(args, spec: FieldSpec) -> str:
    return f"compare_{spec.cli_name}_sp{args.species}"


def nearest_indices(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(src, dst)
    idx = np.clip(idx, 1, len(src) - 1)
    left = src[idx - 1]
    right = src[idx]
    return np.where(np.abs(dst - left) <= np.abs(right - dst), idx - 1, idx)


def resample_rectilinear(
    data_zx: np.ndarray,
    x_src: np.ndarray,
    z_src: np.ndarray,
    x_dst: np.ndarray,
    z_dst: np.ndarray,
) -> np.ndarray:
    if HAVE_SCIPY:
        interp = RegularGridInterpolator(
            (z_src, x_src),
            data_zx,
            bounds_error=False,
            fill_value=np.nan,
        )
        zz, xx = np.meshgrid(z_dst, x_dst, indexing="ij")
        return interp((zz, xx))

    out = np.full((len(z_dst), len(x_dst)), np.nan, dtype=np.float64)
    valid_x = (x_dst >= x_src[0]) & (x_dst <= x_src[-1])
    valid_z = (z_dst >= z_src[0]) & (z_dst <= z_src[-1])
    if not np.any(valid_x) or not np.any(valid_z):
        return out

    ix = nearest_indices(x_src, x_dst[valid_x])
    iz = nearest_indices(z_src, z_dst[valid_z])
    out[np.ix_(valid_z, valid_x)] = data_zx[np.ix_(iz, ix)]
    return out


def centers_from_extent(vmin: float, vmax: float, n: Optional[int], dv: float) -> np.ndarray:
    if n is not None:
        return np.linspace(vmin, vmax, n, dtype=np.float64)
    steps = int(round((vmax - vmin) / dv)) + 1
    return vmin + np.arange(steps, dtype=np.float64) * dv


def extent_from_centers(x: np.ndarray, z: np.ndarray) -> Tuple[float, float, float, float]:
    if len(x) < 2 or len(z) < 2:
        raise ValueError("Need at least 2 samples per dimension")
    dx = float(x[1] - x[0])
    dz = float(z[1] - z[0])
    return (x[0] - 0.5 * dx, x[-1] + 0.5 * dx, z[0] - 0.5 * dz, z[-1] + 0.5 * dz)


def scan_thc_snapshots(thc_dir: str, field_name: str) -> List[SnapshotRef]:
    file_path = os.path.join(thc_dir, f"{field_name}.xz.h5")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Missing THC file: {file_path}")

    refs = []
    with h5py.File(file_path, "r") as h5f:
        for key in h5f.keys():
            match = THC_DATASET_RE.match(key)
            if not match:
                continue
            if match.group("field") != field_name:
                continue
            if int(match.group("rl")) != 0:
                continue
            dset = h5f[key]
            refs.append(
                SnapshotRef(
                    it=int(match.group("it")),
                    time=float(dset.attrs["time"]),
                    path=file_path,
                    dataset_key=key,
                )
            )
    refs.sort(key=lambda ref: (ref.time, ref.it))
    return refs


def read_thc_snapshot(snapshot: SnapshotRef) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(snapshot.path, "r") as h5f:
        dset = h5f[snapshot.dataset_key]
        data_xz = np.asarray(dset[()], dtype=np.float64).T
        origin = np.asarray(dset.attrs["origin"], dtype=np.float64)
        delta = np.asarray(dset.attrs["delta"], dtype=np.float64)
        x_src = origin[0] + np.arange(data_xz.shape[0], dtype=np.float64) * delta[0]
        z_src = origin[1] + np.arange(data_xz.shape[1], dtype=np.float64) * delta[1]
        return data_xz.T, x_src, z_src


def thc_snapshot_extent(snapshot: SnapshotRef) -> Tuple[float, float, float, float, float, float]:
    with h5py.File(snapshot.path, "r") as h5f:
        dset = h5f[snapshot.dataset_key]
        shape_raw = np.asarray(dset.shape, dtype=int)
        shape_xz = np.array([shape_raw[1], shape_raw[0]], dtype=int)
        origin = np.asarray(dset.attrs["origin"], dtype=np.float64)
        delta = np.asarray(dset.attrs["delta"], dtype=np.float64)
        x0 = origin[0]
        x1 = origin[0] + (shape_xz[0] - 1) * delta[0]
        z0 = origin[1]
        z1 = origin[1] + (shape_xz[1] - 1) * delta[1]
        return x0, x1, z0, z1, delta[0], delta[1]


def nux_component_name(spec: FieldSpec, species: int) -> str:
    return spec.nux_component.format(species=species)


def nux_mesh_pattern(spec: FieldSpec) -> re.Pattern:
    return re.compile(spec.nux_mesh_re)


def scan_nux_snapshots(nux_path: str, spec: FieldSpec, species: int) -> List[SnapshotRef]:
    component_name = nux_component_name(spec, species)
    files = gather_series_files(nux_path)
    if not files:
        raise RuntimeError(f"No BP series found under {nux_path}")

    refs = []
    for file_path in files:
        series = io.Series(file_path, io.Access.read_only)
        try:
            it_list = list(series.iterations)
            if not it_list:
                continue
            it = int(it_list[0])
            itobj = series.iterations[it]
            time_value = None
            for mesh_name in itobj.meshes:
                mesh = itobj.meshes[mesh_name]
                if component_name not in mesh:
                    continue
                try:
                    time_value = float(getattr(itobj, "time"))
                except Exception:
                    try:
                        time_value = float(itobj.get_attribute("time"))
                    except Exception:
                        time_value = None
                break
            if time_value is not None:
                refs.append(SnapshotRef(it=it, time=time_value, path=file_path))
        finally:
            series.close()

    refs.sort(key=lambda ref: (ref.time, ref.it))
    return refs


def infer_nux_y_plane(snapshot: SnapshotRef, spec: FieldSpec, species: int) -> float:
    component_name = nux_component_name(spec, species)
    pattern = nux_mesh_pattern(spec)
    series = io.Series(snapshot.path, io.Access.read_only)
    try:
        it = int(next(iter(series.iterations)))
        itobj = series.iterations[it]
        for mesh_name in itobj.meshes:
            if not pattern.match(mesh_name):
                continue
            mesh = itobj.meshes[mesh_name]
            if component_name not in mesh:
                continue
            comp = mesh[component_name]
            spacing = np.asarray(mesh.get_attribute("gridSpacing"), dtype=np.float64)
            offset = np.asarray(mesh.get_attribute("gridGlobalOffset"), dtype=np.float64)
            position = np.asarray(comp.get_attribute("position"), dtype=np.float64)
            return float(offset[1] + position[1] * spacing[1])
    finally:
        series.close()
    raise RuntimeError("Could not infer nuX y plane")


def nux_snapshot_extent(snapshot: SnapshotRef, spec: FieldSpec, species: int) -> Tuple[float, float, float, float, float, float]:
    component_name = nux_component_name(spec, species)
    pattern = nux_mesh_pattern(spec)
    series = io.Series(snapshot.path, io.Access.read_only)
    try:
        it = int(next(iter(series.iterations)))
        itobj = series.iterations[it]
        mins = []
        maxs = []
        dxs = []
        dzs = []
        for mesh_name in itobj.meshes:
            if not pattern.match(mesh_name):
                continue
            mesh = itobj.meshes[mesh_name]
            if component_name not in mesh:
                continue
            comp = mesh[component_name]
            shape = np.asarray(comp.shape, dtype=int)
            spacing = np.asarray(mesh.get_attribute("gridSpacing"), dtype=np.float64)
            offset = np.asarray(mesh.get_attribute("gridGlobalOffset"), dtype=np.float64)
            position = np.asarray(comp.get_attribute("position"), dtype=np.float64)
            z0 = offset[0] + position[0] * spacing[0]
            z1 = offset[0] + (shape[0] - 1 + position[0]) * spacing[0]
            x0 = offset[2] + position[2] * spacing[2]
            x1 = offset[2] + (shape[2] - 1 + position[2]) * spacing[2]
            mins.append((x0, z0))
            maxs.append((x1, z1))
            dxs.append(spacing[2])
            dzs.append(spacing[0])
        if not mins:
            raise RuntimeError("No matching nuX meshes found")
        x0 = min(v[0] for v in mins)
        z0 = min(v[1] for v in mins)
        x1 = max(v[0] for v in maxs)
        z1 = max(v[1] for v in maxs)
        return x0, x1, z0, z1, min(dxs), min(dzs)
    finally:
        series.close()


def nux_index_for_coordinate(offset, spacing, position, size, coordinate):
    idx = int(round((coordinate - offset) / spacing - position))
    return max(0, min(size - 1, idx))


def nux_index_range(offset, spacing, position, size, lower, upper, pad=1):
    if lower > upper:
        lower, upper = upper, lower
    start = int(np.floor((lower - offset) / spacing - position)) - pad
    stop = int(np.ceil((upper - offset) / spacing - position)) + pad + 1
    return max(0, start), min(size, stop)


def read_nux_snapshot(
    snapshot: SnapshotRef,
    spec: FieldSpec,
    species: int,
    y_coord: float,
    x_dst: np.ndarray,
    z_dst: np.ndarray,
) -> np.ndarray:
    component_name = nux_component_name(spec, species)
    pattern = nux_mesh_pattern(spec)
    canvas = np.full((len(z_dst), len(x_dst)), np.nan, dtype=np.float64)
    xmin = float(x_dst[0])
    xmax = float(x_dst[-1])
    zmin = float(z_dst[0])
    zmax = float(z_dst[-1])

    series = io.Series(snapshot.path, io.Access.read_only)
    try:
        it = int(next(iter(series.iterations)))
        itobj = series.iterations[it]

        keys = []
        for mesh_name in itobj.meshes:
            match = pattern.match(mesh_name)
            if not match:
                continue
            mesh = itobj.meshes[mesh_name]
            if component_name not in mesh:
                continue
            level = int(match.group(2))
            patch = int(match.group(1))
            keys.append((level, patch, mesh_name))
        keys.sort()

        for _level, _patch, mesh_name in keys:
            mesh = itobj.meshes[mesh_name]
            comp = mesh[component_name]
            shape = np.asarray(comp.shape, dtype=int)
            spacing = np.asarray(mesh.get_attribute("gridSpacing"), dtype=np.float64)
            offset = np.asarray(mesh.get_attribute("gridGlobalOffset"), dtype=np.float64)
            position = np.asarray(comp.get_attribute("position"), dtype=np.float64)

            z0, z1 = nux_index_range(offset[0], spacing[0], position[0], shape[0], zmin, zmax)
            x0, x1 = nux_index_range(offset[2], spacing[2], position[2], shape[2], xmin, xmax)
            y_idx = nux_index_for_coordinate(offset[1], spacing[1], position[1], shape[1], y_coord)

            patch_data = np.full((z1 - z0, x1 - x0), np.nan, dtype=np.float64)
            for chunk in comp.available_chunks():
                chunk_offset = np.asarray(chunk.offset, dtype=int)
                chunk_extent = np.asarray(chunk.extent, dtype=int)
                if not (chunk_offset[1] <= y_idx < chunk_offset[1] + chunk_extent[1]):
                    continue

                zz0 = max(z0, int(chunk_offset[0]))
                zz1 = min(z1, int(chunk_offset[0] + chunk_extent[0]))
                xx0 = max(x0, int(chunk_offset[2]))
                xx1 = min(x1, int(chunk_offset[2] + chunk_extent[2]))
                if zz1 <= zz0 or xx1 <= xx0:
                    continue

                slab = comp.load_chunk(
                    [zz0, y_idx, xx0],
                    [zz1 - zz0, 1, xx1 - xx0],
                )
                series.flush()
                patch_data[zz0 - z0:zz1 - z0, xx0 - x0:xx1 - x0] = (
                    np.asarray(slab, dtype=np.float64)[:, 0, :]
                )

            x_src = offset[2] + (np.arange(x0, x1, dtype=np.float64) + position[2]) * spacing[2]
            z_src = offset[0] + (np.arange(z0, z1, dtype=np.float64) + position[0]) * spacing[0]
            sampled = resample_rectilinear(patch_data, x_src, z_src, x_dst, z_dst)
            mask = np.isfinite(sampled)
            canvas[mask] = sampled[mask]
    finally:
        series.close()

    return canvas


def build_matched_pairs(
    thc_refs: List[SnapshotRef],
    nux_refs: List[SnapshotRef],
    target_time: Optional[float],
    every: int,
    time_min: Optional[float],
    time_max: Optional[float],
) -> List[Tuple[SnapshotRef, SnapshotRef]]:
    if not thc_refs or not nux_refs:
        return []

    if target_time is not None:
        nux_ref = min(nux_refs, key=lambda ref: abs(ref.time - target_time))
        thc_ref = min(thc_refs, key=lambda ref: abs(ref.time - nux_ref.time))
        return [(thc_ref, nux_ref)]

    pairs = []
    thc_tmin = thc_refs[0].time
    thc_tmax = thc_refs[-1].time
    for idx, nux_ref in enumerate(nux_refs):
        if idx % max(1, every) != 0:
            continue
        if time_min is not None and nux_ref.time < time_min:
            continue
        if time_max is not None and nux_ref.time > time_max:
            continue
        if nux_ref.time < thc_tmin or nux_ref.time > thc_tmax:
            continue
        thc_ref = min(thc_refs, key=lambda ref: abs(ref.time - nux_ref.time))
        pairs.append((thc_ref, nux_ref))
    return pairs


def compute_relative_difference(
    thc_data: np.ndarray,
    nux_data: np.ndarray,
    floor: float,
    signed: bool,
) -> np.ndarray:
    rel = np.full_like(thc_data, np.nan, dtype=np.float64)
    mask = np.isfinite(thc_data) & np.isfinite(nux_data)
    if not np.any(mask):
        return rel

    denom = np.maximum(np.abs(thc_data[mask]), floor)
    if signed:
        rel[mask] = (nux_data[mask] - thc_data[mask]) / denom
    else:
        rel[mask] = np.abs(nux_data[mask] - thc_data[mask]) / denom
    return rel


def choose_main_limits(spec: FieldSpec, args, values: np.ndarray) -> Tuple[float, float]:
    if args.vmin is not None and args.vmax is not None:
        return args.vmin, args.vmax

    if spec.default_vmin is not None and spec.default_vmax is not None:
        vmin = spec.default_vmin if args.vmin is None else args.vmin
        vmax = spec.default_vmax if args.vmax is None else args.vmax
        return vmin, vmax

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0

    vmin = float(np.nanmin(finite)) if args.vmin is None else args.vmin
    vmax = float(np.nanmax(finite)) if args.vmax is None else args.vmax
    if vmin < 0.0 < vmax:
        lim = max(abs(vmin), abs(vmax))
        vmin = -lim
        vmax = lim
    elif vmin >= 0.0 and args.vmin is None:
        vmin = 0.0
    if vmin == vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def choose_diff_vmax(rel_values: np.ndarray, args) -> float:
    if args.diff_vmax is not None:
        return args.diff_vmax
    finite = rel_values[np.isfinite(rel_values)]
    if finite.size == 0:
        return 1.0
    if args.signed_diff:
        return max(float(np.nanmax(np.abs(finite))), 1.0e-12)
    return float(np.nanmax(finite))


def frame_data(
    thc_ref: SnapshotRef,
    nux_ref: SnapshotRef,
    spec: FieldSpec,
    args,
    x_centers: np.ndarray,
    z_centers: np.ndarray,
    y_coord: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    thc_native, thc_x, thc_z = read_thc_snapshot(thc_ref)
    thc_data = resample_rectilinear(thc_native, thc_x, thc_z, x_centers, z_centers)
    nux_data = read_nux_snapshot(nux_ref, spec, args.species, y_coord, x_centers, z_centers)
    rel = compute_relative_difference(thc_data, nux_data, args.diff_floor, args.signed_diff)
    return thc_data, nux_data, rel


def estimate_limits(
    pairs: List[Tuple[SnapshotRef, SnapshotRef]],
    spec: FieldSpec,
    args,
    x_centers: np.ndarray,
    z_centers: np.ndarray,
    y_coord: float,
) -> Tuple[float, float, float]:
    main_chunks = []
    diff_chunks = []
    for thc_ref, nux_ref in pairs:
        thc_data, nux_data, rel = frame_data(thc_ref, nux_ref, spec, args, x_centers, z_centers, y_coord)
        main_chunks.append(thc_data[np.isfinite(thc_data)])
        main_chunks.append(nux_data[np.isfinite(nux_data)])
        if args.signed_diff:
            diff_chunks.append(rel[np.isfinite(rel)])
        else:
            rel_log = np.full_like(rel, np.nan, dtype=np.float64)
            mask = np.isfinite(rel)
            if np.any(mask):
                rel_log[mask] = np.log10(np.maximum(rel[mask], args.diff_floor))
            diff_chunks.append(rel_log[np.isfinite(rel_log)])

    main_vals = np.concatenate([v for v in main_chunks if v.size > 0]) if any(v.size > 0 for v in main_chunks) else np.array([])
    diff_vals = np.concatenate([v for v in diff_chunks if v.size > 0]) if any(v.size > 0 for v in diff_chunks) else np.array([])

    main_vmin, main_vmax = choose_main_limits(spec, args, main_vals)
    diff_vmax = choose_diff_vmax(diff_vals, args)
    return main_vmin, main_vmax, diff_vmax


def plot_three_panel(
    thc_data: np.ndarray,
    nux_data: np.ndarray,
    rel: np.ndarray,
    spec: FieldSpec,
    args,
    main_vmin: float,
    main_vmax: float,
    diff_vmax: float,
    thc_ref: SnapshotRef,
    nux_ref: SnapshotRef,
    y_coord: float,
    x_centers: np.ndarray,
    z_centers: np.ndarray,
):
    cmap = spec.default_cmap if args.cmap is None else args.cmap
    extent = extent_from_centers(x_centers, z_centers)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.2, 4.6),
        constrained_layout=True,
    )

    if main_vmin < 0.0 < main_vmax:
        main_norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=main_vmin, vmax=main_vmax)
    else:
        main_norm = mcolors.Normalize(vmin=main_vmin, vmax=main_vmax)

    if args.signed_diff:
        diff_norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-diff_vmax, vmax=diff_vmax)
        diff_cmap = "RdBu_r"
        rel_plot = rel
        diff_title = r"$(\mathrm{nuX}-\mathrm{THC}) / \max(|\mathrm{THC}|,\epsilon)$"
        diff_cbar_label = "relative difference"
    else:
        rel_plot = np.full_like(rel, np.nan, dtype=np.float64)
        mask = np.isfinite(rel)
        if np.any(mask):
            rel_plot[mask] = np.log10(np.maximum(rel[mask], args.diff_floor))
        diff_norm = mcolors.Normalize(vmin=np.log10(args.diff_floor), vmax=diff_vmax)
        diff_cmap = args.diff_cmap
        diff_title = r"$\log_{10}\left(|\mathrm{nuX}-\mathrm{THC}| / \max(|\mathrm{THC}|,\epsilon)\right)$"
        diff_cbar_label = r"$\log_{10}$ relative difference"

    im0 = axes[0].imshow(
        thc_data,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=main_norm,
        interpolation="none",
    )
    im1 = axes[1].imshow(
        nux_data,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=main_norm,
        interpolation="none",
    )
    im2 = axes[2].imshow(
        rel_plot,
        origin="lower",
        extent=extent,
        cmap=diff_cmap,
        norm=diff_norm,
        interpolation="none",
    )

    axes[0].set_title(f"THC {spec.label}[{args.species}]\nit={thc_ref.it}, t={thc_ref.time:.6g}")
    axes[1].set_title(f"nuX {spec.label}[{args.species}]\nit={nux_ref.it}, t={nux_ref.time:.6g}")
    axes[2].set_title(diff_title)

    for ax in axes:
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$z$")

    fig.colorbar(im0, ax=axes[0], pad=0.02, shrink=0.95, label=f"{spec.label}[{args.species}]")
    fig.colorbar(im1, ax=axes[1], pad=0.02, shrink=0.95, label=f"{spec.label}[{args.species}]")
    fig.colorbar(
        im2, ax=axes[2], pad=0.02, shrink=0.95, label=diff_cbar_label
    )
    return fig


def build_common_grid(
    thc_ref: SnapshotRef,
    nux_ref: SnapshotRef,
    spec: FieldSpec,
    args,
) -> Tuple[np.ndarray, np.ndarray]:
    tx0, tx1, tz0, tz1, tdx, tdz = thc_snapshot_extent(thc_ref)
    nx0, nx1, nz0, nz1, ndx, ndz = nux_snapshot_extent(nux_ref, spec, args.species)

    xmin = min(tx0, nx0) if args.xmin is None else args.xmin
    xmax = max(tx1, nx1) if args.xmax is None else args.xmax
    zmin = min(tz0, nz0) if args.zmin is None else args.zmin
    zmax = max(tz1, nz1) if args.zmax is None else args.zmax

    x_centers = centers_from_extent(xmin, xmax, args.nx, min(tdx, ndx))
    z_centers = centers_from_extent(zmin, zmax, args.nz, min(tdz, ndz))
    return x_centers, z_centers


def save_single_frame(
    pair: Tuple[SnapshotRef, SnapshotRef],
    spec: FieldSpec,
    args,
    x_centers: np.ndarray,
    z_centers: np.ndarray,
    y_coord: float,
    out_root: str,
):
    thc_ref, nux_ref = pair
    thc_data, nux_data, rel = frame_data(thc_ref, nux_ref, spec, args, x_centers, z_centers, y_coord)
    main_vmin, main_vmax, diff_vmax = estimate_limits([pair], spec, args, x_centers, z_centers, y_coord)
    fig = plot_three_panel(
        thc_data, nux_data, rel, spec, args,
        main_vmin, main_vmax, diff_vmax,
        thc_ref, nux_ref, y_coord, x_centers, z_centers,
    )
    os.makedirs(out_root, exist_ok=True)
    out_name = f"{make_output_stem(args, spec)}_t{nux_ref.time:.6f}.png"
    out_path = os.path.join(out_root, out_name)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


def save_movie(
    pairs: List[Tuple[SnapshotRef, SnapshotRef]],
    spec: FieldSpec,
    args,
    x_centers: np.ndarray,
    z_centers: np.ndarray,
    y_coord: float,
    out_root: str,
):
    stem = make_output_stem(args, spec)
    frames_dir = os.path.join(out_root, f"{stem}_frames")
    movie_path = os.path.join(out_root, f"{stem}.mp4")
    os.makedirs(frames_dir, exist_ok=True)

    main_vmin, main_vmax, diff_vmax = estimate_limits(pairs, spec, args, x_centers, z_centers, y_coord)

    frame_paths = []
    for idx, (thc_ref, nux_ref) in enumerate(pairs):
        thc_data, nux_data, rel = frame_data(thc_ref, nux_ref, spec, args, x_centers, z_centers, y_coord)
        fig = plot_three_panel(
            thc_data, nux_data, rel, spec, args,
            main_vmin, main_vmax, diff_vmax,
            thc_ref, nux_ref, y_coord, x_centers, z_centers,
        )
        frame_path = os.path.join(frames_dir, f"frame_{idx:05d}.png")
        fig.savefig(frame_path, dpi=args.dpi)
        plt.close(fig)
        frame_paths.append(frame_path)
        print(f"[{idx + 1}/{len(pairs)}] {frame_path}")

    with imageio.get_writer(movie_path, fps=args.fps) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))

    print(f"Saved {movie_path}")


def run_for_field(args, spec: FieldSpec):
    out_root = os.path.abspath(os.path.expanduser(args.out_root))

    thc_field = spec.thc_component.format(species=args.species)
    thc_refs = scan_thc_snapshots(args.thc_dir, thc_field)
    print(f"Scanning {spec.cli_name}[{args.species}]")
    nux_refs = scan_nux_snapshots(args.nux_path, spec, args.species)
    pairs = build_matched_pairs(
        thc_refs, nux_refs,
        target_time=args.time,
        every=args.every,
        time_min=args.time_min,
        time_max=args.time_max,
    )
    if not pairs:
        raise RuntimeError("No matched THC/nuX frames found")

    y_coord = args.nux_y
    if y_coord is None:
        y_coord = infer_nux_y_plane(pairs[0][1], spec, args.species)

    x_centers, z_centers = build_common_grid(pairs[0][0], pairs[0][1], spec, args)

    if args.time is not None:
        save_single_frame(pairs[0], spec, args, x_centers, z_centers, y_coord, out_root)
    else:
        save_movie(pairs, spec, args, x_centers, z_centers, y_coord, out_root)


def main():
    args = parse_args()
    specs = resolve_requested_fields(args)
    for spec in specs:
        field_args = args_for_field(args, spec.cli_name)
        run_for_field(field_args, spec)


if __name__ == "__main__":
    main()
