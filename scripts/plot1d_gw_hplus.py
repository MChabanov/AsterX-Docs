#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plot1d_common import (
    TIME_CU_TO_MS,
    add_shared_arguments,
    gather_output_data_dirs,
    plt,
    prepare_output_dir,
    radius_to_tag,
    style_1d_axis,
)


DEFAULT_RADII = [100.0, 150.0, 200.0, 250.0, 300.0, 400.0]


def build_radius_colors(radii: list[float]) -> dict[float, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("viridis")
    if len(radii) == 1:
        return {radii[0]: cmap(0.6)}
    return {
        radius: cmap(0.12 + 0.76 * index / (len(radii) - 1))
        for index, radius in enumerate(radii)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct R_det h+ from Psi4 l=2,m=2 files and make 1D plots."
    )
    add_shared_arguments(parser)
    parser.add_argument(
        "--radii",
        nargs="+",
        type=float,
        default=DEFAULT_RADII,
        help="Extraction radii to reconstruct and plot.",
    )
    parser.add_argument(
        "--reference-radius",
        type=float,
        default=None,
        help="Radius used to estimate the FFI cutoff. Defaults to the largest requested radius.",
    )
    parser.add_argument(
        "--w0",
        type=float,
        default=None,
        help="Override the FFI cutoff instead of estimating it from the reference radius.",
    )
    parser.add_argument(
        "--ffi-order",
        type=int,
        default=2,
        help="Number of time integrations to apply in the frequency domain.",
    )
    parser.add_argument(
        "--taper-alpha",
        type=float,
        default=0.1,
        help="Tukey window alpha for the pre-FFT taper. Set to 0 to disable tapering.",
    )
    return parser.parse_args()


def _time_key(value: float) -> float:
    return round(float(value), 10)


def stitch_radius(data_dirs: list[Path], radius: float) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    file_name = f"mp_NP_Psi4_l2_m2_r{radius:.2f}.tsv"
    files_used = []
    combined = {}

    for data_dir in data_dirs:
        file_path = data_dir / file_name
        if not file_path.is_file():
            continue

        files_used.append(file_path)
        data = np.loadtxt(file_path)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 3:
            raise ValueError(f"Expected at least 3 columns in {file_path}")

        for t, re_part, im_part in data[:, :3]:
            key = _time_key(t)
            new_is_zero = re_part == 0.0 and im_part == 0.0
            existing = combined.get(key)
            if existing is None:
                combined[key] = (float(t), float(re_part), float(im_part))
                continue

            _old_t, old_re, old_im = existing
            old_is_zero = old_re == 0.0 and old_im == 0.0
            if old_is_zero and not new_is_zero:
                combined[key] = (float(t), float(re_part), float(im_part))

    if not files_used:
        raise FileNotFoundError(f"No Psi4 files found for r={radius:.2f}")

    stitched = np.array([combined[key] for key in sorted(combined)], dtype=float)
    time_cu = stitched[:, 0]
    psi4 = stitched[:, 1] + 1j * stitched[:, 2]
    return time_cu, psi4, files_used


def estimate_w0_from_phase(time_cu: np.ndarray, psi4: np.ndarray) -> float:
    amplitude = np.abs(psi4)
    if amplitude.size == 0 or np.all(amplitude <= 0.0):
        return 0.02

    mask = amplitude > 0.02 * amplitude.max()
    if np.count_nonzero(mask) < 20:
        return 0.02

    time_sel = time_cu[mask]
    psi4_sel = psi4[mask]
    phase = np.unwrap(np.angle(psi4_sel))
    omega = np.abs(np.gradient(phase, time_sel))
    if omega.size == 0:
        return 0.02

    n_take = min(80, len(omega))
    omega0 = float(np.median(omega[:n_take]))
    return max(0.01, min(0.03, omega0))


def tukey_window(size: int, alpha: float = 0.1) -> np.ndarray:
    if alpha <= 0.0:
        return np.ones(size)
    if alpha >= 1.0:
        return np.hanning(size)

    coordinate = np.linspace(0.0, 1.0, size)
    window = np.ones(size)
    left = coordinate < alpha / 2.0
    right = coordinate >= (1.0 - alpha / 2.0)

    window[left] = 0.5 * (
        1.0 + np.cos(2.0 * np.pi / alpha * (coordinate[left] - alpha / 2.0))
    )
    window[right] = 0.5 * (
        1.0 + np.cos(2.0 * np.pi / alpha * (coordinate[right] - 1.0 + alpha / 2.0))
    )
    return window


def integrate_ffi(
    time_cu: np.ndarray,
    psi4: np.ndarray,
    w0: float,
    order: int = 2,
    taper_alpha: float = 0.1,
) -> np.ndarray:
    if len(time_cu) < 2:
        raise ValueError("Need at least two samples for FFI integration")

    dt = float(np.median(np.diff(time_cu)))
    tapered = psi4.copy()
    if taper_alpha > 0.0:
        tapered = tapered * tukey_window(len(tapered), alpha=taper_alpha)

    spectrum = np.fft.fft(tapered)
    angular_freq = np.fft.fftfreq(len(time_cu), d=dt) * (2.0 * np.pi)
    abs_freq = np.abs(angular_freq)
    cutoff = np.where(abs_freq > w0, abs_freq, w0)
    factor = (-1j * np.sign(angular_freq) / cutoff) ** int(order)
    return np.fft.ifft(spectrum * factor)


def main() -> None:
    args = parse_args()
    _sim_root, sim_name, data_dirs = gather_output_data_dirs(args.sim_dir)
    _sim_root, _sim_name, out_dir = prepare_output_dir(args.sim_dir, "gw_hplus", args.out_root)

    plot_radii = sorted({float(radius) for radius in args.radii})
    if not plot_radii:
        raise ValueError("At least one extraction radius is required")

    reference_radius = (
        float(args.reference_radius) if args.reference_radius is not None else max(plot_radii)
    )

    time_ref, psi4_ref, _files_ref = stitch_radius(data_dirs, reference_radius)
    w0 = float(args.w0) if args.w0 is not None else estimate_w0_from_phase(time_ref, psi4_ref)

    waveforms = {}
    source_merger_times = []
    for radius in plot_radii:
        time_cu, psi4, files_used = stitch_radius(data_dirs, radius)
        strain = integrate_ffi(
            time_cu,
            psi4,
            w0=w0,
            order=args.ffi_order,
            taper_alpha=args.taper_alpha,
        )
        h_plus = strain.real
        scaled_h_plus = radius * h_plus
        observed_peak_time = time_cu[np.argmax(np.abs(scaled_h_plus))]
        source_merger_time = observed_peak_time - radius
        source_merger_times.append(source_merger_time)
        waveforms[radius] = {
            "time_cu": time_cu,
            "scaled_h_plus": scaled_h_plus,
            "observed_peak_time": observed_peak_time,
            "source_merger_time": source_merger_time,
            "files_used": files_used,
        }

    common_source_merger_time = float(np.median(source_merger_times))
    saved_paths = []
    summary_lines = []
    radius_colors = build_radius_colors(plot_radii)

    for radius in plot_radii:
        time_cu = waveforms[radius]["time_cu"]
        scaled_h_plus = waveforms[radius]["scaled_h_plus"]
        time_ms = (time_cu - common_source_merger_time - radius) * TIME_CU_TO_MS
        radius_tag = radius_to_tag(radius)

        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        ax.axhline(0.0, color="#94a3b8", linewidth=0.9, alpha=0.7, zorder=0)
        ax.plot(
            time_ms,
            scaled_h_plus,
            linewidth=1.6,
            color=radius_colors[radius],
            label=rf"$r = {radius_tag}$",
        )
        ax.set_xlabel(r"$t - t_{\rm merge} - R_{\rm det}/c$ [ms]")
        ax.set_ylabel(r"$R_{\rm det}\, h_{+}$")
        ax.legend(loc="best")
        style_1d_axis(ax)
        fig.tight_layout()

        plot_path = out_dir / f"{sim_name}_Rdet_hplus_r{radius_tag}.png"
        fig.savefig(plot_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        saved_paths.append(plot_path)
        summary_lines.append(
            "r={radius}: {samples} samples, files={files}, x-range=[{xmin:.3f}, {xmax:.3f}] ms".format(
                radius=radius_tag,
                samples=len(time_cu),
                files=len(waveforms[radius]["files_used"]),
                xmin=float(time_ms.min()),
                xmax=float(time_ms.max()),
            )
        )

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    ax.axhline(0.0, color="#94a3b8", linewidth=0.9, alpha=0.7, zorder=0)
    for radius in plot_radii:
        time_cu = waveforms[radius]["time_cu"]
        scaled_h_plus = waveforms[radius]["scaled_h_plus"]
        time_ms = (time_cu - common_source_merger_time - radius) * TIME_CU_TO_MS
        ax.plot(
            time_ms,
            scaled_h_plus,
            linewidth=1.5,
            color=radius_colors[radius],
            label=rf"$r = {radius_to_tag(radius)}$",
        )

    ax.set_xlabel(r"$t - t_{\rm merge} - R_{\rm det}/c$ [ms]")
    ax.set_ylabel(r"$R_{\rm det}\, h_{+}$")
    ax.legend(loc="best")
    style_1d_axis(ax)
    fig.tight_layout()

    overlay_path = out_dir / f"{sim_name}_Rdet_hplus_overlay.png"
    fig.savefig(overlay_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(overlay_path)

    summary_path = out_dir / "README_summary.txt"
    with summary_path.open("w", encoding="utf-8") as stream:
        stream.write("R_det h_plus reconstruction from Psi4 l=2,m=2\n")
        stream.write(f"Simulation: {sim_name}\n")
        stream.write(f"FFI cutoff w0 = {w0:.6f}\n")
        stream.write(f"FFI order = {args.ffi_order}\n")
        stream.write(f"Tukey taper alpha = {args.taper_alpha:.3f}\n")
        stream.write(f"Reference radius for w0 = {radius_to_tag(reference_radius)}\n")
        stream.write(f"Median source-frame merger time = {common_source_merger_time:.6f} code units\n")
        stream.write(f"Time conversion: ms = code_time * {TIME_CU_TO_MS:.16f}\n\n")
        for line in summary_lines:
            stream.write(f"{line}\n")
    saved_paths.append(summary_path)

    print(f"Simulation: {sim_name}")
    print(f"Estimated FFI cutoff w0 = {w0:.6f}")
    print(f"Median source-frame merger time = {common_source_merger_time:.6f} code units")
    print("Created files:")
    for path in saved_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()
