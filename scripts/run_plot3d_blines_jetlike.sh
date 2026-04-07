#!/usr/bin/env bash
set -euo pipefail

# Run the 3D magnetic-field-line movie with all plot3d_blines.py parser options
# exposed here as shell variables. Edit the values below, then run this script.
# Extra CLI arguments passed to this wrapper are appended at the end, so they
# override any duplicated options defined here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # Make the `module` command available in non-interactive shells on the cluster.
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

if command -v module >/dev/null 2>&1; then
  module load adios2
fi

# ----------------------------- paths ---------------------------------
SIM_DIR="/scratch/09228/jkalinan/simulations/AsterX_BNS_APRLDP_RPA_8lvl_sc_fixedGrid_dx025"
OUT_ROOT="${SCRIPT_DIR}/plots/3d"
PYVISTA_ROOT="${SCRIPT_DIR}/pyvista"

# ------------------------- display scale -----------------------------
# Multiplies image resolution and pixel-based annotation sizes so the saved
# render keeps the same visual balance at higher output resolution.
DISPLAY_SCALE=2.5

# ------------------------- frame selection ---------------------------
CADENCE=3
MAX_FRAMES=""
GRID_SIZE=288
WIDTH=1088
HEIGHT=1088
FPS=6
SPIN_DEG=0.0
FAN_DEG=0.0
CAMERA_ZOOM=1.0
NO_SNAPSHOTS=false
THEME="dark"

# -------------------------- colorbars --------------------------------
SHOW_RHO_COLORBAR=true
NO_B_COLORBAR=false
NO_COLORBAR=false
RHO_CMAP="inferno"
B_CMAP="viridis"
RHO_COLORBAR_X=0.89
B_COLORBAR_X=0.89
COLORBAR_Y=0.36
RHO_COLORBAR_Y=0.56
B_COLORBAR_Y=0.16
COLORBAR_WIDTH=0.024
COLORBAR_HEIGHT=0.28
COLORBAR_TITLE_SIZE=22
COLORBAR_LABEL_SIZE=22
TIME_LABEL_FONT_SIZE=22
AXES_LINE_WIDTH=3.0

# --------------------------- domain ----------------------------------
BBOX_HALF_WIDTH_CU=150
AUTO_BBOX=false

# -------------------------- rho context ------------------------------
RHO_LOG_MIN=8
RHO_LOG_MAX=14.0
CONTOUR_COUNT=10
CONTOUR_LOW_FRAC=0.45
CONTOUR_OPACITY=0.25

# -------------------------- B coloring -------------------------------
B_LOG_MIN=13
B_LOG_MAX=16

# ---------------------------- seeds ----------------------------------
SEED_MODE="both"
SEED_STRENGTH_FRAC=1e-3
SEED_RHO_MIN_CGS=1e8
MAX_SEEDS=192
REMNANT_SEEDS=96
FUNNEL_SEEDS=96
MIN_SEED_SEPARATION_KM=2.5
FUNNEL_THETA_MAX_DEG=15
FUNNEL_RHO_MAX_CGS=5e12
FUNNEL_MIN_ABS_Z_KM=20
FUNNEL_AZIMUTH_BINS=16

# ------------------------- streamlines -------------------------------
STREAMLINE_MAX_LENGTH_KM=800
STREAMLINE_INITIAL_STEP=0.5
STREAMLINE_TERMINAL_SPEED=5e-8
STREAMLINE_MAX_STEPS=8000
STREAMLINE_BATCH_SIZE=32
STREAMLINE_STYLE="line"
STREAMLINE_COLOR_MODE="scalar"
STREAMLINE_SOLID_COLOR="white"
STREAMLINE_LINE_WIDTH=0.8
STREAMLINE_OPACITY=0.6
STREAMLINE_TUBE_RADIUS_KM=0.14
STREAMLINE_TUBE_SIDES=8

# ---------------------- stopping / timing ----------------------------
FINAL_AFTER_MS=""
MERGER_TIME_MS=15.0

# --------------------- post-merger filtering -------------------------
POST_MERGER_MODE="current"
JETLIKE_KEEP_CURVES=96
JETLIKE_THETA_MAX_DEG=20
JETLIKE_MIN_RADIUS_KM=80
JETLIKE_MIN_ABS_Z_KM=20
JETLIKE_START_DELAY_MS=12
JETLIKE_MIN_KEEP_CURVES=24
JETLIKE_MIN_CURVE_LENGTH_KM=40

scale_float() {
  awk -v value="$1" -v scale="${DISPLAY_SCALE}" 'BEGIN { printf "%.12g", value * scale }'
}

scale_int() {
  awk -v value="$1" -v scale="${DISPLAY_SCALE}" 'BEGIN { printf "%d", int(value * scale + 0.5) }'
}

# Viewport-normalized sizes like COLORBAR_WIDTH and COLORBAR_HEIGHT already
# scale with image resolution, so only pixel-based controls need explicit
# scaling here.
SCALED_WIDTH="$(scale_int "${WIDTH}")"
SCALED_HEIGHT="$(scale_int "${HEIGHT}")"
SCALED_COLORBAR_TITLE_SIZE="$(scale_int "${COLORBAR_TITLE_SIZE}")"
SCALED_COLORBAR_LABEL_SIZE="$(scale_int "${COLORBAR_LABEL_SIZE}")"
SCALED_TIME_LABEL_FONT_SIZE="$(scale_int "${TIME_LABEL_FONT_SIZE}")"
SCALED_AXES_LINE_WIDTH="$(scale_float "${AXES_LINE_WIDTH}")"
SCALED_STREAMLINE_LINE_WIDTH="$(scale_float "${STREAMLINE_LINE_WIDTH}")"

cmd=(python3 plot3d_blines.py)

if [[ -n "${SIM_DIR}" ]]; then
  cmd+=("${SIM_DIR}")
fi

cmd+=(
  --out-root "${OUT_ROOT}"
  --cadence "${CADENCE}"
  --grid-size "${GRID_SIZE}"
  --width "${SCALED_WIDTH}"
  --height "${SCALED_HEIGHT}"
  --fps "${FPS}"
  --spin-deg "${SPIN_DEG}"
  --fan-deg "${FAN_DEG}"
  --camera-zoom "${CAMERA_ZOOM}"
  --time-label-font-size "${SCALED_TIME_LABEL_FONT_SIZE}"
  --axes-line-width "${SCALED_AXES_LINE_WIDTH}"
  --theme "${THEME}"
  --rho-cmap "${RHO_CMAP}"
  --b-cmap "${B_CMAP}"
  --rho-colorbar-x "${RHO_COLORBAR_X}"
  --b-colorbar-x "${B_COLORBAR_X}"
  --colorbar-y "${COLORBAR_Y}"
  --rho-colorbar-y "${RHO_COLORBAR_Y}"
  --b-colorbar-y "${B_COLORBAR_Y}"
  --colorbar-width "${COLORBAR_WIDTH}"
  --colorbar-height "${COLORBAR_HEIGHT}"
  --colorbar-title-size "${SCALED_COLORBAR_TITLE_SIZE}"
  --colorbar-label-size "${SCALED_COLORBAR_LABEL_SIZE}"
  --bbox-half-width-cu "${BBOX_HALF_WIDTH_CU}"
  --rho-log-min "${RHO_LOG_MIN}"
  --rho-log-max "${RHO_LOG_MAX}"
  --contour-count "${CONTOUR_COUNT}"
  --contour-low-frac "${CONTOUR_LOW_FRAC}"
  --contour-opacity "${CONTOUR_OPACITY}"
  --b-log-min "${B_LOG_MIN}"
  --b-log-max "${B_LOG_MAX}"
  --seed-mode "${SEED_MODE}"
  --seed-strength-frac "${SEED_STRENGTH_FRAC}"
  --seed-rho-min-cgs "${SEED_RHO_MIN_CGS}"
  --max-seeds "${MAX_SEEDS}"
  --remnant-seeds "${REMNANT_SEEDS}"
  --funnel-seeds "${FUNNEL_SEEDS}"
  --min-seed-separation-km "${MIN_SEED_SEPARATION_KM}"
  --funnel-theta-max-deg "${FUNNEL_THETA_MAX_DEG}"
  --funnel-rho-max-cgs "${FUNNEL_RHO_MAX_CGS}"
  --funnel-min-abs-z-km "${FUNNEL_MIN_ABS_Z_KM}"
  --funnel-azimuth-bins "${FUNNEL_AZIMUTH_BINS}"
  --streamline-max-length-km "${STREAMLINE_MAX_LENGTH_KM}"
  --streamline-initial-step "${STREAMLINE_INITIAL_STEP}"
  --streamline-terminal-speed "${STREAMLINE_TERMINAL_SPEED}"
  --streamline-max-steps "${STREAMLINE_MAX_STEPS}"
  --streamline-batch-size "${STREAMLINE_BATCH_SIZE}"
  --streamline-style "${STREAMLINE_STYLE}"
  --streamline-color-mode "${STREAMLINE_COLOR_MODE}"
  --streamline-solid-color "${STREAMLINE_SOLID_COLOR}"
  --streamline-line-width "${SCALED_STREAMLINE_LINE_WIDTH}"
  --streamline-opacity "${STREAMLINE_OPACITY}"
  --streamline-tube-radius-km "${STREAMLINE_TUBE_RADIUS_KM}"
  --streamline-tube-sides "${STREAMLINE_TUBE_SIDES}"
  --pyvista-root "${PYVISTA_ROOT}"
  --merger-time-ms "${MERGER_TIME_MS}"
  --post-merger-mode "${POST_MERGER_MODE}"
  --jetlike-keep-curves "${JETLIKE_KEEP_CURVES}"
  --jetlike-theta-max-deg "${JETLIKE_THETA_MAX_DEG}"
  --jetlike-min-radius-km "${JETLIKE_MIN_RADIUS_KM}"
  --jetlike-min-abs-z-km "${JETLIKE_MIN_ABS_Z_KM}"
  --jetlike-start-delay-ms "${JETLIKE_START_DELAY_MS}"
  --jetlike-min-keep-curves "${JETLIKE_MIN_KEEP_CURVES}"
  --jetlike-min-curve-length-km "${JETLIKE_MIN_CURVE_LENGTH_KM}"
)

if [[ -n "${MAX_FRAMES}" ]]; then
  cmd+=(--max-frames "${MAX_FRAMES}")
fi

if [[ -n "${FINAL_AFTER_MS}" ]]; then
  cmd+=(--final-after-ms "${FINAL_AFTER_MS}")
fi

if [[ "${NO_SNAPSHOTS}" == true ]]; then
  cmd+=(--no-snapshots)
fi

if [[ "${SHOW_RHO_COLORBAR}" == true ]]; then
  cmd+=(--show-rho-colorbar)
fi

if [[ "${NO_B_COLORBAR}" == true ]]; then
  cmd+=(--no-b-colorbar)
fi

if [[ "${NO_COLORBAR}" == true ]]; then
  cmd+=(--no-colorbar)
fi

if [[ "${AUTO_BBOX}" == true ]]; then
  cmd+=(--auto-bbox)
fi

cmd+=("$@")

"${cmd[@]}"
