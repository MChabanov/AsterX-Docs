#!/usr/bin/env python3
from plot2d_field_movie import FieldConfig, run_field_movie


CONFIG = FieldConfig(
    field_name="w_lorentz",
    record_component="asterx_w_lorentz",
    mesh_name_re=r"^asterx_diagnostics_patch(\d+)_lev(\d+)$",
    colorbar_label=r"$W$",
    cmap_name="viridis",
    scale_to_cgs=1.0,
    norm="linear",
    vmin=1.0,
    vmax=1.3,
    valid_min=1.0,
    valid_max=10.0,
    under_color=None,
)


if __name__ == "__main__":
    run_field_movie(CONFIG)
