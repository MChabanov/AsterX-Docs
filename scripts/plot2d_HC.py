#!/usr/bin/env python3
from plot2d_field_movie import FieldConfig, run_field_movie


CONFIG = FieldConfig(
    field_name="HC",
    record_component="z4c_hc",
    mesh_name_re=r"^z4c_hc_patch(\d+)_lev(\d+)$",
    colorbar_label=r"$\mathrm{H}$",
    cmap_name="RdBu_r",
    scale_to_cgs=1.0,
    norm="symlog",
    vmin=-2e-3,
    vmax=2e-3,
    linthresh=1e-9,
    under_color=None,
)


if __name__ == "__main__":
    run_field_movie(CONFIG)
