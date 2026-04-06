#!/usr/bin/env python3
from plot2d_field_movie import FieldConfig, run_field_movie


CONFIG = FieldConfig(
    field_name="con2prim_flag",
    record_component="asterx_con2prim_flag",
    mesh_name_re=r"^asterx_con2prim_flag_patch(\d+)_lev(\d+)$",
    colorbar_label="Con2Prim Flag",
    cmap_name="tab10",
    scale_to_cgs=1.0,
    norm="discrete",
    vmin=0.0,
    vmax=8.0,
    boundaries=(-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5),
    valid_min=0.0,
    valid_max=8.0,
    under_color=None,
)


if __name__ == "__main__":
    run_field_movie(CONFIG)
