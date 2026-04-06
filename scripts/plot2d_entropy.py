#!/usr/bin/env python3
from plot2d_field_movie import FieldConfig, run_field_movie


CONFIG = FieldConfig(
    field_name="entropy",
    record_component="hydrobasex_entropy",
    mesh_name_re=r"^hydrobasex_entropy_patch(\d+)_lev(\d+)$",
    colorbar_label=r"$s$",
    cmap_name="turbo",
    scale_to_cgs=1.0,
    norm="linear",
    vmin=0.0,
    vmax=20.0,
    valid_min=0.0,
    valid_max=50.0,
    under_color=None,
)


if __name__ == "__main__":
    run_field_movie(CONFIG)
