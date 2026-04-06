#!/usr/bin/env python3
from plot2d_field_movie import FieldConfig, run_field_movie


CONFIG = FieldConfig(
    field_name="Ye",
    record_component="hydrobasex_ye",
    mesh_name_re=r"^hydrobasex_ye_patch(\d+)_lev(\d+)$",
    colorbar_label=r"$Y_e$",
    cmap_name="cividis",
    scale_to_cgs=1.0,
    norm="linear",
    vmin=0.0,
    vmax=0.15,
    valid_min=0.0,
    valid_max=1.0,
    under_color=None,
)


if __name__ == "__main__":
    run_field_movie(CONFIG)
