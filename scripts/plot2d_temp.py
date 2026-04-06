#!/usr/bin/env python3
from plot2d_field_movie import FieldConfig, run_field_movie


CONFIG = FieldConfig(
    field_name="temp",
    record_component="hydrobasex_temperature",
    mesh_name_re=r"^hydrobasex_temperature_patch(\d+)_lev(\d+)$",
    colorbar_label=r"$T$ [MeV]",
    cmap_name="hot",
    scale_to_cgs=1.0,
    norm="log",
    vmin=1.0,
    vmax=100.0,
    valid_min=0.0,
)


if __name__ == "__main__":
    run_field_movie(CONFIG)
