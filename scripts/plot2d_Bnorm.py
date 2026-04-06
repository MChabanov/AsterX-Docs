#!/usr/bin/env python3
from plot2d_field_movie import FieldConfig, B_CU_TO_GAUSS, run_field_movie


CONFIG = FieldConfig(
    field_name="Bnorm",
    record_component="asterx_b_norm",
    mesh_name_re=r"^asterx_diagnostics_patch(\d+)_lev(\d+)$",
    colorbar_label=r"$||B||$ [G]",
    cmap_name="magma",
    scale_to_cgs=B_CU_TO_GAUSS,
    norm="log",
    vmin=1e8,
    vmax=5e16,
)


if __name__ == "__main__":
    run_field_movie(CONFIG)
