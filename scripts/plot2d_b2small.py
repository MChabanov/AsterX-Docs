#!/usr/bin/env python3
from plot2d_field_movie import FieldConfig, PRESSURE_CU_TO_CGS, run_field_movie


CONFIG = FieldConfig(
    field_name="b2small",
    record_component="asterx_b2small",
    mesh_name_re=r"^asterx_diagnostics_patch(\d+)_lev(\d+)$",
    colorbar_label=r"$b^2$ [erg/cm$^3$]",
    cmap_name="inferno",
    scale_to_cgs=PRESSURE_CU_TO_CGS,
    norm="log",
    vmin=1e20,
    vmax=1e34,
)


if __name__ == "__main__":
    run_field_movie(CONFIG)
