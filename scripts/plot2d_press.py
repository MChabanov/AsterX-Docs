#!/usr/bin/env python3
from plot2d_field_movie import FieldConfig, PRESSURE_CU_TO_CGS, run_field_movie


CONFIG = FieldConfig(
    field_name="press",
    record_component="hydrobasex_press",
    mesh_name_re=r"^hydrobasex_press_patch(\d+)_lev(\d+)$",
    colorbar_label=r"$P$ [dyn/cm$^2$]",
    cmap_name="plasma",
    scale_to_cgs=PRESSURE_CU_TO_CGS,
    norm="log",
    vmin=1e27,
    vmax=1e36,
)


if __name__ == "__main__":
    run_field_movie(CONFIG)
