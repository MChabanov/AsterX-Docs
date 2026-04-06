#!/usr/bin/env python3
from plot2d_field_movie import FieldConfig, RHO_CU_TO_CGS, run_field_movie


CONFIG = FieldConfig(
    field_name="rho",
    record_component="hydrobasex_rho",
    mesh_name_re=r"^hydrobasex_rho_patch(\d+)_lev(\d+)$",
    colorbar_label=r"$\rho$ [g/cm$^3$]",
    cmap_name="plasma",
    scale_to_cgs=RHO_CU_TO_CGS,
    norm="log",
    vmin=5e-15 * RHO_CU_TO_CGS,
    vmax=1e-3 * RHO_CU_TO_CGS,
)


if __name__ == "__main__":
    run_field_movie(CONFIG)
