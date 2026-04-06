#!/usr/bin/env python3
from plot2d_field_movie import FieldConfig, EPS_CU_TO_ERG_PER_G, run_field_movie


CONFIG = FieldConfig(
    field_name="eps",
    record_component="hydrobasex_eps",
    mesh_name_re=r"^hydrobasex_eps_patch(\d+)_lev(\d+)$",
    colorbar_label=r"$\epsilon$ [erg/g]",
    cmap_name="viridis",
    scale_to_cgs=EPS_CU_TO_ERG_PER_G,
    norm="log",
    #vmin=1e18,
    vmin=1e15,
    #vmax=2e20,
    vmax=1e19,
)


if __name__ == "__main__":
    run_field_movie(CONFIG)
