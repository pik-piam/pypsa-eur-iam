"""
Read sectoral electricity demand from REMIND and export it for use in PyPSA-Eur.

Stage 1 of the demand pipeline. Reads regional sectoral electricity demand via the
``Coupler`` (backend-selected by ``RemindLoader``):

- GDX backend: reads ``load_sector`` symbol (``v32_load_sector`` / ``p32_load_sector``),
  converts TWa→MWh via the symbol spec.
- IAMC backend: derives demand from SE|Electricity, transmission losses, and FE sector
  variables, applying an implicit T&D efficiency and computing an AC residual for
  untracked loads.

``downscale_REMIND_demand`` splits this to countries (Stage 2, backend-agnostic).
"""

import logging

from _helpers import configure_logging, mock_snakemake
from scripts.remind._remind_helpers import build_remind_coupler

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "import_REMIND_demand",
            snakefile_choices=["Snakefile_REMIND"],
            scen_REMIND="TEST_multiregion",
            iter_REMIND="1",
            configfiles="config/config.remind_multiregion.yaml",
        )

    configure_logging(snakemake)
    logger.info("Loading REMIND regional demand ...")

    countries = set(snakemake.params["countries"])
    coupler = build_remind_coupler(snakemake.input["remind_data"], countries)
    demand = coupler.build_regional_demand()

    # Drop demand_h2 (different unit, should not get folded into AC)
    demand = demand[demand["sector"] != "demand_h2"]

    years = snakemake.params["years"]
    demand = demand[demand["year"].isin(years) & demand["region"].isin(coupler.model_regions)]

    demand.to_csv(snakemake.output["sectoral_load"], index=False)
    logger.info(
        "Wrote %d rows of REMIND demand (%s backend) to %s",
        len(demand),
        coupler.loader.backend,
        snakemake.output["sectoral_load"],
    )
