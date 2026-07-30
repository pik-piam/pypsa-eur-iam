"""
Read installed-capacity targets from REMIND and export them as PyPSA-Eur lower bounds.

Thin wrapper over ``Coupler.build_capacity_targets``: reads the capacity
spec (unit conversion handled by the spec), applies any ``consolidation`` block declared on the
spec (VRE-variant merge, battery scaling, link output→input adjustment), and maps model tech
tokens to PyPSA-Eur carriers.

Output columns: [year, region_REMIND, carrier, value, unit]
  - value: input-basis capacity in MW (link-like techs already divided by efficiency)
  - unit: "MW" for all rows
"""

import logging

from _helpers import configure_logging, mock_snakemake
from scripts.remind._remind_helpers import build_capacity_technology_mapping, build_remind_coupler

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "import_REMIND_capacities",
            snakefile_choices=["Snakefile_REMIND"],
            scen_REMIND="TEST_multiregion",
            iter_REMIND="1",
            configfiles="config/config.remind_multiregion.yaml",
        )

    configure_logging(snakemake)
    logger.info("Building REMIND capacity targets via Coupler.build_capacity_targets")

    countries = set(snakemake.params["countries"])
    tech_map = build_capacity_technology_mapping(snakemake.input["technology_mapping"])

    years = snakemake.params["years"]

    coupler = build_remind_coupler(snakemake.input["remind_data"], countries)
    mapped_regions = coupler.model_regions
    capacities = coupler.build_capacity_targets(
        tech_map, map_tech_col="REMIND-EU", map_carrier_col="PyPSA-Eur",
    )
    capacities = (
        capacities.rename(columns={"region": "region_REMIND"})[
            ["year", "region_REMIND", "carrier", "value", "unit"]
        ]
        .query("year in @years and region_REMIND in @mapped_regions")
        .sort_values(["year", "region_REMIND", "carrier"])
        .reset_index(drop=True)
    )

    capacities.to_csv(snakemake.output["capacities"], index=False)
    logger.info(
        "Wrote %d capacity-target rows to %s",
        len(capacities),
        snakemake.output["capacities"],
    )
