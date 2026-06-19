"""
Read sectoral electricity demand from REMIND and export it for use in PyPSA-Eur.

Stage 1 of the demand pipeline. Reads the load-sector symbol (``v32_load_sector`` with
``p32_load_sector`` fallback, resolved from the central symbol config), converts TWa to annual
MWh (applied by ``load_frame`` via the symbol's ``to_unit``), and filters to the REMIND regions
overlapping the configured countries. ``downscale_REMIND_demand`` then splits this to countries.
"""

import logging

from _helpers import configure_logging, mock_snakemake
from rpycpl.io import RemindLoader
from rpycpl.io.remind_symbols import load_frame, load_symbol_specs
from rpycpl.transforms.loads import convert_loads
from rpycpl.transforms.mapping import read_region_map as get_region_mapping

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "import_REMIND_demand",
            scen_REMIND="TEST_multiregion",
            iter_REMIND="1",
            configfiles="config/config.remind_multiregion.yaml",
        )

    configure_logging(snakemake)
    logger.info("Loading REMIND regional demand from the rpycpl symbol config ...")

    region_mapping = get_region_mapping(
        snakemake.input["region_mapping"], source="PyPSA-EUR", target="REMIND-EU"
    )
    mapped_regions = sorted({r for rs in region_mapping.values() for r in rs if r})

    loader = RemindLoader(snakemake.input["remind_data"])
    symbols = load_symbol_specs()

    raw = load_frame(loader, symbols["load_sector"])  # TWa -> MWh applied here
    raw["year"] = raw["year"].astype(int)
    demand = convert_loads(raw, regions=mapped_regions, unit_factor=1.0)

    demand.to_csv(snakemake.output["sectoral_load"], index=False)
    logger.info(
        "Wrote %s rows of REMIND demand to %s",
        len(demand),
        snakemake.output["sectoral_load"],
    )
