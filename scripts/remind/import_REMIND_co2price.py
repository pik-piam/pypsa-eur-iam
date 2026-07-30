"""
Import a year-level CO2 price pathway from REMIND, per REMIND region.

Reads the CO2-price symbol (resolved from the central symbol config; tC->tCO2 applied by
``load_frame`` via the symbol's ``to_unit``), filters to the mapped REMIND regions and reindexes
to REMIND's coupled-year set (missing filled with 0). The extract/convert chain lives in the
coupling package via ``Coupler.build_co2_prices(years=...)``.
"""

import logging

from _helpers import configure_logging, mock_snakemake
from scripts.remind._remind_helpers import build_remind_coupler
from iampypsa.io.remind_symbols import load_frame

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "import_REMIND_co2price",
            snakefile_choices=["Snakefile_REMIND"],
            scen_REMIND="PkBudg1000_EU",
            iter_REMIND="1",
            configfiles="config/config.remind_europe.yaml",
        )

    configure_logging(snakemake)
    logger.info("Building REMIND CO2 price pathway from the iampypsa symbol config")

    countries = set(snakemake.config["countries"])
    coupler = build_remind_coupler(snakemake.input["remind_data"], countries)

    # GDX: reindex to REMIND's coupled-year set (symbol `t`).
    # IAMC: mif starts in 2005 so we can't infer coupling years from the data; use config instead.
    if "coupled_years" in coupler.symbols:
        coupled_years = sorted(
            load_frame(coupler.loader, coupler.symbols["coupled_years"])["year"].astype(int).unique()
        )
    else:
        coupled_years = sorted(int(y) for y in snakemake.config["remind_coupling"]["years"])

    co2_price = coupler.build_co2_prices(years=coupled_years)  # tC -> tCO2 applied here
    co2_price = (
        co2_price.rename(columns={"value": "co2_price"})[
            ["region", "year", "co2_price"]
        ]
        .sort_values(["region", "year"])
        .reset_index(drop=True)
    )

    logger.info("Exporting CO2 price pathway to %s", snakemake.output["co2_price"])
    co2_price.to_csv(snakemake.output["co2_price"], index=False)
