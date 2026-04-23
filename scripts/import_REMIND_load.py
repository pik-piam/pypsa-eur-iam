# -*- coding: utf-8 -*-

# %%

import logging

import pandas as pd
from _helpers import (
    configure_logging,
    get_region_mapping,
    mock_snakemake,
    read_remind_data,
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "import_REMIND_load",
            scenario="PyPSA_PkBudg1000_start2030_exportnewEVload_2025-07-09_13.38.18",
            iteration="1",
            year="2050",
            #configfiles="resources/PyPSA_PkBudg1000_start2025_newLoad_2025-07-07_11.29.49/i1/config.remind_scenario.yaml",
        )

    configure_logging(snakemake)

    # %%
    # Load original load timeseries from PyPSA-EUR
    load = pd.read_csv(snakemake.input["load_timeseries"], index_col=0)

    def combine_sector(demand, source, target):
        """Transfer value from one sector to another and drop the source."""
        value = demand.loc[demand["sector"] == source, "value"].squeeze()
        demand.loc[demand["sector"] == target, "value"] += value
        demand = demand[demand["sector"] != source]
        return demand

    # Load demand data
    demand_sector = read_remind_data(
        snakemake.input["remind_data"],
        "v32_load_sector",
        rename_columns={
            "ttot": "year",
            "all_regi": "region",
            "loadPy32": "sector",
            "level": "value",
        },
    ).query("year == @snakemake.wildcards.year_REMIND")
    # Convert TWa to MWh and assign units
    demand_sector["value"] *= 1e6 * 8760
    demand_sector["unit"] = "MWh_el"
    demand_sector = demand_sector[["region", "sector", "unit", "value"]]

    # Sector coupling settings
    sc_settings = snakemake.params["sector_coupling"]

    # Handle additional hydrogen demand
    if not sc_settings["additional_hydrogen"]["enable"]:
        demand_sector = combine_sector(demand_sector, "electrolysis", "AC")
    else:
        eta = read_remind_data(
            snakemake.input["remind_data"],
            "pm_eta_conv",
            rename_columns={
                "tall": "year",
                "all_regi": "region",
                "all_te": "technology",
                "value": "eta",
            },
        ).query("year == @snakemake.wildcards.year_REMIND & technology == 'elh2'")
        h2_demand = demand_sector.query("sector == 'electrolysis'").merge(eta)
        h2_demand["value"] *= h2_demand["eta"]
        mask = demand_sector["sector"] == "electrolysis"
        demand_sector.loc[mask, "value"] = h2_demand["value"].values[0]
        demand_sector.loc[mask, "unit"] = "MWh_H2"

    # Handle passenger EVs
    if not sc_settings["EV_pass"]["enable"]:
        demand_sector = combine_sector(demand_sector, "EV_pass", "AC")
        
    # Handle freight EVs
    if not sc_settings["EV_freight"]["enable"]:
        demand_sector = combine_sector(demand_sector, "EV_freight", "AC")

    # Handle heat pumps
    if not sc_settings["heat_pumps"]["enable"]:
        demand_sector = combine_sector(demand_sector, "heatpump", "AC")
        
    # Handle resistive heating
    if not sc_settings["resistive"]["enable"]:
        demand_sector = combine_sector(demand_sector, "resistive", "AC")

    # Save sectoral demand to CSV
    demand_sector.to_csv(snakemake.output["sectoral_load"])

    # Get region mapping
    region_mapping = (
        pd.DataFrame.from_dict(
            get_region_mapping(
                snakemake.input["region_mapping"],
                source="PyPSA-EUR",
                target="REMIND-EU",
            ),
            "index",
        )
        .reset_index()
        .rename(columns={"index": "PyPSA-EUR", 0: "REMIND-EU"})
    )

    # Calculate load from PyPSA-EUR load-timeseries for REMIND-EU regions
    regional_load = (
        region_mapping.join(load.sum(axis="rows").rename("annual_load"), on="PyPSA-EUR")
        .groupby("REMIND-EU")["annual_load"]
        .sum()
    )
    regional_load = regional_load[regional_load > 0]

    # Factor by which PyPSA-EUR loads have to be scaled to match REMIND-EU demand
    load_scaling_factor = (
        (demand_sector.query("sector == 'AC'").value.values[0] / regional_load).rename(
            "load_scaling_factor"
        )
    ).fillna(0)
    load_scaling_factor = region_mapping.join(
        load_scaling_factor,
        on="REMIND-EU",
        how="inner",
    ).set_index("PyPSA-EUR")["load_scaling_factor"]

    load_scaling_factor.to_csv(snakemake.output["load_scaling_factor"])

# %%
