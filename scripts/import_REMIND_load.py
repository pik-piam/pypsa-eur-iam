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
            "import_REMIND_load", scenario="TEST", iteration="1", year="2030"
        )

    configure_logging(snakemake)

    # %%
    # Load original load timeseries from PyPSA-EUR
    load = pd.read_csv(snakemake.input["load_timeseries"], index_col=0)

    # Load REMIND-EU total load
    demand = read_remind_data(
        snakemake.input["remind_data"],
        "p32_load",
        rename_columns={
            "ttot": "year",
            "all_regi": "region",
            "value": "total",
        },
    )
    
    # Subtract sectoral loads if enabled in config
    sc_settings = snakemake.params["sector_coupling"]

    if sc_settings["EVs"]["enable"]:
        demand_evs = read_remind_data(
            snakemake.input["remind_data"],
            "p32_load_EVs",
            rename_columns={
                "ttot": "year",
                "all_regi": "region",
                "value": "EVs",
            },
        )

        demand = demand.join(
            demand_evs.set_index(["year", "region"])["EVs"],
            on=["year", "region"],
            how="left",
        )

    if sc_settings["heating"]["enable"]:
        demand_heatpump = read_remind_data(
            snakemake.input["remind_data"],
            "p32_load_heatpump",
            rename_columns={
                "ttot": "year",
                "all_regi": "region",
                "value": "heatpump",
            },
        )
        demand_resistive = read_remind_data(
            snakemake.input["remind_data"],
            "p32_load_resistive",
            rename_columns={
                "ttot": "year",
                "all_regi": "region",
                "value": "resistive",
            },
        )

        demand = demand.join(
            demand_heatpump.set_index(["year", "region"])["heatpump"],
            on=["year", "region"],
            how="left",
        )
        demand = demand.join(
            demand_resistive.set_index(["year", "region"])["resistive"],
            on=["year", "region"],
            how="left",
        )

    # Calculate residual demand
    demand["residual"] = demand["total"]
    if sc_settings["EVs"]["enable"]:
        demand["residual"] -= demand["EVs"]
    if sc_settings["heating"]["enable"]:
        demand["residual"] -= demand["heatpump"] + demand["resistive"]
        
    # If negative set to zero and write warning
    if (demand["residual"] < 0).any():
        logger.warning(
            "Negative residual demand detected. Setting negative values to zero."
        )
        # Set negative values to zero
        demand.loc[demand["residual"] < 0, "residual"] = 0

    demand = demand.loc[demand["year"] == snakemake.wildcards["year"],]
    demand = demand.set_index(["region", "year"])
    demand = demand * 1e6 * 8760  # Convert from TWa to MWh
    
    # TODO: Save demand and use in later steps when attaching sectoral load

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

    # Factor by which PyPSA-EUR loads have to be scaled to match REMIND-EU demand
    load_scaling_factor = (
        (demand.reset_index().set_index(["region"])["residual"] / regional_load)
        .dropna()
        .rename("load_scaling_factor")
    )
    load_scaling_factor = region_mapping.join(
        load_scaling_factor,
        on="REMIND-EU",
        how="inner",
    ).set_index("PyPSA-EUR")["load_scaling_factor"]

    load_scaling_factor.to_csv(snakemake.output["load_scaling_factor"])

# %%
