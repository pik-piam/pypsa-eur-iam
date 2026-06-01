
"""
Build REMIND-adjusted technology costs for PyPSA-Eur.

Reads investment costs, fixed/variable O&M, lifetime, efficiency, CO2 intensity, and
fuel costs from the REMIND GDX file, maps them to PyPSA-Eur carrier names via the
technology mapping CSV, and merges the result as overrides on top of the PyPSA-Eur
baseline cost CSV. Investment costs for electrolysis and battery inverter are converted
from output-capacity to input-capacity basis. Per-region discount rates from REMIND are
used, and PyPSA-Eur's ``prepare_costs`` function computes annualised capital costs and
marginal costs — called once per mapped REMIND region.

Outputs
-------
- ``costs_raw_overwritten.csv``: raw cost table restricted to mapped technologies, with REMIND overrides applied; one block per region (region column is the first column).
- ``costs_processed.csv``: processed cost table (capital_cost, marginal_cost, etc.) ready for the network build; indexed by (region, technology) MultiIndex.
"""

import logging

import pandas as pd
import pypsa
from _helpers import configure_logging, get_region_mapping, read_remind_data

logger = logging.getLogger(__name__)


def extract_remind_parameter_data(snakemake, mapped_regions: set[str]) -> pd.DataFrame:
    """Read all REMIND parameters needed for cost overrides and return a long-format DataFrame."""
    year = str(snakemake.wildcards["year_REMIND"])  # noqa: F841 — used via @year in .query()

    costs = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="p32_capCost",
        rename_columns={"ttot": "year", "all_regi": "region", "all_te": "technology"},
    ).query("year == @year")
    costs["value"] *= 1e6
    costs["parameter"] = "investment"
    costs["unit"] = "USD/MW"
    costs.loc[costs["technology"].isin(["h2stor", "btstor"]), "unit"] = "USD/MWh"

    pm_data = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="pm_data",
        rename_columns={"all_regi": "region", "all_te": "technology"},
    )

    lifetime = pm_data.query("char == 'lifetime'").copy()
    lifetime["parameter"] = "lifetime"
    lifetime["unit"] = "years"

    fom = pm_data.query("char == 'omf'").copy()
    fom["value"] *= 100
    fom["parameter"] = "FOM"
    fom["unit"] = "%/year"

    vom = pm_data.query("char == 'omv'").copy()
    vom["value"] *= 1e6 / 8760
    vom["parameter"] = "VOM"
    vom["unit"] = "USD/MWh"

    co2_intensity = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="pm_emifac",
        rename_columns={
            "tall_0": "year",
            "all_regi_1": "region",
            "all_enty_2": "from_carrier",
            "all_enty_3": "to_carrier",
            "all_te_4": "technology",
            "all_enty_5": "emission_type",
        },
    ).query("to_carrier == 'seel' & emission_type == 'co2' & year == @year")
    co2_intensity["value"] *= 1e9 * ((2 * 16 + 12) / 12) / 8760 / 1e6
    co2_intensity["parameter"] = "CO2 intensity"
    co2_intensity["unit"] = "t_CO2/MWh_th"

    eta_conv = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="pm_eta_conv",
        rename_columns={"tall": "year", "all_regi": "region", "all_te": "technology"},
    ).query("year == @year")
    dataeta = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="pm_dataeta",
        rename_columns={"tall": "year", "all_regi": "region", "all_te": "technology"},
    ).query("year == @year")
    # pm_eta_conv has time-varying efficiency; pm_dataeta is the static fallback.
    # Use pm_eta_conv where available; add pm_dataeta only for technologies absent from pm_eta_conv.
    eta_conv_keys = set(zip(eta_conv["region"], eta_conv["technology"]))
    dataeta_fallback = dataeta[
        ~pd.MultiIndex.from_arrays([dataeta["region"], dataeta["technology"]]).isin(eta_conv_keys)
    ]
    efficiency = pd.concat([eta_conv, dataeta_fallback])
    efficiency["parameter"] = "efficiency"
    efficiency["unit"] = "p.u."
    efficiency.loc[efficiency["technology"].isin(["fnrs", "tnrs"]), "value"] *= 8760 / 1e6
    efficiency.loc[efficiency["technology"].isin(["fnrs", "tnrs"]), "unit"] = "MWh/g_U"
    efficiency.loc[efficiency["technology"] == "btin", "value"] **= 2

    fuel_costs = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="p32_PEPriceAvg",
        rename_columns={"ttot": "year", "all_regi": "region", "all_enty": "technology"},
    ).query("year == @year")
    fuel_costs["parameter"] = "fuel"
    fuel_costs.loc[~(fuel_costs["technology"] == "peur"), "value"] *= 1e6 / 8760
    fuel_costs["unit"] = "USD/MWh_th"
    fuel_costs.loc[fuel_costs["technology"] == "peur", "unit"] = "USD/g_U"

    df = pd.concat([costs, lifetime, fom, vom, co2_intensity, efficiency, fuel_costs])[
        ["region", "technology", "parameter", "value", "unit"]
    ].rename(columns={"technology": "reference"})

    return df[df["region"].isin(mapped_regions)]


def build_mapped_overrides(
    technology_mapping: pd.DataFrame,
    remind_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Direct 1:1 lookup of REMIND parameter values for mapped PyPSA-Eur carriers.

    Returns exactly one row per (region, technology, parameter) combination;
    raises if a duplicate is found.
    """
    mapped = technology_mapping.query(
        "`source` == 'REMIND'"
    ).drop(columns=["unit"])

    merged = mapped.merge(
        remind_df,
        on=["reference", "parameter"],
        how="left",
    )

    # Technologies absent from the GDX produce NaN values after the left join.
    # Fall back to PyPSA-Eur baseline for those (technology, parameter) pairs so
    # that merge_overrides_into_baseline never overwrites valid baseline costs with NaN.
    missing_mask = merged["value"].isna()
    if missing_mask.any():
        missing = (
            merged.loc[missing_mask, ["PyPSA-Eur technology", "reference", "parameter"]]
            .drop_duplicates()
        )
        for _, row in missing.iterrows():
            logger.warning(
                "REMIND reference '%s' (→ '%s', parameter '%s') not found in GDX "
                "— falling back to PyPSA-Eur baseline value.",
                row["reference"],
                row["PyPSA-Eur technology"],
                row["parameter"],
            )
        merged = merged[~missing_mask]

    out = merged.rename(columns={"PyPSA-Eur technology": "technology"})[
        ["region", "technology", "parameter", "value", "unit"]
    ].copy()
    dups = out.duplicated(subset=["region", "technology", "parameter"], keep=False)
    if dups.any():
        raise ValueError(
            f"Duplicate (region, technology, parameter) after REMIND merge:\n{out[dups]}"
        )
    out["source"] = "REMIND-EU"
    out["further description"] = "Extracted from REMIND-EU model in import_REMIND_costs.py"
    return out[["region", "technology", "parameter", "value", "unit", "source", "further description"]]


def build_pypsa_default_overrides(
    technology_mapping: pd.DataFrame,
    baseline_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Pull parameter values from the PyPSA-Eur baseline costs for rows marked source=PyPSA-Eur."""
    df = technology_mapping.query(
        "`source` == 'PyPSA-Eur'"
    ).drop(columns=["unit"])
    df = df.merge(
        baseline_raw,
        left_on=["PyPSA-Eur technology", "parameter"],
        right_on=["technology", "parameter"],
        how="left",
        validate="one_to_one",
    )
    df["source"] = "PyPSA-EUR"
    df["further description"] = "Default parameter from PyPSA-EUR baseline cost file"
    return df[["technology", "parameter", "value", "unit", "source", "further description"]]


def build_set_value_overrides(technology_mapping: pd.DataFrame, mapping_file: str) -> pd.DataFrame:
    """Return overrides for rows marked source=fixed, converting the reference column to a numeric value."""
    set_df = technology_mapping.query("`source` == 'fixed'").rename(
        columns={
            "PyPSA-Eur technology": "technology",
            "reference": "value",
            "comment": "further description",
        }
    )[["technology", "parameter", "value", "unit", "further description"]].copy()
    set_df["value"] = pd.to_numeric(set_df["value"], errors="raise")
    set_df["source"] = f"Set via configuration file: {mapping_file}"
    set_df["further description"] = set_df["further description"].fillna("")
    return set_df


def get_discount_rates(snakemake, mapped_regions: set[str]) -> pd.Series:
    """Return REMIND discount rate per mapped region for the model year, as a Series indexed by region."""
    year = str(snakemake.wildcards["year_REMIND"])  # noqa: F841 — used via @year in .query()
    p_r = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="p_r",
        rename_columns={"ttot": "year", "all_regi": "region"},
    ).query("year == @year and region in @mapped_regions")

    if p_r.empty:
        raise ValueError(
            f"No p_r interest rate found for year {year} and regions {mapped_regions}"
        )

    missing = mapped_regions - set(p_r["region"])
    if missing:
        raise ValueError(f"No discount rate found for regions: {missing}")

    rates = p_r.set_index("region")["value"]
    logger.info("Regional REMIND discount rates for year %s: %s", year, rates.round(4).to_dict())
    return rates


def add_discount_rate_for_region(costs: pd.DataFrame, discount_rate: float) -> pd.DataFrame:
    """Append a discount rate row for every technology in *costs* that does not already carry one."""
    with_discount = costs.loc[costs["parameter"] == "discount rate", "technology"]
    no_discount = costs.loc[~costs["technology"].isin(with_discount)][["technology"]].drop_duplicates()

    dr = pd.DataFrame({
        "parameter": ["discount rate"],
        "value": [discount_rate],
        "unit": ["p.u."],
        "source": ["REMIND-EU"],
        "further description": ["p_r"],
    })
    dr = dr.merge(no_discount, how="cross")
    return pd.concat([costs, dr], ignore_index=True)


def convert_investment_to_input_capacity_basis(costs: pd.DataFrame) -> pd.DataFrame:
    """
    REMIND investment costs are per kW of output capacity; PyPSA needs per kW of input (p_nom).

    Converts by multiplying by efficiency (eta = output/input): cost_per_kW_in = cost_per_kW_out * eta.

    - electrolysis: stored efficiency is eta_H2/el (not modified), multiply directly.
    - battery inverter: stored efficiency is already eta_rt = eta_oneway**2 (pre-squared so that
      add_electricity.py's **0.5 recovers the one-way value). Capital cost conversion needs
      eta_oneway, so take sqrt of stored efficiency before multiplying.
    - fuel cell: handled at Link creation in add_electricity_sector_REMIND.py.
    """
    costs = costs.copy()
    # exponent applied to stored efficiency: 1 = use directly, 0.5 = take sqrt
    # (battery inverter efficiency is pre-squared to eta_rt so add_electricity's **0.5 recovers eta_oneway)
    eta_exponents = {"electrolysis": 1, "battery inverter": 0.5}
    for tech, exp in eta_exponents.items():
        inv_mask = (costs["technology"] == tech) & (costs["parameter"] == "investment")
        eff_mask = (costs["technology"] == tech) & (costs["parameter"] == "efficiency")
        if inv_mask.any() and eff_mask.any():
            costs.loc[inv_mask, "value"] *= costs.loc[eff_mask, "value"].values ** exp
            logger.info("Converted investment costs for %s from output to input capacity basis.", tech)
    return costs


def merge_overrides_into_baseline(
    baseline_raw: pd.DataFrame,
    overrides: pd.DataFrame,
) -> pd.DataFrame:
    """Apply overrides onto the baseline cost table, adding new rows where needed."""
    base = baseline_raw.set_index(["technology", "parameter"]).copy()
    ov = overrides.set_index(["technology", "parameter"]).copy()

    if ov.index.duplicated().any():
        raise ValueError(
            "Duplicate overrides for (technology, parameter): "
            f"{ov.index[ov.index.duplicated()].tolist()}"
        )

    extra_idx = ov.index.difference(base.index)
    if len(extra_idx) > 0:
        base = pd.concat([base, ov.loc[extra_idx, base.columns.intersection(ov.columns)]])

    shared_idx = ov.index.intersection(base.index)
    for col in ["value", "unit", "source", "further description"]:
        if col in ov.columns:
            base.loc[shared_idx, col] = ov.loc[shared_idx, col]

    merged = base.reset_index()
    if merged.duplicated(subset=["technology", "parameter"]).any():
        dups = merged[merged.duplicated(subset=["technology", "parameter"], keep=False)]
        raise ValueError(f"Duplicates after merge: {dups}")
    return merged


if __name__ == "__main__":
    import sys
    from pathlib import Path
    # When run directly, Python adds scripts/ to sys.path, not the repo root.
    # scripts.process_cost_data must be imported as a package from the repo root,
    # so we insert it explicitly. Not needed under Snakemake (which sets up sys.path correctly).
    sys.path.insert(0, str(Path(__file__).parents[1]))
    import scripts.process_cost_data as process_cost_data
    from scripts.process_cost_data import prepare_costs

    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "import_REMIND_costs",
            scen_REMIND="PkBudg1000_EU",
            iter_REMIND="1",
            year_REMIND="2050",
            configfiles="config/config.remind_europe.yaml",
        )

    configure_logging(snakemake)
    year = str(snakemake.wildcards["year_REMIND"])
    logger.info("Building REMIND-adjusted costs for year %s", year)

    countries = set(snakemake.config["countries"])
    full_mapping = get_region_mapping(snakemake.input["region_mapping"], source="PyPSA-EUR", target="REMIND-EU")
    mapped_regions = {r for c, rs in full_mapping.items() if c in countries for r in rs if r}

    technology_mapping = pd.read_csv(snakemake.input["technology_cost_mapping"])
    mapped_technologies = set(technology_mapping["PyPSA-Eur technology"].dropna().unique())

    remind_long = extract_remind_parameter_data(snakemake, mapped_regions)
    baseline_raw = pd.read_csv(snakemake.input["original_costs"])

    # REMIND-derived overrides keep their region dimension; non-regional overrides are
    # the same for every region (PyPSA-Eur baseline values and fixed-value entries).
    regional_mapped_overrides = build_mapped_overrides(technology_mapping, remind_long)
    pypsa_overrides = build_pypsa_default_overrides(technology_mapping, baseline_raw)
    set_overrides = build_set_value_overrides(
        technology_mapping,
        snakemake.input["technology_cost_mapping"],
    )
    non_regional_overrides = pd.concat([pypsa_overrides, set_overrides], ignore_index=True)

    discount_rates = get_discount_rates(snakemake, mapped_regions)

    n = pypsa.Network(snakemake.input["network"])
    nyears = n.snapshot_weightings.generators.sum() / 8760.0
    # `prepare_costs` currently resolves `snakemake` and `planning_horizon`
    # from module-level globals in `scripts.process_cost_data`. We set them
    # here to keep `process_cost_data.py` unchanged while calling it from REMIND.
    process_cost_data.snakemake = snakemake
    process_cost_data.planning_horizon = year

    all_raw = []
    all_processed = []

    for region in sorted(mapped_regions):
        region_overrides = regional_mapped_overrides[
            regional_mapped_overrides["region"] == region
        ].drop(columns="region")

        combined = pd.concat([region_overrides, non_regional_overrides], ignore_index=True)
        combined = add_discount_rate_for_region(combined, discount_rates[region])
        combined = convert_investment_to_input_capacity_basis(combined)

        merged_raw = merge_overrides_into_baseline(baseline_raw, combined)

        merged_raw_mapped = merged_raw.loc[merged_raw["technology"].isin(mapped_technologies)].copy()
        merged_raw_mapped.insert(0, "region", region)
        all_raw.append(merged_raw_mapped)

        costs_processed = prepare_costs(
            costs=merged_raw.set_index(["technology", "parameter"]),
            config=snakemake.params["costs"],
            max_hours=snakemake.params["max_hours"],
            nyears=nyears,
            custom_costs_fn=snakemake.input.get("custom_costs"),
        )
        costs_processed = costs_processed.loc[
            costs_processed.index.isin(mapped_technologies)
        ].copy()
        costs_processed.index = pd.MultiIndex.from_tuples(
            [(region, t) for t in costs_processed.index], names=["region", "technology"]
        )
        all_processed.append(costs_processed)

    raw_combined = pd.concat(all_raw, ignore_index=True)
    processed_combined = pd.concat(all_processed)

    logger.info(
        "Keeping %d raw cost rows across %d regions × %d mapped technologies",
        len(raw_combined),
        raw_combined["region"].nunique(),
        raw_combined["technology"].nunique(),
    )
    logger.info(
        "Keeping %d processed cost rows across %d regions × %d mapped technologies",
        len(processed_combined),
        processed_combined.index.get_level_values("region").nunique(),
        processed_combined.index.get_level_values("technology").nunique(),
    )

    required_cols = ["capital_cost", "marginal_cost"]
    missing_required = [c for c in required_cols if c not in processed_combined.columns]
    if missing_required:
        raise ValueError(f"Missing required columns in processed costs: {missing_required}")
    if processed_combined[required_cols].isna().any().any():
        nan_cols = list(processed_combined[required_cols].columns[processed_combined[required_cols].isna().any()])
        raise ValueError(f"NaN values in required processed cost columns: {nan_cols}")

    logger.info(
        "Exporting overwritten raw costs to %s",
        snakemake.output["costs_raw_overwritten"],
    )
    raw_combined.to_csv(snakemake.output["costs_raw_overwritten"], index=False)

    logger.info(
        "Exporting processed costs to %s",
        snakemake.output["costs_processed"],
    )
    processed_combined.to_csv(snakemake.output["costs_processed"])

    # Region-averaged flat costs: single-index by technology, used by prepare_network.py
    # which expects the upstream load_costs() format (index_col=0 → technology index).
    costs_flat = processed_combined.groupby(level="technology").mean()
    logger.info(
        "Exporting flat (region-averaged) processed costs to %s",
        snakemake.output["costs_processed_flat"],
    )
    costs_flat.to_csv(snakemake.output["costs_processed_flat"])
