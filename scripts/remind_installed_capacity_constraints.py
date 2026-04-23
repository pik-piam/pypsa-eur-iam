# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""REMIND installed-capacity lower-bound constraints for custom extra functionality."""

import logging

import pandas as pd
import pypsa
import xarray as xr

from scripts._helpers import get_region_mapping, get_technology_mapping

logger = logging.getLogger(__name__)


def _build_country_to_region_map(fp_region_mapping: str) -> pd.Series:
    mapping = get_region_mapping(
        fp_region_mapping,
        source="PyPSA-EUR",
        target="REMIND-EU",
        flatten=True,
    )
    return pd.Series(mapping)


def _build_carrier_to_technology_group_map(fp_technology_mapping: str) -> pd.Series:
    mapping = get_technology_mapping(fp_technology_mapping, group_technologies=True)
    mapping = mapping[["PyPSA-Eur", "technology_group"]].drop_duplicates()
    carrier_to_group = (
        mapping.groupby("PyPSA-Eur", observed=False)["technology_group"].first()
    )

    # Manual mappings for carrier names that are handled outside the CSV mapping.
    manual_mapping = {
        "H2 Electrolysis": "electrolysis",
        "H2 Fuel Cell": "fuel cell",
        "battery charger": "battery inverter",
        "H2": "hydrogen storage underground",
        "battery": "battery storage",
    }
    for carrier, group in manual_mapping.items():
        carrier_to_group.loc[carrier] = group

    return carrier_to_group


def _build_component_group_labels(
    n: pypsa.Network,
    component_df: pd.DataFrame,
    bus_col: str,
    country_to_region: pd.Series,
    carrier_to_group: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    countries = component_df[bus_col].map(n.buses.country)
    regions = countries.map(country_to_region)

    carriers = component_df["carrier"].astype(str)
    technology_groups = carriers.map(carrier_to_group)

    return regions, technology_groups


def _prepare_targets(snakemake) -> pd.Series:
    capacities = pd.read_csv(snakemake.input["capacities"])
    year = int(snakemake.wildcards.year_REMIND)

    required_columns = {"year", "region_REMIND", "technology_group", "p_nom_min"}
    missing_columns = required_columns.difference(capacities.columns)
    if missing_columns:
        raise ValueError(
            "installed_capacities.csv misses required columns: "
            + ", ".join(sorted(missing_columns))
        )

    capacities = capacities.query("year == @year").copy()
    if capacities.empty:
        logger.warning("No REMIND installed capacities for year %s. No constraints added.", year)
        return pd.Series(dtype=float)

    targets = capacities.groupby(["region_REMIND", "technology_group"], observed=False)[
        "p_nom_min"
    ].sum()
    return targets[targets > 0]


def _add_component_lower_bound_constraints(
    n: pypsa.Network,
    component_name: str,
    component_df: pd.DataFrame,
    variable_name: str,
    extendable_col: str,
    bus_col: str,
    targets: pd.Series,
    country_to_region: pd.Series,
    carrier_to_group: pd.Series,
    constraint_name: str,
) -> None:
    if targets.empty:
        return

    regions, technology_groups = _build_component_group_labels(
        n,
        component_df,
        bus_col,
        country_to_region,
        carrier_to_group,
    )

    labels = pd.DataFrame(
        {
            "region_REMIND": regions,
            "technology_group": technology_groups,
        },
        index=component_df.index,
    )

    unmapped_carriers = component_df.loc[
        labels["technology_group"].isna(), "carrier"
    ].dropna()
    if not unmapped_carriers.empty:
        logger.warning(
            "%s: unmapped carriers for technology group mapping: %s",
            component_name,
            ", ".join(sorted(unmapped_carriers.astype(str).unique())),
        )

    valid_idx = labels.dropna().index
    if valid_idx.empty:
        logger.warning("%s: no components with valid region and technology group labels.", component_name)
        return

    ext_idx = component_df.index[component_df[extendable_col]].intersection(valid_idx)

    if ext_idx.empty:
        missing = targets[targets > 0]
        if not missing.empty:
            logger.warning(
                "%s: no extendable assets for %s target groups with positive lower bounds.",
                component_name,
                len(missing),
            )
        return

    labels_ext = labels.loc[ext_idx]
    grouper = xr.DataArray(
        pd.MultiIndex.from_arrays(
            [labels_ext["region_REMIND"], labels_ext["technology_group"]]
        ),
        dims=[n.model[variable_name].dims[0]],
    )

    lhs_total = n.model[variable_name].loc[ext_idx].groupby(grouper).sum()

    rhs_groups = targets[targets > 0].index
    lhs_groups = lhs_total.indexes["group"]

    missing_in_network = rhs_groups.difference(lhs_groups)
    if not missing_in_network.empty:
        logger.warning(
            "%s: %s target groups are not represented by extendable assets.",
            component_name,
            len(missing_in_network),
        )

    unconstrained_groups = lhs_groups.difference(targets.index)
    if len(unconstrained_groups):
        logger.info(
            "%s: %s modeled groups have no installed-capacity lower bound in RHS.",
            component_name,
            len(unconstrained_groups),
        )

    index = lhs_groups.intersection(targets.index)
    if index.empty:
        logger.warning(
            "%s: no overlapping groups between modeled capacities and REMIND targets.",
            component_name,
        )
        return

    n.model.add_constraints(
        lhs_total.sel(group=index) >= targets.loc[index].values,
        name=constraint_name,
    )


def add_installed_capacity_lower_bound_constraints(n: pypsa.Network, snakemake) -> None:
    """Add REMIND installed-capacity lower-bound constraints by region and technology group."""
    targets = _prepare_targets(snakemake)
    if targets.empty:
        return

    country_to_region = _build_country_to_region_map(snakemake.input["region_mapping"])
    carrier_to_group = _build_carrier_to_technology_group_map(
        snakemake.input["technology_cost_mapping"]
    )

    logger.info(
        "Adding REMIND installed-capacity lower-bound constraints for %s groups.",
        len(targets),
    )

    _add_component_lower_bound_constraints(
        n=n,
        component_name="Generator",
        component_df=n.generators,
        variable_name="Generator-p_nom",
        extendable_col="p_nom_extendable",
        bus_col="bus",
        targets=targets,
        country_to_region=country_to_region,
        carrier_to_group=carrier_to_group,
        constraint_name="REMIND_installed_capacity_generator_min",
    )

    _add_component_lower_bound_constraints(
        n=n,
        component_name="Link",
        component_df=n.links,
        variable_name="Link-p_nom",
        extendable_col="p_nom_extendable",
        bus_col="bus0",
        targets=targets,
        country_to_region=country_to_region,
        carrier_to_group=carrier_to_group,
        constraint_name="REMIND_installed_capacity_link_min",
    )

    _add_component_lower_bound_constraints(
        n=n,
        component_name="Store",
        component_df=n.stores,
        variable_name="Store-e_nom",
        extendable_col="e_nom_extendable",
        bus_col="bus",
        targets=targets,
        country_to_region=country_to_region,
        carrier_to_group=carrier_to_group,
        constraint_name="REMIND_installed_capacity_store_min",
    )
