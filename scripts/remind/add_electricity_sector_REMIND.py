# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Orchestrate the REMIND-coupled electricity + sector build.

Two layers, reflected in the phases of ``__main__``:
1. Upstream-equivalent assembly — the same steps as PyPSA-Eur ``add_electricity``,
   reusing its functions (imported from ``scripts.add_electricity``) with REMIND
   inputs/params substituted (regional costs, adjusted power plants, per-bus load
   scaling); hydro is REMIND-adjusted via ``attach_hydro_remind``.
2. REMIND overlay — per-node hydrogen storage/conversion (buses, cavern/tank stores,
   electrolysis and fuel-cell links; no H2 grid) and sector-coupling loads
   (electrolysis / EV / heat) that upstream ``add_electricity`` never builds, plus a
   per-region cost override.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from iampypsa.couplers.remind import read_region_map as get_region_mapping

from scripts.add_electricity import (
    add_co2_emissions,
    add_missing_carriers,
    attach_conventional_generators,
    attach_load,
    attach_storageunits,
    attach_stores,
    attach_wind_and_solar,
    configure_logging,
    get_snapshots,
    load_and_aggregate_powerplants,
    normed,
    sanitize_carriers,
    sanitize_locations,
    set_scenario_config,
    set_transmission_costs,
    update_p_nom_max,
)
from scripts.build_transport_demand import transport_degree_factor

logger = logging.getLogger(__name__)

def overwrite_ppl_efficiency_with_costs(ppl: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    """Replace each plant's efficiency and marginal cost with the REMIND cost table value for its carrier."""
    ppl = ppl.copy()
    ppl["efficiency"] = ppl["carrier"].map(costs["efficiency"])
    ppl["marginal_cost"] = (
        ppl["carrier"].map(costs["VOM"])
        + ppl["carrier"].map(costs["fuel"]) / ppl["efficiency"]
    )
    return ppl


def build_costs_with_suptech_aliases(costs: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of costs with a fallback row per missing carrier prefix.

    Upstream ``add_co2_emissions`` (scripts.add_electricity) looks up each carrier's
    ``-``-prefix (e.g. ``gas-chp`` → ``gas``), but REMIND carriers have no bare-prefix
    rows; alias them to the carrier's own row so the lookup does not raise KeyError.

    The aliasing gives *approximate* CO₂ intensities (all ``coal-*`` share one row);
    ``overwrite_carrier_co2_intensities`` corrects them before any intensity is used.
    """
    costs = costs.copy()
    for carrier in list(costs.index):
        suptech = carrier.split("-")[0]
        if suptech not in costs.index:
            costs.loc[suptech] = costs.loc[carrier]
    return costs


def fold_sector_into_ac(demand: pd.DataFrame, source_sector: str) -> pd.DataFrame:
    """Re-label a sector as AC and aggregate duplicate rows."""
    demand = demand.copy()
    demand.loc[demand["sector"] == source_sector, "sector"] = "AC"
    return (
        demand.groupby(["year", "region", "sector", "unit"], as_index=False)["value"]
        .sum()
        .sort_values(["year", "region", "sector"])
    )


def prepare_sectoral_load(
    sectoral_load_country_fn: str,
    sector_coupling: dict,
    costs: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Load and aggregate REMIND sectoral demand according to sector-coupling config."""
    sectoral_load = pd.read_csv(sectoral_load_country_fn)
    sectoral_load = sectoral_load.query("year == @year").copy()

    if sector_coupling["electrolysis"]["enable"]:
        eta = costs.loc["electrolysis", "efficiency"]
        # Multiply electrolysis load by efficiency to convert from electricity demand to hydrogen demand
        h2_mask = sectoral_load["sector"] == "electrolysis"
        sectoral_load.loc[h2_mask, "value"] *= eta
        sectoral_load.loc[h2_mask, "unit"] = "MWh_H2"
    else:
        sectoral_load = fold_sector_into_ac(sectoral_load, "electrolysis")

    for sector in ("EV_pass", "EV_freight", "heatpump", "resistive"):
        if not sector_coupling[sector]["enable"]:
            sectoral_load = fold_sector_into_ac(sectoral_load, sector)

    return (
        sectoral_load.groupby(["year", "region", "sector", "unit"], as_index=False)
        .sum(numeric_only=True)
        .sort_values(["year", "region", "sector"])
    )


def compute_load_scaling_per_country(
    n: pypsa.Network,
    busmap_fn: str,
    load_fn: str,
    sectoral_load: pd.DataFrame,
) -> pd.Series:
    """Return a per-country Series (indexed by iso2) scaling AC load to REMIND demand."""
    remind_ac = (
        sectoral_load.query("sector == 'AC'")
        .set_index("region")["value"]
    )

    # Annual MWh per cluster bus: sum the hourly raw demand over time, then apply
    # the busmap (OSM bus names → cluster buses), then aggregate to country level.
    busmap = pd.read_csv(busmap_fn, index_col="name").squeeze()
    country_load = (
        xr.open_dataarray(load_fn)
        .sum("time")
        .to_pandas()
        .groupby(busmap)
        .sum()
        .groupby(n.buses["country"])
        .sum()
    )

    per_country = remind_ac / country_load

    return per_country


def _normalize_weights(s: pd.Series) -> pd.Series:
    """Normalize a non-negative Series to sum to 1; return uniform weights if the total is zero."""
    s = s.astype(float).clip(lower=0.0)
    total = s.sum()
    if total <= 0:
        return pd.Series(1.0 / len(s), index=s.index) if len(s) else s
    return s / total


def _to_scalar_region(value):
    """Return the first non-null member of a list-like value, or the value itself if scalar."""
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if pd.notna(item):
                return item
        return float("nan")
    return value


def _get_country_to_region() -> pd.Series:
    """Return a Series mapping PyPSA-EUR country codes to scalar REMIND-EU region labels."""
    mapping = get_region_mapping(source="country", target="model_region")
    return pd.Series(mapping).map(_to_scalar_region)


def _optional_input_path(value: Any) -> str | None:
    """Coerce a snakemake input (None, [], [path] or path) to a path string, or None."""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return str(value) if value else None


def _fleet_evs_by_node_from_rds(
    n: pypsa.Network,
    fleet_file: str,
    number_cars: pd.Series,
    year: int,
    kind: str,
    country_to_region: pd.Series,
) -> pd.Series | None:
    """Read BEV fleet sizes from an EDGE-T RDS file and distribute to nodes by REMIND region; None if unavailable."""
    try:
        import pyreadr
    except ImportError as exc:
        raise ModuleNotFoundError(
            "pyreadr is required to read fleetVehiclesPerTech.RDS but is not available."
        ) from exc

    subsector_map = {
        "pass": "trn_pass_road_LDV",
        "freight": "trn_freight_road_tmp_subsectorL2",
    }

    rds = pyreadr.read_r(fleet_file)
    fleet = next((v for v in rds.values() if isinstance(v, pd.DataFrame)), None)
    if fleet is None or fleet.empty:
        return None

    required = {"period", "region", "technology", "subsectorL2", "totVeh"}
    if not required.issubset(fleet.columns):
        logger.warning("Fleet RDS file is missing required columns %s; using fallback fleet estimate.", required)
        return None

    period = pd.to_numeric(fleet["period"], errors="coerce")
    tot_veh = pd.to_numeric(fleet["totVeh"], errors="coerce")
    rows = fleet[
        period.eq(int(year))
        & fleet["technology"].eq("BEV")
        & fleet["subsectorL2"].eq(subsector_map[kind])
        & tot_veh.notna()
    ].copy()
    if rows.empty:
        return None

    node_country = n.buses["country"].reindex(number_cars.index)
    node_region = node_country.map(country_to_region)

    number_evs = pd.Series(0.0, index=number_cars.index)
    found_any_region = False
    for _, row in rows.iterrows():
        region = _to_scalar_region(row["region"])
        if pd.isna(region):
            continue
        region_nodes = number_evs.index[node_region.eq(region)]
        if len(region_nodes) == 0:
            continue
        weights = _normalize_weights(number_cars.reindex(region_nodes).fillna(0.0))
        # EDGE-T reports totVeh in millions of vehicles.
        number_evs.loc[region_nodes] += weights * (float(row["totVeh"]) * 1e6)
        found_any_region = True

    if not found_any_region or number_evs.sum() <= 0:
        return None
    return number_evs


def attach_hydrogen_demand_remind(
    n: pypsa.Network,
    sectoral_load: pd.DataFrame,
) -> None:
    """Add a fixed H2 Load at each H2 bus, weighted by AC load within each country."""
    h2_demand = (
        sectoral_load.query("sector == 'electrolysis'")
        .groupby("region", as_index=True)["value"]
        .sum()
    )
    if h2_demand.empty:
        logger.info("No REMIND electrolysis demand found to attach.")
        return

    h2_buses = n.buses.query("carrier == 'H2'").copy()
    if h2_buses.empty:
        logger.warning("No H2 buses in network; skipping REMIND hydrogen demand attachment.")
        return

    h2_buses["country"] = h2_buses["location"].map(n.buses["country"])
    bus_load = n.loads_t.p_set.sum(axis=0).groupby(n.loads["bus"]).sum()
    h2_buses["ac_load"] = h2_buses["location"].map(bus_load).fillna(0.0)

    # Split each country's H2 demand across its H2 buses in proportion to the parent
    # AC bus's annual load (uniform if a country's AC load is all zero).
    h2_buses["weight"] = h2_buses.groupby("country")["ac_load"].transform(_normalize_weights)

    # REMIND gives annual H2 energy (MWh); a static Load needs constant power (MW), so
    # divide by the represented hours over the horizon (snapshot weightings sum, ≈ 8760).
    total_hours = float(n.snapshot_weightings.generators.sum())
    h2_buses["p_set"] = (
        h2_buses["country"].map(h2_demand).fillna(0.0) * h2_buses["weight"] / total_hours
    )

    h2_buses = h2_buses[h2_buses["p_set"] > 0]
    n.add(
        "Load",
        h2_buses.index + " demand REMIND",
        carrier="H2",
        bus=h2_buses.index,
        p_set=h2_buses["p_set"].values,
    )
    logger.info("Attached REMIND hydrogen demand to %s H2 loads.", len(h2_buses))


def cycling_shift(df: pd.DataFrame, steps: int = 1) -> pd.DataFrame:
    """Circularly shift all rows of a DataFrame by `steps` positions along the time axis."""
    shifted = df.copy()
    shifted.values[:] = shifted.reindex(index=np.roll(shifted.index, steps)).values
    return shifted


def distribute_annual_demand(
    demand_by_country: pd.Series,
    profile: pd.DataFrame,
    node_country: pd.Series,
) -> pd.DataFrame:
    """
    Shape each country's annual demand onto its nodes using an hourly profile.

    Per country, the profile columns of its nodes are rescaled so their summed energy
    equals the country demand; countries with no nodes or a zero-sum profile give zero.
    """
    out = pd.DataFrame(0.0, index=profile.index, columns=profile.columns)
    for country, demand in demand_by_country.items():
        cnodes = node_country[node_country == country].index.intersection(profile.columns)
        if cnodes.empty:
            logger.warning("No nodes for country %s in demand profile; skipping.", country)
            continue
        total = float(profile[cnodes].to_numpy().sum())
        if total <= 0:
            continue
        out[cnodes] = profile[cnodes] * (demand / total)
    return out


def attach_ev_demand_remind(
    n: pypsa.Network,
    sectoral_load: pd.DataFrame,
    options_ev: dict,
    options_sector: dict,
    transport_demand_fn: str,
    transport_data_fn: str,
    avail_profile_fn: str,
    dsm_profile_fn: str,
    temp_air_total_fn: str,
    fleet_file: str | list,
    year: int,
    kind: str,
    country_to_region: pd.Series,
) -> None:
    """
    Add EV demand, charging links, and optional DSM stores for passenger or freight vehicles.

    REMIND annual total is distributed to an hourly profile via transport demand scaled by
    heating degree days. Fleet size uses EDGE-T RDS if available, else config-based estimate.
    """
    if kind not in ("pass", "freight"):
        raise ValueError("kind must be 'pass' or 'freight'")

    sector = f"EV_{kind}"
    demand_mwh = sectoral_load.query("sector == @sector")["value"].sum()
    if demand_mwh <= 0:
        logger.info("No REMIND %s demand found; skipping.", sector)
        return

    spatial_nodes = n.buses.query("carrier == 'AC'").index
    transport = pd.read_csv(transport_demand_fn, index_col=0, parse_dates=True)
    transport = transport.reindex(index=n.snapshots)
    common_cols = [c for c in spatial_nodes if c in transport.columns]
    if not common_cols:
        logger.warning("No overlap between EV transport demand and AC buses; skipping %s.", sector)
        return
    transport = transport[common_cols].fillna(0.0)

    # Fill any residual gaps with the heating deadband's lower bound, where
    # transport_degree_factor yields no demand adjustment (neutral).
    temperature = xr.open_dataarray(temp_air_total_fn).to_pandas()
    temperature = (
        temperature.reindex(index=n.snapshots, columns=common_cols)
        .ffill()
        .bfill()
        .fillna(options_sector["transport_heating_deadband_lower"])
    )
    dd = transport_degree_factor(
        temperature,
        options_sector["transport_heating_deadband_lower"],
        options_sector["transport_heating_deadband_upper"],
        options_sector["EV_lower_degree_factor"],
        options_sector["EV_upper_degree_factor"],
    )
    transport = transport.mul(1 + dd, axis=0)

    demand_by_country = (
        sectoral_load.query("sector == @sector").set_index("region")["value"]
    )
    node_country = n.buses.loc[common_cols, "country"]
    load_p_set = distribute_annual_demand(demand_by_country, transport, node_country)

    transport_data = pd.read_csv(transport_data_fn, index_col=0)
    number_cars = transport_data["number cars"].reindex(common_cols).fillna(0.0)
    car_weights = _normalize_weights(number_cars)

    fleet_file_path = _optional_input_path(fleet_file)
    number_evs = None
    if fleet_file_path:
        number_evs = _fleet_evs_by_node_from_rds(
            n,
            fleet_file_path,
            number_cars,
            year,
            kind,
            country_to_region,
        )
        if number_evs is not None:
            number_evs = number_evs.reindex(common_cols).fillna(0.0)
            logger.info("Using EV fleet from %s for %s.", fleet_file_path, sector)

    if number_evs is None:
        number_evs_total = float(demand_mwh) / float(options_ev["annual_consumption"])
        number_evs = car_weights * number_evs_total
        logger.info("Using config-based EV fleet fallback for %s.", sector)

    charge_power = number_evs * float(options_ev["charge_rate"]) * float(options_ev["share_charger"])
    link_p_nom = charge_power * float(options_ev["dsm_availability"])
    store_e_nom = number_evs * float(options_ev["battery_size"]) * float(options_ev["dsm_availability"])

    avail = pd.read_csv(avail_profile_fn, index_col=0, parse_dates=True).reindex(index=n.snapshots)
    avail = avail.reindex(columns=common_cols).fillna(1.0)
    dsm = pd.read_csv(dsm_profile_fn, index_col=0, parse_dates=True).reindex(index=n.snapshots)
    dsm = dsm.reindex(columns=common_cols).fillna(0.0)

    carrier_name = f"EV {kind} battery"
    ev_nodes = pd.Index(common_cols) + f" {carrier_name}"

    n.add("Carrier", [carrier_name])
    n.add(
        "Bus",
        ev_nodes,
        location=common_cols,
        carrier=carrier_name,
        unit="MWh_el",
    )

    if options_ev["dsm"]:
        p_set = load_p_set
    else:
        p_set = (load_p_set + cycling_shift(load_p_set, 1) + cycling_shift(load_p_set, 2)) / 3

    n.add(
        "Load",
        common_cols,
        suffix=f" land transport EV {kind}",
        bus=ev_nodes,
        carrier=f"land transport EV {kind}",
        p_set=p_set,
    )

    n.add(
        "Link",
        common_cols,
        suffix=f" BEV {kind} charger",
        bus0=common_cols,
        bus1=ev_nodes,
        p_nom=link_p_nom.reindex(common_cols).fillna(0.0),
        carrier=f"BEV {kind} charger",
        p_max_pu=avail,
        efficiency=1.0,
    )

    if options_ev["dsm"]:
        n.add(
            "Store",
            common_cols,
            suffix=f" {carrier_name}",
            bus=ev_nodes,
            carrier=carrier_name,
            e_cyclic=True,
            e_nom=store_e_nom.reindex(common_cols).fillna(0.0),
            e_max_pu=1.0,
            e_min_pu=dsm,
        )

    logger.info("Attached REMIND %s demand (%.2f MWh).", sector, float(demand_mwh))


def attach_heat_demand_remind(
    n: pypsa.Network,
    sectoral_load: pd.DataFrame,
    options_heat: dict,
    wh_share_fn: str,
    hourly_heat_demand_fn: str,
    hourly_water_heat_demand_fn: str,
    year: int,
    kind: str,
    cop_profiles_fn: str | None = None,
) -> None:
    """
    Add electric heat demand for heat pumps or resistive heaters, scaled to the REMIND annual total.

    Space/water split uses wh_share; heatpump electricity = thermal / hourly COP (rural air-source).
    Optionally adds a fixed-size thermal storage buffer for DSM with E/P ratio from config.
    """
    if kind not in ("heatpump", "resistive"):
        raise ValueError("kind must be 'heatpump' or 'resistive'")

    demand_by_country = (
        sectoral_load.query("sector == @kind").set_index("region")["value"]
    )
    if demand_by_country.sum() <= 0:
        logger.info("No REMIND %s demand found; skipping.", kind)
        return

    spatial_nodes = n.buses.query("carrier == 'AC'").index
    wh_share = pd.read_csv(wh_share_fn)
    wh_year = min(int(year), 2100)  # wh_share data stops at 2100
    wh = wh_share[wh_share["year"] == wh_year]
    item = "heat pumps" if kind == "heatpump" else "resistive heating"
    wh = wh[wh["item"] == item]
    if wh.empty:
        raise ValueError(
            f"No water-heating share for item={item!r}, year={wh_year} in {wh_share_fn}"
        )
    water_share = float(wh["value"].iloc[0])

    space_ds = xr.open_dataset(hourly_heat_demand_fn)
    space_heat = (space_ds["residential space"] + space_ds["services space"]).to_pandas()
    space_heat = space_heat.reindex(index=n.snapshots, columns=spatial_nodes).fillna(0.0)

    water_ds = xr.open_dataset(hourly_water_heat_demand_fn)
    water_heat = water_ds.to_dataframe()
    if "snapshots" in water_heat.index.names:
        water_heat = water_heat.reset_index().set_index("snapshots")
    water_heat.index = pd.to_datetime(water_heat.index)
    water_heat = water_heat.reindex(index=n.snapshots, columns=spatial_nodes).fillna(0.0)

    if kind == "heatpump":
        if cop_profiles_fn is None:
            raise ValueError("cop_profiles_fn is required for heatpump demand")
        cop = xr.open_dataarray(cop_profiles_fn)
        # REMIND-specific: fill nodes/timesteps missing after reindex with COP=1
        # (heat pump degrades to resistive). Upstream cop profiles span all nodes.
        cop = (
            cop.sel(heat_system="rural", heat_source="air")
            .to_pandas()
            .reindex(index=n.snapshots, columns=spatial_nodes)
            .fillna(1.0)
        )
        space_heat = space_heat / cop
        water_heat = water_heat / cop

    node_country = n.buses.loc[spatial_nodes, "country"]
    total_elec_profile = distribute_annual_demand(
        demand_by_country * (1.0 - water_share), space_heat, node_country
    ) + distribute_annual_demand(
        demand_by_country * water_share, water_heat, node_country
    )

    carrier = f"{kind} electricity"
    heat_nodes = spatial_nodes + f" {carrier}"

    n.add("Carrier", [carrier])
    n.add(
        "Bus",
        heat_nodes,
        location=spatial_nodes,
        carrier=carrier,
        unit="MWh_el",
    )

    n.add(
        "Load",
        spatial_nodes,
        suffix=f" {carrier}",
        bus=heat_nodes,
        carrier=carrier,
        p_set=total_elec_profile,
    )

    p_nom_max = total_elec_profile.max(axis=0)
    n.add(
        "Link",
        spatial_nodes,
        suffix=f" {kind}",
        carrier=kind,
        bus0=spatial_nodes,
        bus1=heat_nodes,
        p_nom=p_nom_max,
        p_nom_extendable=False,
        efficiency=1.0,
        p_min_pu=0.0,
        p_max_pu=1.0,
    )

    if options_heat["dsm"]:
        ep_ratio = float(options_heat["ep_ratio"])

        if kind == "heatpump":
            total_heat_profile = total_elec_profile * cop
            p_therm_max = total_heat_profile.max(axis=0).fillna(0.0)
            e_therm = p_therm_max * ep_ratio

            cop_safe = cop.clip(lower=0.001)  # div-by-zero guard, matching upstream
            elec_storage = e_therm / cop_safe
            size_store = elec_storage.max(axis=0).fillna(0.0)
            max_pu_store = (elec_storage / size_store.replace(0.0, np.nan)).fillna(0.0)
        else:
            p_therm_max = total_elec_profile.max(axis=0).fillna(0.0)
            size_store = p_therm_max * ep_ratio
            max_pu_store = 1.0

        n.add(
            "Store",
            spatial_nodes,
            suffix=f" {kind} storage",
            bus=heat_nodes,
            carrier=f"{kind} storage",
            e_cyclic=True,
            e_nom=size_store,
            e_nom_extendable=False,
            e_max_pu=max_pu_store,
            e_min_pu=0.0,
        )

        logger.info(
            "Attached %s DSM store with fixed E/P ratio %.2f h (total e_nom=%.2f MWh_el).",
            kind,
            ep_ratio,
            float(pd.Series(size_store).sum()),
        )

    logger.info("Attached REMIND %s demand (%.2f MWh).", kind, float(demand_by_country.sum()))


def attach_hydro_remind(
    n: pypsa.Network,
    costs: pd.DataFrame,
    ppl: pd.DataFrame,
    profile_hydro: str,
    hydro_capacities: str,
    carriers: list,
    hydro_targets_fn: str,
    year: int,
    country_to_region: pd.Series,
    **hydro_params,
) -> None:
    """
    Attach hydro following the upstream attach_hydro flow, with two REMIND-specific injections:
    scale ror/hydro capacities by REMIND region and scale inflow energy by REMIND region.
    """
    add_missing_carriers(n, carriers)
    add_co2_emissions(n, costs, carriers)

    ror = ppl.query('carrier == "ror"').copy()
    phs = ppl.query('carrier == "PHS"').copy()
    hydro = ppl.query('carrier == "hydro"').copy()

    # Region-scaling and inflow need ror/hydro assets. When only PHS exists (both empty),
    # skip straight to the add-blocks below
    inflow_idx = ror.index.union(hydro.index)
    if not inflow_idx.empty:
        # Build country/region labels for hydro assets.
        hydro_assets = pd.concat([ror, hydro], axis=0)
        hydro_assets["country"] = hydro_assets["bus"].map(n.buses.country)
        hydro_assets["region"] = hydro_assets["country"].map(country_to_region)

        hydro_targets = pd.read_csv(hydro_targets_fn)
        hydro_targets["year"] = hydro_targets["year"].astype(str)
        hydro_targets_year = hydro_targets[hydro_targets["year"] == str(year)]
        if hydro_targets_year.empty:
            raise ValueError(
                f"No REMIND hydro targets for year {year} in {hydro_targets_fn}"
            )

        target_capacity = (
            hydro_targets_year.groupby("region", observed=False)["hydro_capacity_mw"]
            .sum()
            .astype(float)
        )
        target_generation = (
            hydro_targets_year.groupby("region", observed=False)["hydro_generation_mwh"]
            .sum()
            .astype(float)
        )

        # 1) Capacity scaling by REMIND region.
        current_capacity = hydro_assets.groupby("region", observed=False)["p_nom"].sum()
        capacity_factor = (target_capacity / current_capacity).replace([np.inf, -np.inf], np.nan)
        capacity_factor = capacity_factor.fillna(1.0)
        hydro_assets["cap_factor"] = hydro_assets["region"].map(capacity_factor).fillna(1.0)
        hydro_assets["p_nom"] = hydro_assets["p_nom"] * hydro_assets["cap_factor"]

        # Push adjusted p_nom back to ror/hydro tables.
        ror.loc[hydro_assets.loc[ror.index].index, "p_nom"] = hydro_assets.loc[ror.index, "p_nom"]
        hydro.loc[hydro_assets.loc[hydro.index].index, "p_nom"] = hydro_assets.loc[hydro.index, "p_nom"]

        country = hydro_assets["country"]
        dist_key = hydro_assets.loc[inflow_idx, "p_nom"].groupby(country.loc[inflow_idx]).transform(normed)

        with xr.open_dataarray(profile_hydro) as inflow:
            inflow_countries = pd.Index(country.loc[inflow_idx])
            missing_c = inflow_countries.unique().difference(inflow.indexes["countries"])
            assert missing_c.empty, (
                f"'{profile_hydro}' is missing "
                f"inflow time-series for at least one country: {', '.join(missing_c)}"
            )

            inflow_t = (
                inflow.sel(countries=inflow_countries)
                .rename({"countries": "name"})
                .assign_coords(name=inflow_idx)
                .transpose("time", "name")
                .to_pandas()
                .multiply(dist_key, axis=1)
            )

        # 2) Inflow energy scaling by REMIND region.
        asset_region = hydro_assets.loc[inflow_idx, "region"]
        current_generation = inflow_t.T.groupby(asset_region).sum().sum(axis=1)
        generation_factor = (target_generation / current_generation).replace([np.inf, -np.inf], np.nan)
        generation_factor = generation_factor.fillna(1.0)
        inflow_factor = asset_region.map(generation_factor).fillna(1.0)
        inflow_t = inflow_t.multiply(inflow_factor, axis=1)

    if "ror" in carriers and not ror.empty:
        n.add(
            "Generator",
            ror.index,
            carrier="ror",
            bus=ror["bus"],
            p_nom=ror["p_nom"],
            efficiency=costs.at["ror", "efficiency"],
            capital_cost=costs.at["ror", "capital_cost"],
            weight=ror["p_nom"],
            p_max_pu=(
                inflow_t[ror.index]
                .divide(ror["p_nom"], axis=1)
                .where(lambda df: df <= 1.0, other=1.0)
            ),
        )

    if "PHS" in carriers and not phs.empty:
        max_hours = hydro_params.get("PHS_max_hours", 6)
        phs = phs.replace({"max_hours": {0: max_hours, np.nan: max_hours}})
        n.add(
            "StorageUnit",
            phs.index,
            carrier="PHS",
            bus=phs["bus"],
            p_nom=phs["p_nom"],
            capital_cost=costs.at["PHS", "capital_cost"],
            max_hours=phs["max_hours"],
            efficiency_store=np.sqrt(costs.at["PHS", "efficiency"]),
            efficiency_dispatch=np.sqrt(costs.at["PHS", "efficiency"]),
            cyclic_state_of_charge=True,
        )

    if "hydro" in carriers and not hydro.empty:
        hydro_max_hours = hydro_params.get("hydro_max_hours")

        assert hydro_capacities is not None, "No path for hydro capacities given."

        hydro_stats = pd.read_csv(
            hydro_capacities, comment="#", na_values="-", index_col=0
        )
        e_target = hydro_stats["E_store[TWh]"].clip(lower=0.2) * 1e6
        e_installed = hydro.eval("p_nom * max_hours").groupby(hydro.country).sum()
        e_missing = e_target - e_installed
        missing_mh_i = hydro.query("max_hours.isnull() or max_hours == 0").index
        missing_mh_single_i = hydro.index[
            ~hydro.country.duplicated() & hydro.country.isin(e_missing.dropna().index)
        ]
        missing_mh_i = missing_mh_i.union(missing_mh_single_i)

        if hydro_max_hours == "energy_capacity_totals_by_country":
            max_hours_country = (
                e_missing / hydro.loc[missing_mh_i].groupby("country").p_nom.sum()
            )

        elif hydro_max_hours == "estimate_by_large_installations":
            max_hours_country = (
                hydro_stats["E_store[TWh]"] * 1e3 / hydro_stats["p_nom_discharge[GW]"]
            )
        else:
            raise ValueError(f"Unknown hydro_max_hours method: {hydro_max_hours}")

        max_hours_country.clip(0, inplace=True)

        missing_countries = pd.Index(hydro["country"].unique()).difference(
            max_hours_country.dropna().index
        )
        if not missing_countries.empty:
            logger.warning(
                f"Assuming max_hours=6 for hydro reservoirs in the countries: {', '.join(missing_countries)}"
            )
        hydro_max_hours = hydro.max_hours.where(
            (hydro.max_hours > 0) & ~hydro.index.isin(missing_mh_single_i),
            hydro.country.map(max_hours_country),
        ).fillna(6)

        if hydro_params.get("flatten_dispatch", False):
            buffer = hydro_params.get("flatten_dispatch_buffer", 0.2)
            average_capacity_factor = inflow_t[hydro.index].mean() / hydro["p_nom"]
            p_max_pu = (average_capacity_factor + buffer).clip(upper=1)
        else:
            p_max_pu = 1

        n.add(
            "StorageUnit",
            hydro.index,
            carrier="hydro",
            bus=hydro["bus"],
            p_nom=hydro["p_nom"],
            max_hours=hydro_max_hours,
            capital_cost=costs.at["hydro", "capital_cost"],
            marginal_cost=costs.at["hydro", "marginal_cost"],
            p_max_pu=p_max_pu,
            p_min_pu=0.0,
            efficiency_dispatch=costs.at["hydro", "efficiency"],
            efficiency_store=0.0,
            cyclic_state_of_charge=True,
            inflow=inflow_t.loc[:, hydro.index],
        )


def attach_hydrogen_storage_remind(
    n: pypsa.Network,
    costs: pd.DataFrame,
    h2_cavern_fn: str,
    storage_settings: dict,
    electrolysis_settings: dict,
) -> None:
    """Attach REMIND hydrogen buses, storage, and conversion links."""
    ac_buses = n.buses.query("carrier == 'AC'").index
    if ac_buses.empty:
        logger.info("No AC buses available for REMIND hydrogen infrastructure.")
        return

    n.add("Carrier", "H2")

    h2_buses = pd.Index([f"{bus} H2" for bus in ac_buses])
    n.add(
        "Bus",
        h2_buses,
        location=ac_buses,
        carrier="H2",
        unit="MWh_H2",
        x=n.buses.loc[ac_buses, "x"].values,
        y=n.buses.loc[ac_buses, "y"].values,
    )

    h2_caverns = pd.read_csv(h2_cavern_fn, index_col=0)
    enabled_types = [
        cavern_type
        for cavern_type in storage_settings["hydrogen_underground_storage_locations"]
        if cavern_type in h2_caverns.columns
    ]
    if enabled_types:
        # Salt-cavern potentials are in TWh; keep sizeable sites (> 2 TWh), convert to
        # MWh, and cap per node.
        cavern_twh = h2_caverns[enabled_types].sum(axis=1)
        cavern_twh = cavern_twh[cavern_twh > 2]
        cavern_capacity_mwh = (cavern_twh * 1e6).clip(upper=1e9)
    else:
        cavern_capacity_mwh = pd.Series(dtype=float)

    h2_cavern_buses = pd.Index([])
    if not cavern_capacity_mwh.empty:
        underground_capital_cost = costs.at["hydrogen storage underground", "capital_cost"]
        underground_lifetime = costs.at["hydrogen storage underground", "lifetime"]
        h2_cavern_buses = cavern_capacity_mwh.index.intersection(ac_buses)

        n.add(
            "Store",
            h2_cavern_buses.map(lambda bus: f"{bus} H2 Store"),
            bus=h2_cavern_buses.map(lambda bus: f"{bus} H2"),
            carrier="H2 Store",
            e_nom_extendable=True,
            e_nom_max=cavern_capacity_mwh.reindex(h2_cavern_buses).values,
            e_cyclic=True,
            capital_cost=underground_capital_cost,
            lifetime=underground_lifetime,
        )

    tank_tech = "hydrogen storage tank type 1 including compressor"
    tank_buses = ac_buses.difference(cavern_capacity_mwh.index)
    if not tank_buses.empty:
        n.add(
            "Store",
            tank_buses.map(lambda bus: f"{bus} H2 Store"),
            bus=tank_buses.map(lambda bus: f"{bus} H2"),
            carrier="H2 Store",
            e_nom_extendable=True,
            e_cyclic=True,
            capital_cost=costs.at[tank_tech, "capital_cost"],
            lifetime=costs.at[tank_tech, "lifetime"],
        )

    if electrolysis_settings.get("enable", True):
        electrolysis_names = ac_buses.map(lambda bus: f"{bus} H2 Electrolysis")
        ramp_limit = electrolysis_settings.get("ramp_limit", np.nan)
        n.add(
            "Link",
            electrolysis_names,
            bus0=ac_buses,
            bus1=h2_buses,
            carrier="H2 Electrolysis",
            p_nom_extendable=True,
            p_min_pu=electrolysis_settings.get("p_min_pu", 0.0),
            efficiency=costs.at["electrolysis", "efficiency"],
            capital_cost=costs.at["electrolysis", "capital_cost"],
            marginal_cost=costs.at["electrolysis", "marginal_cost"],
            lifetime=costs.at["electrolysis", "lifetime"],
            ramp_limit_up=None if pd.isna(ramp_limit) else ramp_limit,
            ramp_limit_down=None if pd.isna(ramp_limit) else ramp_limit,
        )

    fuel_cell_names = ac_buses.map(lambda bus: f"{bus} H2 Fuel Cell")
    n.add(
        "Link",
        fuel_cell_names,
        bus0=h2_buses,
        bus1=ac_buses,
        carrier="H2 Fuel Cell",
        p_nom_extendable=True,
        efficiency=costs.at["fuel cell", "efficiency"],
        # NB: fuel cell investment cost is per MWel, so scale by efficiency (per MWH2 in).
        capital_cost=costs.at["fuel cell", "capital_cost"] * costs.at["fuel cell", "efficiency"],
        marginal_cost=costs.at["fuel cell", "marginal_cost"],
        lifetime=costs.at["fuel cell", "lifetime"],
    )

    logger.info(
        "Attached REMIND hydrogen infrastructure to %s AC buses (%s cavern sites, %s tank sites).",
        len(ac_buses),
        len(h2_cavern_buses),
        len(tank_buses),
    )


def apply_regional_costs(
    n: pypsa.Network,
    costs_regional: pd.DataFrame,
    co2_price_by_region: dict,
    country_to_region: pd.Series,
) -> None:
    """
    Override capital_cost and marginal_cost per component using REMIND regional values.

    Also adds the regional CO₂ cost to generator/storage marginal costs, replacing the
    global add_emission_prices() call that prepare_network.py would otherwise make.
    """
    def _apply_costs(df: pd.DataFrame, bus_col: str) -> None:
        if df.empty:
            return
        countries = n.buses.loc[df[bus_col], "country"].values
        regions = pd.Series(countries, index=df.index).map(country_to_region)
        # Build (region, carrier) lookup keys for each component row
        keys = pd.MultiIndex.from_arrays([regions, df["carrier"]])
        valid_mask = keys.isin(costs_regional.index)
        if not valid_mask.any():
            return
        valid_idx = df.index[valid_mask]
        valid_keys = keys[valid_mask]
        for col in ["capital_cost", "marginal_cost"]:
            if col not in costs_regional.columns:
                continue
            new_vals = costs_regional.loc[valid_keys, col].values
            not_nan = ~pd.isna(new_vals)
            df.loc[valid_idx[not_nan], col] = new_vals[not_nan]

    _apply_costs(n.generators, "bus")
    _apply_costs(n.links, "bus0")
    _apply_costs(n.storage_units, "bus")
    _apply_costs(n.stores, "bus")

    # Add regional CO₂ cost contribution to generator marginal costs.
    # This replaces the global add_emission_prices() that prepare_network.py would
    # otherwise call; kept consistent via costs.emission_prices.enable: false in
    # config/config.remind.yaml.
    if not n.generators.empty and co2_price_by_region:
        countries = n.buses.loc[n.generators["bus"], "country"].values
        regions = pd.Series(countries, index=n.generators.index).map(country_to_region)
        co2_price_series = regions.map(co2_price_by_region).fillna(0.0)
        co2_intensity = n.generators["carrier"].map(n.carriers["co2_emissions"]).fillna(0.0)
        efficiency = n.generators["efficiency"].replace(0.0, 1.0)
        n.generators["marginal_cost"] += co2_intensity * co2_price_series / efficiency

    if not n.storage_units.empty and co2_price_by_region:
        countries = n.buses.loc[n.storage_units["bus"], "country"].values
        regions = pd.Series(countries, index=n.storage_units.index).map(country_to_region)
        co2_price_series = regions.map(co2_price_by_region).fillna(0.0)
        co2_intensity = n.storage_units["carrier"].map(n.carriers["co2_emissions"]).fillna(0.0)
        efficiency = n.storage_units["efficiency_dispatch"].replace(0.0, 1.0)
        n.storage_units["marginal_cost"] += co2_intensity * co2_price_series / efficiency

    logger.info("Applied regional REMIND costs to all network components.")


def apply_battery_e_min_pu(
    n: pypsa.Network, battery_settings: dict, extendable_carriers: dict
) -> None:
    """Set a minimum state-of-charge on battery stores from REMIND battery settings."""
    if "e_min_pu" in battery_settings and "battery" in extendable_carriers["Store"]:
        e_min_pu = battery_settings["e_min_pu"]
        battery_mask = n.stores["carrier"] == "battery"
        n.stores.loc[battery_mask, "e_min_pu"] = e_min_pu
        logger.info(f"Applied REMIND battery e_min_pu constraint: {e_min_pu}")


def overwrite_carrier_co2_intensities(n: pypsa.Network, costs: pd.DataFrame) -> None:
    """
    Set each carrier's CO₂ intensity from its full-name cost row.

    Corrects the suptech-prefix approximation from ``build_costs_with_suptech_aliases``,
    which conflates e.g. biomass-chp and biomass-igcc-ccs (BECCS is carbon-negative).
    """
    for carrier in costs.index:
        if carrier in n.carriers.index and not pd.isna(
            costs.at[carrier, "CO2 intensity"]
        ):
            n.carriers.at[carrier, "co2_emissions"] = costs.at[carrier, "CO2 intensity"]


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_electricity_sector_REMIND",
            scen_REMIND="PkBudg1000_DE_TEST",
            iter_REMIND="1",
            year_REMIND="2050",
            clusters=4,
            configfiles="config/config.remind_de.yaml",
        )

    configure_logging(snakemake)  # pylint: disable=E0606
    set_scenario_config(snakemake)

    # ------------------------------------------------------------------
    # Setup: network, snapshots, REMIND costs / CO₂ prices / region map,
    # power plants, and the per-bus load scaling to REMIND demand.
    # ------------------------------------------------------------------
    params = snakemake.params
    max_hours = params.electricity["max_hours"]
    year = int(snakemake.wildcards.year_REMIND)
    landfall_lengths = {
        tech: settings["landfall_length"]
        for tech, settings in params.renewable.items()
        if "landfall_length" in settings.keys()
    }

    # Not supported in the REMIND build.
    assert not params.electricity["estimate_renewable_capacities"]["enable"], (
        "estimate_renewable_capacities is not supported in the REMIND build"
    )
    assert not params.electricity.get("estimate_battery_capacities", False), (
        "estimate_battery_capacities is not supported in the REMIND build"
    )

    n = pypsa.Network(snakemake.input.base_network)

    time = get_snapshots(snakemake.params.snapshots, snakemake.params.drop_leap_day)
    n.set_snapshots(time)

    # Load costs with (region, technology) MultiIndex; derive region-averaged fallback
    # for upstream attach_* functions that don't know about regional variation.
    costs_regional = pd.read_csv(snakemake.input.costs, index_col=[0, 1])
    costs = costs_regional.groupby(level="technology").mean()

    # Per-region CO₂ prices (applied in apply_regional_costs below)
    co2_price_df = pd.read_csv(snakemake.input.co2_price)
    co2_price_by_region = (
        co2_price_df.query("year == @year")
        .set_index("region")["co2_price"]
        .to_dict()
    )

    country_to_region = _get_country_to_region()

    ppl = load_and_aggregate_powerplants(
        snakemake.input.powerplants,
        costs,
        params.consider_efficiency_classes,
        params.aggregation_strategies,
        params.exclude_carriers,
    )
    ppl = overwrite_ppl_efficiency_with_costs(ppl, costs)

    sector_coupling = snakemake.params["sector_coupling"]
    sectoral_load = prepare_sectoral_load(
        snakemake.input.sectoral_load_country, sector_coupling, costs, year=year
    )
    load_scaling_per_country = compute_load_scaling_per_country(
        n, snakemake.input.busmap, snakemake.input.load, sectoral_load
    )
    load_scaling_per_bus = n.buses["country"].map(load_scaling_per_country)

    renewable_carriers = set(params.electricity["renewable_carriers"])
    extendable_carriers = params.electricity["extendable_carriers"]
    conventional_carriers = params.electricity["conventional_carriers"]
    conventional_inputs = {
        k: v for k, v in snakemake.input.items() if k.startswith("conventional_")
    }

    if params.conventional["unit_commitment"]:
        unit_commitment = pd.read_csv(snakemake.input.unit_commitment, index_col=0)
    else:
        unit_commitment = None

    if params.conventional["dynamic_fuel_price"]:
        fuel_price = pd.read_csv(
            snakemake.input.fuel_price, index_col=0, parse_dates=True
        )
        fuel_price = fuel_price.reindex(n.snapshots).ffill()
    else:
        fuel_price = None

    # ------------------------------------------------------------------
    # Upstream-equivalent assembly: same steps as PyPSA-Eur add_electricity,
    # with REMIND-specific inputs/params (regional costs, adjusted power
    # plants, per-bus load scaling) substituted; hydro is REMIND-adjusted.
    # ------------------------------------------------------------------
    attach_load(
        n,
        snakemake.input.load,
        snakemake.input.busmap,
        load_scaling_per_bus,
    )

    set_transmission_costs(
        n,
        costs,
        params.line_length_factor,
        params.link_length_factor,
    )

    attach_conventional_generators(
        n,
        build_costs_with_suptech_aliases(costs),
        ppl,
        conventional_carriers,
        extendable_carriers,
        renewable_carriers,
        conventional_params=params.conventional,
        conventional_inputs=conventional_inputs,
        unit_commitment=unit_commitment,
        fuel_price=fuel_price,
    )

    attach_wind_and_solar(
        n,
        costs,
        snakemake.input,
        renewable_carriers,
        extendable_carriers,
        params.line_length_factor,
        landfall_lengths,
    )

    if "hydro" in renewable_carriers:
        p = params.renewable["hydro"].copy()
        carriers = p.pop("carriers", [])
        attach_hydro_remind(
            n,
            costs,
            ppl,
            snakemake.input.profile_hydro,
            snakemake.input.hydro_capacities,
            carriers,
            snakemake.input.hydro_targets,
            year,
            country_to_region,
            **p,
        )

    update_p_nom_max(n)

    attach_storageunits(
        n, costs, n.buses.index, extendable_carriers["StorageUnit"], max_hours
    )

    # Exclude H2 from generic store attachment to use REMIND-specific hydrogen setup
    store_carriers = [c for c in extendable_carriers["Store"] if c != "H2"]
    attach_stores(n, costs, n.buses.index, store_carriers)

    # ------------------------------------------------------------------
    # REMIND overlay: per-node hydrogen storage/conversion (no H2 grid),
    # battery floor, and sector-coupling loads (electrolysis / EV / heat)
    # that upstream add_electricity never builds.
    # ------------------------------------------------------------------
    # H2 storage/links must precede sector coupling, which attaches demand to the H2 buses.
    if "H2" in extendable_carriers["Store"]:
        attach_hydrogen_storage_remind(
            n,
            costs,
            snakemake.input.h2_cavern,
            snakemake.params["h2_settings"],
            sector_coupling["electrolysis"],
        )

    apply_battery_e_min_pu(
        n, snakemake.params.get("battery_settings", {}), extendable_carriers
    )

    if sector_coupling["electrolysis"]["enable"]:
        attach_hydrogen_demand_remind(n, sectoral_load)

    for kind in ("pass", "freight"):
        if sector_coupling[f"EV_{kind}"]["enable"]:
            attach_ev_demand_remind(
                n,
                sectoral_load,
                sector_coupling[f"EV_{kind}"],
                snakemake.params["sector"],
                snakemake.input.transport_demand,
                snakemake.input.transport_data,
                snakemake.input.avail_profile,
                snakemake.input.dsm_profile,
                snakemake.input.temp_air_total,
                snakemake.input.fleet_file,
                year,
                kind=kind,
                country_to_region=country_to_region,
            )

    for kind in ("heatpump", "resistive"):
        if sector_coupling[kind]["enable"]:
            attach_heat_demand_remind(
                n,
                sectoral_load,
                sector_coupling[kind],
                snakemake.input.wh_share,
                snakemake.input.hourly_heat_demand_total,
                snakemake.input.hourly_water_heat_demand_total,
                year,
                kind=kind,
                cop_profiles_fn=(
                    snakemake.input.cop_profiles if kind == "heatpump" else None
                ),
            )

    # ------------------------------------------------------------------
    # Finalisation: carrier metadata, name normalisation, per-region costs.
    # ------------------------------------------------------------------
    sanitize_carriers(n, snakemake.config)
    if "location" in n.buses:
        sanitize_locations(n)

    overwrite_carrier_co2_intensities(n, costs)

    # Override component costs with per-region REMIND values and add regional CO₂ costs.
    # Must run after carrier CO₂ intensities are set above (needed for CO₂ cost calc).
    apply_regional_costs(n, costs_regional, co2_price_by_region, country_to_region)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
