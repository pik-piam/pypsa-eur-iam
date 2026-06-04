"""
Retrieve SSP population and GDP projections from the IIASA scenario database.

Calls the IIASA SSP REST API directly via httpx (no ixmp4/pyam dependency).
Saves country-level population and GDP|PPP time series as CSV files used by
``downscale_REMIND_demand`` to disaggregate REMIND regional demand.

Country names returned by the API (English full names) are mapped to ISO-2 codes
using the ``country_converter`` package. Data is at 5-year intervals matching
REMIND's time resolution — no interpolation is applied.

API pagination
--------------
The IIASA API caps responses at 25 000 rows when queried without filters, silently
truncating ~30 countries (GB, SE, CH, TR, …) that fall beyond the limit. The fix
is to pass model and scenario as server-side filters in the request body, which
reduces the response to ~3 000 rows. The API also returns a ``total`` count of
matching rows; the script raises an error if ``total`` exceeds the received count
so that any future truncation is detected immediately rather than silently.

Outputs
-------
- ``population.csv``: columns [iso2, year, value] (population in millions)
- ``gdp.csv``: columns [iso2, year, value] (GDP|PPP in billion USD 2005)
"""

import logging

import country_converter as coco
import httpx
import pandas as pd
from _helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)

IIASA_URL = "https://ixmp4.ece.iiasa.ac.at/v1/ssp/iamc/datapoints/"
IIASA_PARAMS = {"join_parameters": "true", "join_runs": "true", "table": "true"}


def _fetch_variable(variable: str, model: str, scenario: str) -> pd.DataFrame:
    """Download one IAMC variable filtered to model/scenario from the IIASA SSP platform.

    Filtering server-side keeps the response well below the 25 000-row API cap.
    The ``total`` field in the response is checked against the received row count;
    a RuntimeError is raised if they differ so truncation is never silent.
    """
    logger.info("Fetching '%s' (%s / %s) from IIASA SSP API …", variable, model, scenario)
    resp = httpx.patch(
        IIASA_URL,
        params=IIASA_PARAMS,
        json={
            "variable": {"name": variable},
            "run": {"model": {"name": model}, "scenario": {"name": scenario}},
        },
        timeout=120,
    )
    resp.raise_for_status()
    payload = resp.json()["results"]
    df = pd.DataFrame(payload["data"], columns=payload["columns"])

    total = payload.get("total")
    if total is not None and total != len(df):
        raise RuntimeError(
            f"IIASA API returned {len(df)} of {total} total rows for '{variable}' "
            f"({model} / {scenario}) — response was truncated. "
            "Check API filter syntax or implement pagination."
        )
    logger.info("  received %d rows", len(df))
    return df


def _retrieve_variable(
    variable: str,
    model: str,
    scenario: str,
    label: str,
) -> pd.DataFrame:
    """Fetch variable for specific model/scenario, map regions to ISO-2."""
    df = _fetch_variable(variable, model, scenario)

    cc = coco.CountryConverter()
    df["iso2"] = cc.pandas_convert(
        pd.Series(df["region"]), to="ISO2", not_found=None
    )
    n_before = len(df)
    df = df.dropna(subset=["iso2"])
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.debug("Dropped %d non-country entries for %s", n_dropped, label)

    result = (
        df[["iso2", "step_year", "value"]]
        .rename(columns={"step_year": "year"})
        .groupby(["iso2", "year"], as_index=False)["value"]
        .sum()
        .sort_values(["iso2", "year"])
        .reset_index(drop=True)
    )
    logger.info("Retrieved %s for %d country-year combinations.", label, len(result))
    return result


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "retrieve_ssp_data",
            configfiles="config/config.remind.yaml",
        )

    configure_logging(snakemake)

    params = snakemake.params
    scenario = params.ssp_scenario

    population = _retrieve_variable(
        variable="Population",
        model=params.ssp_population_model,
        scenario=scenario,
        label="population",
    )
    population.to_csv(snakemake.output.population, index=False)
    logger.info("Wrote population data to %s", snakemake.output.population)

    gdp = _retrieve_variable(
        variable="GDP|PPP",
        model=params.ssp_gdp_model,
        scenario=scenario,
        label="GDP|PPP",
    )
    gdp.to_csv(snakemake.output.gdp, index=False)
    logger.info("Wrote GDP data to %s", snakemake.output.gdp)
