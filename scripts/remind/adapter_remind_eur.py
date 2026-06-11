"""REMIND → PyPSA-Eur coupling adapter.

Subclasses ``rpycpl.CouplingAdapter``. Inherits the generic Stage-1 builders (CO2 prices,
country loads, capacity targets, and cost-parameter extraction) and supplies only the
EUR-specific pieces:

- ``prepare_capacities`` — REMIND-tech-specific capacity prep (VRE-variant merge, battery
  scaling), ported from the old ``import_REMIND_capacities``.
- ``adjust_cost_efficiencies`` — the EUR-specific ``btin`` (battery inverter) efficiency tweak.
- ``build_config_overrides`` — the PyPSA-Eur config-key structure.

Depends only on ``rpycpl`` + pandas (no PyPSA-Eur ``_helpers``), so it is unit-testable.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from rpycpl.adapters.base import CouplingAdapter

# Link-like techs whose REMIND output-capacity is converted to input-capacity (÷ efficiency).
LINK_TECHS = {"elh2", "h2turb", "btin", "elh2VRE", "h2turbVRE"}
_VRE_TO_PRIMARY = {"elh2VRE": "elh2", "h2turbVRE": "h2turb"}
_BATTERY_SCALING = {"storspv": 4.0, "storwindon": 1.2, "storwindoff": 1.2}


class RemindEurAdapter(CouplingAdapter):
    """Expose REMIND-derived inputs to the PyPSA-Eur workflow."""

    def adjust_cost_efficiencies(self, eff: pd.DataFrame) -> pd.DataFrame:
        """Square the ``btin`` (battery inverter) round-trip efficiency — EUR-specific."""
        eff = super().adjust_cost_efficiencies(eff)
        eff.loc[eff["technology"] == "btin", "value"] **= 2
        return eff

    def build_config_overrides(self) -> dict[str, Any]:
        """Return the REMIND-derived config overrides to merge onto the PyPSA-Eur config.

        Only values that cannot be k`nown until REMIND output is available: the planning horizons
        and the per-(region, year) CO2 price pathway. Everything else stays in the workflow's own
        config files (Snakemake ``--configfile`` layering merges these on top).
        """
        # Bind each piece to a named variable (no inline calls in the dict) so the
        # config-override inputs can be inspected when debugging.
        planning_horizons = list(self.config.get("planning_horizons", []))
        co2_prices = self.build_co2_prices().to_dict(orient="records")
        return {
            "scenario": {"planning_horizons": planning_horizons},
            "co2_prices": co2_prices,
        }

    def prepare_capacities(self, caps: pd.DataFrame) -> pd.DataFrame:
        """Merge VRE-coupled variants and scale battery techs before carrier mapping."""
        caps = caps.copy()
        tech = caps["technology"].astype(str)
        caps["technology"] = tech.map(lambda t: _VRE_TO_PRIMARY.get(t, t))

        tech = caps["technology"].astype(str)
        # for unidirectional btin is not included
        is_btin_present = ((tech == "btin") & (caps["value"] > 0)).any()
        is_stor = tech.isin(_BATTERY_SCALING)
        if is_btin_present:
            return caps[~is_stor].copy()
        scale = tech.map(_BATTERY_SCALING)
        caps.loc[scale.notna(), "value"] *= scale[scale.notna()]
        caps["technology"] = tech.map(lambda t: "btin" if t in _BATTERY_SCALING else t)
        return caps
