# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

from scripts.remind_installed_capacity_constraints import (
    add_installed_capacity_lower_bound_constraints,
)


def remind_extra_constraints(n, snapshots, snakemake):
    """Add REMIND-specific extra constraints."""
    add_installed_capacity_lower_bound_constraints(n, snakemake)
