# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

import logging
import os
import socket
from datetime import datetime
from typing import Any

from _helpers import (
    check_gurobi_license,
    configure_logging,
    is_tunnel_alive,
    setup_gurobi_tunnel_and_env,
)

logger = logging.getLogger(__name__)

# Provided by Snakemake at runtime when used via `script:`.
snakemake: Any


if __name__ == "__main__":
    configure_logging(snakemake)

    solving_cfg = snakemake.config.get("solving", {})
    solver_name = solving_cfg.get("solver", {}).get("name")
    tunnel_cfg = solving_cfg.get("gurobi_hpc_tunnel", None)

    should_setup_tunnel = (
        solver_name == "gurobi"
        and tunnel_cfg is not None
        and tunnel_cfg.get("use_tunnel", False)
    )

    if should_setup_tunnel:
        logger.info("Setting up Gurobi tunnel before solve rule.")
        setup_gurobi_tunnel_and_env(tunnel_cfg, logger=logger)

        if not is_tunnel_alive(tunnel_cfg):
            logger.warning("Tunnel not alive after first setup; retrying once.")
            setup_gurobi_tunnel_and_env(tunnel_cfg, logger=logger)

        if not is_tunnel_alive(tunnel_cfg):
            raise RuntimeError("Failed to establish Gurobi tunnel.")

        timeout = int(tunnel_cfg.get("timeout_s", 10))
        if not check_gurobi_license(attempts=1, timeout=timeout):
            raise RuntimeError("Gurobi license check failed in pre-solve rule.")
    else:
        logger.info(
            "Skipping tunnel setup (solver is not gurobi or no tunnel config/use_tunnel=false)."
        )

    os.makedirs(os.path.dirname(snakemake.output.ready_marker), exist_ok=True)
    with open(snakemake.output.ready_marker, "w") as marker:
        marker.write(
            f"host={socket.gethostname()}\n"
            f"pid={os.getpid()}\n"
            f"timestamp_utc={datetime.utcnow().isoformat()}\n"
            f"solver={solver_name}\n"
        )
