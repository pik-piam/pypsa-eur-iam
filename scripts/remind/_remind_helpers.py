# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""REMIND coupling and Gurobi HPC tunnel helpers."""

import functools
import logging
import multiprocessing
import os
import subprocess
from pathlib import Path

import gurobipy
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_TUNNEL_PORT = 1080


@functools.lru_cache
def build_capacity_technology_mapping(fn: str | Path) -> pd.DataFrame:
    """
    Return a 1:1 mapping between PyPSA-Eur carriers and REMIND technologies with a capacity target.

    Filters technologies (from technology_mapping_REMIND.yaml) to those whose ``iam_name``
    is in ``iampypsa.io.build_capacity_reporting_technologies()`` — capacity and cost sourcing
    don't always coincide (e.g. csp/geothermal report capacity but not investment cost).

    Parameters
    ----------
    fn : str or Path
        Path to the technology mapping YAML (model overlay).

    Returns
    -------
    pd.DataFrame
        Columns "PyPSA-Eur" and "REMIND-EU".
    """
    from iampypsa.io import build_capacity_reporting_technologies, load_technology_parameters
    from iampypsa.io.technology_mapping import iam_name

    technologies = load_technology_parameters(fn)["technologies"]
    reports_capacity = build_capacity_reporting_technologies()
    rows = [
        {"PyPSA-Eur": tech, "REMIND-EU": iam_name(tech, spec)}
        for tech, spec in technologies.items()
        if iam_name(tech, spec) in reports_capacity
    ]

    return pd.DataFrame(rows, columns=["PyPSA-Eur", "REMIND-EU"]).drop_duplicates().reset_index(drop=True)


def build_remind_coupler(remind_data_path: str | Path, countries: set, *, config: dict = None):
    """
    Construct a backend-appropriate REMIND Coupler for ``remind_data_path``.

    Centralizes the region-mapping, loader, symbol-spec, and Gdx-vs-Iamc backend selection.

    Parameters
    ----------
    remind_data_path : str or Path
        Path to the REMIND output file (GDX or IAMC ``.mif``/``.csv``).
    countries : set
        Configured PyPSA-Eur countries; used to restrict ``model_regions`` to REMIND regions
        that map to at least one of them.
    config : dict, optional
        Passed through to the Coupler's ``config``; defaults to ``{}``.

    Returns
    -------
    iampypsa.couplers.base.Coupler
        A ``RemindGdxCoupler`` or ``RemindIamcCoupler`` instance, per ``loader.backend``.
    """
    from iampypsa import RemindGdxCoupler, RemindIamcCoupler
    from iampypsa.couplers.remind import read_region_map as get_region_mapping
    from iampypsa.io import RemindLoader
    from iampypsa.io.remind_symbols import load_symbol_specs

    remind_couplers = {"gdx": RemindGdxCoupler, "iamc": RemindIamcCoupler}

    region_mapping = get_region_mapping(source="country", target="model_region")
    mapped_regions = sorted(
        {r for c, rs in region_mapping.items() if c in countries for r in rs if r}
    )
    loader = RemindLoader(remind_data_path)
    symbols = load_symbol_specs(backend=loader.backend)
    coupler_cls = remind_couplers[loader.backend]
    return coupler_cls(
        loader, symbols, region_mapping, config or {}, model_regions=mapped_regions,
    )


def is_tunnel_alive(tunnel_config: dict):
    """Check if the SSH tunnel is running by checking if the port is in use."""
    port = tunnel_config.get("tunnel_port", DEFAULT_TUNNEL_PORT)
    result = subprocess.run(
        f"netstat -tlnp 2>/dev/null | grep :{port}",
        shell=True,
        stdout=subprocess.PIPE
    )
    return result.returncode == 0

def setup_gurobi_tunnel_and_env(
    tunnel_config: dict, logger: logging.Logger = None, attempts=4
) -> subprocess.Popen:
    """A utility function to set up the Gurobi environment variables and establish an
    SSH tunnel on HPCs. Otherwise the license check will fail if the compute nodes do
     not have internet access or a token server isn't set up

    Args:
        tunnel_config (dict): ``solving.gurobi_hpc_tunnel`` config. ``use_tunnel``
            and ``tunnel_port``/``timeout_s`` (defaulting to ``DEFAULT_TUNNEL_PORT``/
            60s) are optional; ``login_host`` and the HPC-specific paths
            ``ssl_cert_file``, ``grb_cafile``, ``gurobi_home``, ``grb_license_file``
            are required whenever ``use_tunnel`` is true — there's no cluster-agnostic
            default for any of them.
        logger (logging.Logger, optional): Logger. Defaults to None.
        attempts (int, optional): ssh connection attemps. Defaults to 4.
    """
    if not tunnel_config.get("use_tunnel", False):
        return
    logger.info("setting up tunnel")
    user = os.getenv("USER")
    port = tunnel_config.get("tunnel_port", DEFAULT_TUNNEL_PORT)
    login_host = tunnel_config["login_host"]
    timeout = tunnel_config.get("timeout_s", 60)

    # bash commands for tunnel: reduce pipe err severity (too high from snakemake)
    pipe_err = "set -o pipefail; "
    ssh_command = f"ssh -vvv -fN -D {port} -o ConnectTimeout={timeout} {user}@{login_host}"
    logger.info(f"Attempting ssh tunnel to login node {login_host}")
    socks_proc = subprocess.Popen(
        pipe_err + ssh_command,
        shell=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )

    try:
        stdout, stderr = socks_proc.communicate(timeout=timeout + 2)
        err = stderr.decode()
        logger.info(f"ssh err returns {str(err)}")
        logger.info(f"ssh stdout returns {str(stdout)}")
        if err.find("Permission") != -1 or err.find("Could not resolve hostname") != -1:
            socks_proc.kill()
        else:
            logger.info("Gurobi Environment variables & tunnel set up successfully at attempt {i}.")
    except subprocess.TimeoutExpired:
        logger.error("SSH tunnel communication timed out.")

    os.environ["https_proxy"] = f"socks5://127.0.0.1:{port}"
    os.environ["SSL_CERT_FILE"] = tunnel_config["ssl_cert_file"]
    os.environ["GRB_CAFILE"] = tunnel_config["grb_cafile"]

    os.environ["GUROBI_HOME"] = tunnel_config["gurobi_home"]
    os.environ["PATH"] += f":{os.environ['GUROBI_HOME']}/bin"
    if "LD_LIBRARY_PATH" in os.environ:
        os.environ["LD_LIBRARY_PATH"] += f":{os.environ['GUROBI_HOME']}/lib"
    os.environ["GRB_LICENSE_FILE"] = tunnel_config["grb_license_file"]
    os.environ["GRB_CURLVERBOSE"] = "1"
    os.environ["GRB_SERVER_TIMEOUT"] = "10"

    return socks_proc

def _check_gurobi_license_subprocess() -> bool:
    """
    Subprocess function to check Gurobi license availability.
    This function will start the Gurobi environment to verify if a license is available.

    Returns:
        bool: True if the license check succeeded, False otherwise.
    """
    try:
        env = gurobipy.Env(empty=True)
        env.start()
        logger.info("Gurobi license is available.")
        env.dispose()
        return True
    except gurobipy.GurobiError as e:
        logger.error(f"Error checking Gurobi license: {e}")
        return False


def check_gurobi_license(attempts=1, timeout=10) -> bool:
    """
    Checks the availability of the Gurobi license in a subprocess with timeout.

    Args:
        attempts (int): Number of attempts.
        timeout (int): Time to wait before retrying (in seconds).

    Returns:
        bool: True if the license is available, False if the check times out.
    """
    logger.info("Checking Gurobi license availability...")

    for _ in range(attempts):
        process = multiprocessing.Process(target=_check_gurobi_license_subprocess)
        process.start()

        process.join(timeout=timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            logger.warning("License check timeout. Retrying...")
        else:
            if process.exitcode == 0:
                return True
            else:
                logger.warning("License not available during subprocess check. Retrying...")

    return False
