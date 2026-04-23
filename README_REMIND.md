# Installation

* From PyPSA-Eur v2026.02.0, install pixi (see [here](https://pixi.prefix.dev/latest/installation/))
* Call `pixi shell`
* Additional packages to install for REMIND: `gamspy` (probably needs valid GAMS license) via `pip install gamspy`
* Setup Gurobi license if necessary
    * On PIK HPC the license activation requires a tunnel to the login node, because the compute nodes do not have internet access
    * See `setup_gurobi_tunnel_REMIND` in `REMIND_coupling.smk` for details
    * The necessary paths for the license are set in the respective script
    * The tunnel requires a public-private key pair to be accessible from compute nodes (private key) and login node (corresponding public key), e.g. on login node with your user create a new key-pair using:
        ```
            ssh-keygen -t ed25519 -f ~/.ssh/id_rsa.cluster_internal_exchange -C "$USER@cluster_internal_exchange"
        ```
        and add the contents of `~/.ssh/id_rsa_cluster_internal_exchange.pub` to your `~/.ssh/authorized_keys` file in a new line
        with the appropriate entry in `~/.ssh/config`, e.g.
        ```
        Host login01
            Hostname login01
            User <your username>
            PubKeyAuthentication yes
            IdentitiesOnly yes
            IdentityFile ~/.ssh/id_rsa.cluster_internal_exchange
        ```
* To prepare required inputs for PyPSA-Eur run `snakemake -c 4 -s Snakefile_REMIND download_and_prepare` once
    * This will download all required datasets (using rules in `retrieve.smk`) and create some load on the login node to create the renewable profiles
* Running PyPSA-Eur on the PIK HPC requires the cluster profile in `pik_hpc_profile/config.yaml`
    * The command then becomes `snakemake --profile pik_hpc_profile ...`
    * Rules can be designated to groups that are then submitted as one job (useful for a sequence of rules where the input of one is the output of the other), see group `"iy"`
    * This profile submits a collection of SLURM jobs automatically
    * Any rules defined under `localrules` in `Snakefile_REMIND` will not be submitted to SLURM but run on the login node


# Configurations

* Technology mapping between REMIND-EU and PyPSA-Eur is inferred based on `config/technology_cost_mapping.csv` and complemented with manual adjustments via `get_technology_mapping(...)` in `scripts/_helpers.py` (e.g. for `offwind-ac`, `offwind-dc` and `ror`, `PHS`)
    * The technology / cost mapping is mainly used to extract costs and other technology data from REMIND
    * It is further used to map back the results from PyPSA-Eur to REMIND-EU in `scripts/export_to_REMIND.py`
    * The current setup and file structure is build around the rule `import_REMIND_costs`, which makes using the mapping a bit more complicated for other applications.
    * Changes to the `technology_cost_mapping.csv` file should be made carefully (and results checked). Avoid mappings between REMIND and PyPSA-Eur of nature `m:n`; these mappings work e.g. for lignite and coal, but only as long as all mapped technologies are identical, i.e. lignite and coal map from PyPSA-Eur to exactly the same REMIND-EU technologies.
* Region mappings are in `config/regionmapping_21_EU11.csv`

# Updating to newest PyPSA-Eur release

All coupling rules and scripts are defined in `REMIND_coupling.smk`. Duplicated code and overlap has been avoided as much as possible. Still, updating to the newest release will likely lead to merge conflicts and potential other issues.

* Git stuff:
    * Clone REMIND-coupled PyPSA-Eur into new folder, e.g. `git clone git@github.com:aodenweller/pypsa-eur.git pypsa-eur_v2026.02.0`
    * Add PyPSA upstream, `git remote add upstream git@github.com:PyPSA/pypsa-eur.git`
    * Switch to previous branch, e.g. `git checkout dev/pypsa_v2026.02.0_remind` 
    * Create and checkout new branch, e.g. `git checkout -b dev/pypsa_v2027.01.0_remind`
    * Fetch from upstream, `git fetch upstream`, this also gets the release tags
    * Merge release into branch, e.g. `git merge v2027.01.0`
* Use a programm to compare the two repository folders, e.g. "Meld" or "GitLens" for VSCode and see what changes were made and whether the should be compatible with the code-base changes made for the REMIND coupling
* Special attention should be given to the following files (open in compare view side-by-side!)
    * `configs/config.default.yaml` -> configuration changes might be relevant to be transfered into `config/config.remind.yaml`
    * `Snakefile`-> changes might be relevant to be transfered into `Snakefile_REMIND`
    * Changes to `.smk` files (relevant are those that are included in `Snakefile_REMIND`)
    * `add_electricity.py`: Check if anything changed that needs to be transferred to `add_electricity_sector_REMIND.py`
    * `solve_network.py`: Check if minimum capacity constraint is still compatible with changes made
* Check release notes!

## Coupling parameters

### REMIND to PyPSA-Eur

* Dedicated import scripts creating input files for PyPSA-EUR from the REMIND export (`REMIND2PyPSAEUR.gdx`)
* Output form is imitating original PyPSA-EUR files input format, trying to thereby create drop-in replacements for the original files
* New files are then used in the PyPSA-EUR rules, by modifying the specified input files (but ideally not the model scripts) in the Snakefile / <rules>.smk files
* Installed capacities for generators, links and stores are implemented via a constraint that is added in `solve_network.py`
    * This enforces the sum of capacity for each regions and technology group to be greater or equal the capacity from REMIND
    * The config parameter `everywhere_powerplants` must be used to make sure that powerplants are always extendable in all locations, even if no such powerplant exists at a given location
* Costs
    * Currently no regional costs in PyPSA-Eur
    * CO2 price enters in rule `prepare_network_REMIND` via reading from csv file for each year.
    * TODO in future: Implement regional costs, maybe even implement some kind of annualised foresight costs.
* Load
    * Sectoral demand is processed in rule `add_electricity_sector_REMIND`, adjusting the residual demand accordingly

### PyPSA-Eur to REMIND
Parameters are extracted with a specific snakemake rule and script `export_to_REMIND.py`.

Extracted values are aggregated by REMIND region and mapped to REMIND technologies.

* Some mapping is 1:n from PyPSA-EUR:REMIND, see the mapping in the file.
* some extracted parameters are weighted by the installed capacity in *REMIND* in from the run before PyPSA-EUR is called, where the weighing is between the different technologies mapped from 1 PyPSA-EUR tech -> n REMIND techs. Currently weighted are: generation shares, installed capacities

## Changes to config.yaml (incomplete!)

* TODO: Make config.remind.yaml merely overwrite config.default.yaml
* Increase solar potential
    * Reason: Limits to maximum potential were making model in some situations where REMIND-EU wanted to have a higher than permissible build-out of PV in the model.
    The original value of 1.7 was with 1% land availability, the new value represents 3% land availability, following the estimate logic also used in the ENSPRESSO dataset by JRC.
* Special case hydro:
    * Hydro power capacities a inconsistent between PyPSA-Eur (taken from powerplantmatching) and REMIND, there is also an inconsistency on what is considered "hydro power" in both models (dam, ror, PHS in PyPSA-Eur) vs. (dam, ror in REMIND-EU).
    * We currently adjust capacities and the inflow time series in PyPSA-Eur given input from REMIND to make sure that capacities and capacity factors match

# Troubleshooting

* Insufficient memory:
    * Adjust this line in `rules/common.smk`
    ```
        return int(factor * (5000 + 195 * int(w.clusters)))
    ```
    to more base memory per `solve_network` rule, default PyPSA-Eur is:
    ```
        return int(factor * (10000 + 195 * int(w.clusters)))
    ```
* `KeyError` when calling `snakemake` with `--dry-run / -n`: This can happen due to `group-components` in `cluster_config/config.yaml` and be ignored. The non-dry-run of `snakemake` should run without issues
* Random filesystem errors with `snakemake`: This can happen on the cluster when the filesystem index is not updated fast enough. Also see [this GitHub issue](Ihttps://github.com/snakemake/snakemake/issues/39).
    * Workaround 1: Add
    ```
        for f in files:
        os.listdir(os.path.dirname(f))
    ```
    to the function `wait_for_files` in  `io.py` in the `snakemake` directory of the environment, e.g. `/p/tmp/adrianod/software/micromamba_20240118/envs/pypsa-eur-20240118/lib/python3.10/site-packages/snakemake`. This needs to be repeated for every new environment.
        * Actually, this seems to lead to another weird issue where `snakemake` keeps resubmitting jobs, although they were finished. 
    * Workaround 2: Open another session and run an infinite loop
    ```
    while :; do ls $OUTDIR ; sleep 10; done
    ```
    in the `pypsa-eur/results/<scenario>/<iteration>` directory