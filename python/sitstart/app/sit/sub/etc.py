import os
import re
import shlex
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from hydra.utils import instantiate as instantiate_config
from omegaconf import DictConfig, OmegaConf
from typer import Argument, Option

from sitstart import PYTHON_ROOT, REPO_ROOT
from sitstart.logging import get_logger
from sitstart.ml.experiments import CONFIG_ROOT, TRIAL_ARCHIVE_URL
from sitstart.ml.experiments.restore import get_trial_path_from_trial
from sitstart.ml.experiments.util import (
    TrialResolution,
    get_lightning_modules_from_config,
    load_experiment_config,
    register_omegaconf_resolvers,
    resolve,
    validate_experiment_config,
)
from sitstart.util.container import walk
from sitstart.util.run import run

app = typer.Typer()
logger = get_logger(__name__, format="simple")

DEFAULT_PROJECT_PATH = PYTHON_ROOT
DEFAULT_REQUIREMENTS_PATH = f"{REPO_ROOT}/requirements.txt"
DEFAULT_PACKAGE_VARIANTS = ["ray[data,default,train,tune]"]

ConfigNameArg = Annotated[
    str,
    Argument(
        help=f"Name of the experiment config in {CONFIG_ROOT}.",
        show_default=False,
    ),
]


@app.command()
def update_requirements(
    project_path: Annotated[
        str,
        Argument(help="The project path.", show_default=True),
    ] = DEFAULT_PROJECT_PATH,
    requirements_path: Annotated[
        str,
        Option(help="The requirements file path.", show_default=True),
    ] = DEFAULT_REQUIREMENTS_PATH,
    package_variants: Annotated[
        list[str],
        Option(help="Package variants to include.", show_default=True),
    ] = DEFAULT_PACKAGE_VARIANTS,
) -> None:
    """Update a Python requirements file with Pigar."""

    logger.info(f"Updating {requirements_path!r} with Pigar.")
    run(shlex.split(f"pigar generate -f {requirements_path} {project_path}"))

    if not package_variants:
        return

    invocation = " ".join([os.path.split(sys.argv[0])[-1]] + sys.argv[1:])
    requirements = Path(requirements_path).read_text().splitlines()
    requirements.insert(1, f"# Updated with the command `{invocation}`.")
    requirements = "\n".join(requirements) + "\n"

    for entry in package_variants:
        try:
            package = entry.split("[")[0]
            requirements = re.sub(f"(?m)^{package}==", f"{entry}==", requirements)
            logger.info(f"Updated {package!r} to {entry!r}.")
        except RuntimeError as e:
            logger.error(f"Failed to update {package!r} to {entry!r}: {e}")
    Path(requirements_path).write_text(requirements)


@app.command()
def archive_trial(
    trial_id: Annotated[
        str,
        Argument(help="The trial ID to archive.", show_default=False),
    ],
    run_group: Annotated[
        Optional[str],
        Option(
            help="The run group. If not specified, runs are searched based on trial ID "
            "and project name.",
            show_default=False,
        ),
    ] = None,
    config_name: Annotated[
        Optional[str],
        Option(
            help=f"Name of the experiment config in {CONFIG_ROOT}.", show_default=False
        ),
    ] = None,
    storage_path: Annotated[
        Optional[str],
        Option(
            help="The storage path. Defaults to the storage path in the config.",
            show_default=False,
        ),
    ] = None,
    project_name: Annotated[
        Optional[str],
        Option(
            help="The project name. Defaults to the project name in the config.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Archive a trial."""

    # load the config if overrides aren't provided
    if use_config := (storage_path is None or project_name is None):
        if use_config and config_name is None:
            logger.error(
                "Either config_name or project_name + storage_path is required."
            )
            return
        assert config_name is not None

        config = load_experiment_config(config_name)
        if not isinstance(config, DictConfig):
            logger.error(f"Failed to load a `DictConfig` from {config_name!r}.")
            return

        storage_path = storage_path or config.storage_path
        project_name = project_name or config.name

    assert project_name is not None
    assert storage_path is not None

    # get the trial path, searching for the run group if it's not provided
    trial_path = get_trial_path_from_trial(
        project_name=project_name,
        storage_path=storage_path,
        trial_id=trial_id,
        run_group=run_group,
    )
    cmd = f"aws s3 ls {trial_path}"
    if trial_path is None or run(shlex.split(cmd), check=False).returncode != 0:
        logger.error(f"Failed to find path for trial {trial_id!r}.")
        return

    # copy the trial to the archive
    target_trial_path = f"{TRIAL_ARCHIVE_URL}/{project_name}/{trial_id}"
    logger.info(
        f"Archiving trial {trial_id!r} from {trial_path!r} to {target_trial_path!r}."
    )
    cmd = f"aws s3 cp {trial_path}/ {target_trial_path} --recursive"
    result = run(shlex.split(cmd), check=False)
    if result.returncode != 0:
        logger.error(
            f"Archiving failed with command {cmd!r} and "
            f"return code {result.returncode}."
        )
        return


@app.command()
def test_config(
    name: Annotated[
        str,
        Argument(
            help=f"Name of the experiment config in {CONFIG_ROOT}.",
            show_default=False,
        ),
    ],
    no_resolve: Annotated[
        bool,
        Option(
            "--no-resolve",
            help="Don't resolve the config.",
            show_default=False,
        ),
    ] = False,
    tune: Annotated[
        bool,
        Option(
            "--tune",
            help="Test a tuning config. Adds --resolve-trial=sample if instantiating "
            "and --resolve-trial=exclude otherwise.",
            show_default=False,
        ),
    ] = False,
    resolve_trial: Annotated[
        TrialResolution,
        Option(help="How to process the config's trial node if resolving the config."),
    ] = TrialResolution.RESOLVE,
    instantiate: Annotated[
        bool,
        Option(
            "--instantiate",
            help="Instantiate the config.",
            show_default=False,
        ),
    ] = False,
) -> None:
    """Test loading, resolving, and instantiating a Hydra experiment config."""
    register_omegaconf_resolvers()
    config = load_experiment_config(name)
    validate_experiment_config(config)

    if tune:
        if instantiate:
            if resolve_trial != TrialResolution.SAMPLE:
                logger.info(
                    "Setting --resolve-trial=sample for tuning config instantiation."
                )
                resolve_trial = TrialResolution.SAMPLE
        elif resolve_trial == TrialResolution.RESOLVE:
            logger.info("Setting --resolve-trial=exclude for tuning config.")
            resolve_trial = TrialResolution.EXCLUDE

    if not no_resolve:
        resolve(config, resolve_trial=resolve_trial)

    logger.info(f"Config {name!r}:\n{OmegaConf.to_yaml(config)}")

    if instantiate:
        container = OmegaConf.to_container(config)
        instantiated = []

        logger.info("Instantiating config.")
        for top, _, obj_keys in walk(container, topdown=True):
            if "_target_" in obj_keys:
                if not any(top.startswith(node) for node in instantiated):
                    logger.info(f"Instantiating {top!r}.")
                    obj = instantiate_config(OmegaConf.select(config, top))
                    instantiated.append(top)
                    logger.info(f"Instantiated {top!r}:\n{obj}")

        logger.info("Fully instantiating lightning modules.")
        assert isinstance(config, DictConfig)
        data_module, training_module = get_lightning_modules_from_config(config)
        logger.info(f"Data module:\n{data_module}")
        logger.info(f"Training module:\n{training_module}")
