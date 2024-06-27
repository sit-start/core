import os
import re
import shlex
import sys
from pathlib import Path
from typing import Annotated

import typer
from hydra.utils import instantiate as instantiate_config
from omegaconf import OmegaConf
from typer import Argument, Option

from sitstart import PYTHON_ROOT, REPO_ROOT
from sitstart.logging import get_logger
from sitstart.ml.experiments import CONFIG_ROOT
from sitstart.ml.experiments.util import (
    load_experiment_config,
    resolve,
    validate_experiment_config,
)
from sitstart.util.container import walk
from sitstart.ml.experiments.util import register_omegaconf_resolvers
from sitstart.util.run import run

app = typer.Typer()
logger = get_logger(__name__, format="simple")

DEFAULT_PROJECT_PATH = PYTHON_ROOT
DEFAULT_REQUIREMENTS_PATH = f"{REPO_ROOT}/requirements.txt"
DEFAULT_PACKAGE_VARIANTS = ["ray[data,default,train,tune]"]


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
    exclude: Annotated[
        list[str],
        Option(
            help="Exclude a node from resolution.",
            show_default=True,
        ),
    ] = [],
    tune: Annotated[
        bool,
        Option(
            "--tune",
            help="Test a tuning config. Adds --sample-params if instantiating "
            "and --exclude=trial otherwise",
            show_default=False,
        ),
    ] = False,
    sample_params: Annotated[
        bool,
        Option(
            "--sample-params",
            help="Sample the parameter search space to resolve the param_space node.",
            show_default=False,
        ),
    ] = False,
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
            if not sample_params:
                logger.info(
                    "Setting --sample-params=True for tuning config instantiation."
                )
            sample_params = True
        elif "trial" not in exclude:
            logger.info("Adding 'trial' to --exclude for tuning config resolution.")
            exclude.append("trial")

    if not no_resolve:
        resolve(config, exclude=exclude or [], sample_params=sample_params)

    logger.info(f"Config {name!r}:\n{OmegaConf.to_yaml(config)}")

    if instantiate:
        container = OmegaConf.to_container(config)
        instantiated = []
        for top, _, obj_keys in walk(container, topdown=True):
            if "_target_" in obj_keys:
                if not any(top.startswith(node) for node in instantiated):
                    logger.info(f"Instantiating {top!r}.")
                    obj = instantiate_config(OmegaConf.select(config, top))
                    instantiated.append(top)
                    logger.info(f"Instantiated {top!r}:\n{obj}")
