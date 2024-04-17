import os
import re
import shlex
import sys
from pathlib import Path

import typer
from typer import Argument, Option

from ktd.logging import get_logger
from ktd.util.run import run

app = typer.Typer()
logger = get_logger(__name__, format="simple")

DEFAULT_PROJECT_PATH = f"{os.environ['DEV']}/core/python"
DEFAULT_REQUIREMENTS_PATH = f"{os.environ['DEV']}/core/requirements.txt"
DEFAULT_PACKAGE_VARIANTS = ["ray[data,default,train,tune]"]


@app.command()
def update_requirements(
    project_path: str = Argument(
        DEFAULT_PROJECT_PATH,
        help="The project path.",
        show_default=True,
    ),
    requirements_path: str = Option(
        DEFAULT_REQUIREMENTS_PATH,
        help="The requirements file path.",
        show_default=True,
    ),
    package_variants: list[str] = Option(
        DEFAULT_PACKAGE_VARIANTS,
        help="Package variants to include the requirements file.",
        show_default=True,
    ),
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
