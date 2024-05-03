from typing import Annotated, Optional

import typer
from typer import Argument, Option

from sitstart.logging import get_logger
from sitstart.scm.github import create_private_fork

app = typer.Typer()
logger = get_logger(__name__)


@app.command()
def private_fork(
    repository: Annotated[
        str,
        Argument(help="The URL of the repository to fork.", show_default=False),
    ],
    fork_name: Annotated[
        Optional[str],
        Option(help="Rename the forked repository.", show_default=False),
    ] = None,
    clone: Annotated[
        bool,
        Option(help="Clone the forked repository."),
    ] = False,
    org: Annotated[
        Optional[str],
        Option(help="Create the fork in an organization.", show_default=False),
    ] = None,
):
    """Create a private fork of a repository."""
    return create_private_fork(repository, fork_name, clone, org)
