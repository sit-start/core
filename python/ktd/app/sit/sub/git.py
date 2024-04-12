import typer
from typer import Argument, Option

from ktd.logging import get_logger
from ktd.scm.github import create_private_fork

app = typer.Typer()
logger = get_logger(__name__)


@app.command()
def private_fork(
    repository: str = Argument(
        ...,
        help="The URL of the repository to fork.",
        show_default=False,
    ),
    fork_name: str = Option(
        None,
        help="Rename the forked repository.",
        show_default=False,
    ),
    clone: bool = Option(
        False,
        help="Clone the forked repository.",
    ),
    org: str = Option(
        None,
        help="Create the fork in an organization.",
        show_default=False,
    ),
):
    """Create a private fork of a repository."""
    return create_private_fork(repository, fork_name, clone, org)
