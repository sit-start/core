import os
import shlex
import tempfile

import typer
from typer import Argument, Option

from ktd.logging import get_logger
from ktd.scm.git.util import get_github_ssh_url, get_github_user
from ktd.util.run import run

app = typer.Typer()
logger = get_logger(__name__)


def _run(cmd, **kwargs):
    kwargs["output"] = "capture"
    return run(shlex.split(cmd), **kwargs)


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
    repo_name = repository.split("/")[-1].replace(".git", "")
    fork_name = fork_name or repo_name
    account = org or get_github_user()
    qual_fork_name = f"{account}/{fork_name}"
    fork = get_github_ssh_url(account, fork_name)
    fork_dir = f"{os.getcwd()}/{fork_name}"

    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = f"{temp_dir}/{repo_name}.git"
        logger.info(f"Creating a local bare clone of {repository!r}.")
        _run(f"git clone --bare {repository}", cwd=temp_dir)
        logger.info(f"Creating a new private repository {qual_fork_name!r}.")
        _run(f"gh repo create {account}/{fork_name} --private")
        logger.info(f"Mirror-pushing to {qual_fork_name!r}.")
        _run(f"git push --mirror {fork}", cwd=repo_dir)

    if clone:
        logger.info(f"Cloning {qual_fork_name!r}.")
        _run(f"git clone {fork}")
        logger.info(f"Adding {repository!r} as a read-only remote.")
        _run(f"git remote add upstream {repository}", cwd=fork_dir)
        _run("git remote set-url --push upstream DISABLE", cwd=fork_dir)
