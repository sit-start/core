import json
import shlex
import os
import tempfile


from ktd.logging import get_logger
from ktd.util.run import run

logger = get_logger(__name__)


def _run(cmd, **kwargs):
    kwargs["output"] = "capture"
    return run(shlex.split(cmd), **kwargs)


def get_user():
    return json.loads(_run("gh api user").stdout)["login"]


def get_ssh_url(account: str, repo: str):
    return f"git@github.com:{account}/{repo}.git"


def create_private_fork(
    repo_url: str,
    fork_name: str | None = None,
    clone: bool = False,
    org: str | None = None,
) -> None:
    """Create a private fork of a repository.

    Parameters:
        repo_url: The URL of the repository to fork.
        fork_name: Rename the forked repository.
        clone: Clone the forked repository.
        org: Create the fork in an organization.
    """
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    fork_name = fork_name or repo_name
    account = org or get_user()
    qual_fork_name = f"{account}/{fork_name}"
    fork = get_ssh_url(account, fork_name)
    fork_dir = f"{os.getcwd()}/{fork_name}"

    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = f"{temp_dir}/{repo_name}.git"
        logger.info(f"Creating a local bare clone of {repo_url!r}.")
        _run(f"git clone --bare {repo_url}", cwd=temp_dir)
        logger.info(f"Creating a new private repository {qual_fork_name!r}.")
        _run(f"gh repo create {account}/{fork_name} --private")
        logger.info(f"Mirror-pushing to {qual_fork_name!r}.")
        _run(f"git push --mirror {fork}", cwd=repo_dir)

    if clone:
        logger.info(f"Cloning {qual_fork_name!r}.")
        _run(f"git clone {fork}")
        logger.info(f"Adding {repo_url!r} as a read-only remote.")
        _run(f"git remote add upstream {repo_url}", cwd=fork_dir)
        _run("git remote set-url --push upstream DISABLE", cwd=fork_dir)
