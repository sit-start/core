from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Generator

import pytest
from git import Remote, Repo

# https://stackoverflow.com/questions/44677426/how-to-pass-arguments-to-pytest-fixtures


@pytest.fixture
def add_and_commit_file():
    def impl(repo: Repo, filename: str, msg: str | None = None) -> None:
        if repo.working_tree_dir is None:
            raise ValueError("Cannot add and commit file to a bare repo")
        Path(repo.working_tree_dir).joinpath(filename).touch()
        repo.index.add([filename])
        repo.index.commit(msg or f"add {filename!r}")

    return impl


@pytest.fixture
def repo(add_and_commit_file) -> Generator[Repo, Any, Any]:
    with TemporaryDirectory() as temp_dir:
        repo = Repo.init(Path(temp_dir) / "local", initial_branch="main")
        repo.git.config("user.email", "test@test.com")
        repo.git.config("user.name", "Test")
        remote_repo_working_tree_dir = Path(temp_dir) / "remote"
        Repo.init(remote_repo_working_tree_dir, bare=True)
        Remote.create(repo, "origin", str(remote_repo_working_tree_dir))
        add_and_commit_file(repo, "file", "initial commit")
        repo.remote().push("main", set_upstream=True)

        yield repo
