import shlex
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from git import Repo

from sitstart.scm.git.repo_state import RepoState
from sitstart.util.run import run


def test_repo_state_from_repo(repo):
    repo_state = RepoState.from_repo(repo)

    assert repo_state.path == str(repo.working_dir)
    assert repo_state.url == repo.remotes["origin"].url
    assert repo_state.remote == "origin"
    assert repo_state.branch == "main"

    assert repo_state.commit == repo.head.commit.hexsha
    assert repo_state.remote_commit == repo.head.commit.hexsha
    assert not repo_state.uncommitted_changes
    assert not repo_state.local_commits


def test_repo_state_from_dict(repo):
    repo_state = RepoState.from_repo(repo)
    assert repo_state == RepoState.from_dict(repo_state.__dict__)


def test_repo_state_summary(repo, add_and_commit_file):
    repo_state = RepoState.from_repo(repo)
    commit = repo.head.commit.hexsha[:7]
    assert repo_state.summary == commit

    add_and_commit_file(repo, "another_file")
    repo_state = RepoState.from_repo(repo)
    local_commit = repo.head.commit.hexsha[:7]
    assert repo_state.summary == f"{commit}..{local_commit}"

    Path(repo.working_tree_dir).joinpath("untracked_file").write_text("content\n")
    repo_state = RepoState.from_repo(repo)
    assert repo_state.summary[-1] == "+"


def test_repo_state_replay(repo, add_and_commit_file):
    with TemporaryDirectory() as temp_dir:
        # create a copy of the repo that we'll use for the replay
        repo_copy_dir = Path(temp_dir) / "repo"
        shutil.copytree(repo.working_tree_dir, repo_copy_dir)
        repo_copy = Repo(repo_copy_dir)

        # check that the replay is a no-op when there are no changes
        repo_state = RepoState.from_repo(repo)
        repo_state.replay(repo_copy)
        assert repo_copy.head.commit.hexsha == repo.head.commit.hexsha

        # make changes to the original repo - commit a new file, modify
        # an existing one, and create a new untracked file
        add_and_commit_file(repo, "file_1")
        Path(repo.working_tree_dir).joinpath("file").write_text("new content\n")
        Path(repo.working_tree_dir).joinpath("file_2").write_text("content\n")

        # create the repo_state object and replay it on the copy
        repo_state = RepoState.from_repo(repo)
        repo_state.replay(repo_copy)

        # ensure the directory contents are equal
        cmd = shlex.split(
            f"diff -r {repo.working_tree_dir} {repo_copy_dir} --exclude .git"
        )
        assert len(run(cmd, output="capture").stdout) == 0
