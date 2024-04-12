import shlex
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep
from types import SimpleNamespace
from unittest import mock

import pytest
from git import Remote, Repo

from ktd.scm.git.util import (
    RepoState,
    create_tag_with_type,
    diff_vs_commit,
    fetch_tags,
    get_first_remote_ancestor,
    get_github_ssh_url,
    get_github_user,
    get_remote_branches_for_commit,
    get_repo,
    get_staged_files,
    get_tags,
    is_commit_in_remote,
    is_pristine,
    is_synced,
    sync_tags,
    update_to_ref,
)
from ktd.util.identifier import RUN_ID
from ktd.util.run import run


def _add_and_commit_file(repo, filename, msg=None):
    Path(repo.working_tree_dir).joinpath(filename).touch()
    repo.index.add([filename])
    repo.index.commit(msg or f"add {filename!r}")


def _create_repo_with_remote(root: str) -> Repo:
    repo = Repo.init(Path(root) / "local", initial_branch="main")
    repo.git.config("user.email", "test@test.com")
    repo.git.config("user.name", "Test")
    remote_repo_working_tree_dir = Path(root) / "remote"
    Repo.init(remote_repo_working_tree_dir, bare=True)
    Remote.create(repo, "origin", str(remote_repo_working_tree_dir))
    _add_and_commit_file(repo, "file", "initial commit")
    repo.remote().push("main", set_upstream=True)

    return repo


@pytest.fixture
def repo():
    with TemporaryDirectory() as temp_dir:
        yield _create_repo_with_remote(temp_dir)


def test_get_repo(repo):
    repo_path = repo.working_tree_dir

    assert get_repo(repo_path) == repo


def test_sync_tags(repo):
    repo.create_tag("remote-tag")
    repo.remote().push(tags=True)
    repo.create_tag("local-tag")

    sync_tags(repo, repo.remote())

    assert "remote-tag" in repo.tags
    assert "local-tag" not in repo.tags


def test_fetch_tags(repo):
    repo.create_tag("remote-tag")
    repo.remote().push(tags=True)
    repo.create_tag("local-tag")

    fetch_tags(repo, prune=False)
    assert "remote-tag" in repo.tags

    fetch_tags(repo, prune=True)
    assert "remote-tag" in repo.tags
    assert "local-tag" not in repo.tags


def test_is_synced(repo):
    assert is_synced(repo)

    _add_and_commit_file(repo, "another_file")
    assert not is_synced(repo)

    repo.remote().push()
    assert is_synced(repo)


def test_get_first_remote_ancestor(repo):
    for i in range(3):
        repo.git.branch(f"branch-{i}")
        repo.branches[f"branch-{i}"].checkout()
        _add_and_commit_file(repo, f"file_{i}")

    assert repo.branches["main"].commit == get_first_remote_ancestor(repo)


def test_get_tags(repo):
    commit_1 = repo.head.commit
    tag_1 = repo.create_tag("tag-1")
    sleep(1)
    _add_and_commit_file(repo, "another_file")
    commit_2 = repo.head.commit
    tag_2 = create_tag_with_type(repo, RUN_ID)

    assert [tag_1, tag_2] == get_tags(repo)
    assert [tag_1] == get_tags(repo, commit_1)
    assert [tag_2] == get_tags(repo, commit_2)
    assert [tag_2] == get_tags(repo, tag_type=RUN_ID)


def test_create_tag_with_type(repo):
    tag = create_tag_with_type(repo, RUN_ID, message="some message")

    assert RUN_ID.is_valid(tag.name)
    assert tag.tag and tag.tag.message == "some message"


def test_update_to_ref(repo):
    repo.git.branch("branch")
    repo.git.checkout("branch")
    _add_and_commit_file(repo, "another_file")

    update_to_ref(repo, "main")
    assert repo.head.ref.name == "main"


def test_get_remote_branches_for_commit(repo):
    assert get_remote_branches_for_commit(repo, repo.head.commit) == ["origin/main"]

    _add_and_commit_file(repo, "another_file")
    assert get_remote_branches_for_commit(repo, repo.head.commit) == []

    repo.remote().push()
    assert get_remote_branches_for_commit(repo, repo.head.commit) == ["origin/main"]


def test_is_commit_in_remote(repo):
    assert is_commit_in_remote(repo, repo.head.commit)

    _add_and_commit_file(repo, "another_file")
    assert not is_commit_in_remote(repo, repo.head.commit)

    repo.remote().push()
    assert is_commit_in_remote(repo, repo.head.commit)


def test_get_staged_files(repo):
    file = "file"
    Path(repo.working_tree_dir).joinpath(file).write_text("new content\n")

    assert len(get_staged_files(repo)) == 0

    repo.index.add(file)
    assert get_staged_files(repo) == [file]


def test_diff_vs_commit(repo):
    file = "another_file"
    Path(repo.working_tree_dir).joinpath(file).write_text("content\n")

    files = diff_vs_commit(repo)  # include_untracked=False
    assert len(files.splitlines()) == 0

    files = diff_vs_commit(repo, include_untracked=True, name_only=True)
    assert file in files.splitlines()

    repo.index.add(file)
    files = diff_vs_commit(repo, include_staged_untracked=True, name_only=True)
    assert file in files.splitlines()


def test_is_pristine(repo):
    assert is_pristine(repo)

    _add_and_commit_file(repo, "file_1")
    assert not is_pristine(repo)

    repo.remote().push()
    assert is_pristine(repo)

    file_2 = Path(repo.working_tree_dir).joinpath("file_2")
    file_2.touch()
    assert not is_pristine(repo)

    file_2.unlink()
    assert is_pristine(repo)

    repo.git.checkout("HEAD^")
    assert not is_pristine(repo)


@mock.patch("ktd.scm.git.util.run")
def test_get_github_user(run_mock):
    run_mock.return_value = SimpleNamespace(stdout='{"login": "user"}')

    user = get_github_user()

    assert user == "user"
    run_mock.assert_called_once_with(shlex.split("gh api user"), output="capture")


def test_get_github_ssh_url():
    assert get_github_ssh_url("user", "repo") == "git@github.com:user/repo"


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


def test_repo_state_replay(repo):
    with TemporaryDirectory() as temp_dir:
        # create a copy of the repo that we'll use for the replay
        repo_copy_dir = Path(temp_dir) / "repo"
        shutil.copytree(repo.working_tree_dir, repo_copy_dir)
        repo_copy = Repo(repo_copy_dir)

        # make changes to the original repo - commit a new file, modify
        # an existing one, and create a new untracked file
        _add_and_commit_file(repo, "file_1")
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
