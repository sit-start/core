from pathlib import Path
from time import sleep

import pytest
from git import GitCommandError

from ktd.scm.git.util import (
    create_tag_with_type,
    diff_vs_commit,
    fetch_tags,
    get_first_remote_ancestor,
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


def test_is_synced(repo, add_and_commit_file):
    base_commit = repo.head.commit.hexsha
    assert is_synced(repo)

    add_and_commit_file(repo, "another_file")
    assert not is_synced(repo)

    repo.remote().push()
    assert is_synced(repo)

    repo.git.checkout(base_commit)
    assert not is_synced(repo)


def test_get_first_remote_ancestor(repo, add_and_commit_file):
    for i in range(3):
        repo.git.branch(f"branch-{i}")
        repo.branches[f"branch-{i}"].checkout()
        add_and_commit_file(repo, f"file_{i}")

    assert repo.branches["main"].commit == get_first_remote_ancestor(repo)

    repo.git.checkout("orphan-branch", orphan=True)
    add_and_commit_file(repo, "another_file")
    with pytest.raises(GitCommandError):
        get_first_remote_ancestor(repo)


def test_get_tags(repo, add_and_commit_file):
    commit_1 = repo.head.commit
    tag_1 = repo.create_tag("tag-1")
    sleep(1)
    add_and_commit_file(repo, "another_file")
    commit_2 = repo.head.commit
    tag_2 = create_tag_with_type(repo, RUN_ID)

    assert [tag_1, tag_2] == get_tags(repo)
    assert [tag_1] == get_tags(repo, commit_1)
    assert [tag_2] == get_tags(repo, commit_2)
    assert [tag_2] == get_tags(repo, tag_type=RUN_ID)


def test_create_tag_with_type(repo):
    tag = create_tag_with_type(repo, RUN_ID, message="some message", remote="origin")

    assert RUN_ID.is_valid(tag.name)
    assert tag.tag and tag.tag.message == "some message"


def test_update_to_ref(repo, add_and_commit_file):
    repo.git.branch("branch")
    repo.git.checkout("branch")
    add_and_commit_file(repo, "another_file")

    update_to_ref(repo, "main")
    assert repo.head.ref.name == "main"


def test_get_remote_branches_for_commit(repo, add_and_commit_file):
    assert get_remote_branches_for_commit(repo, repo.head.commit) == ["origin/main"]

    add_and_commit_file(repo, "another_file")
    assert get_remote_branches_for_commit(repo, repo.head.commit) == []

    repo.remote().push()
    assert get_remote_branches_for_commit(repo, repo.head.commit) == ["origin/main"]


def test_is_commit_in_remote(repo, add_and_commit_file):
    assert is_commit_in_remote(repo, repo.head.commit)

    add_and_commit_file(repo, "another_file")
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


def test_is_pristine(repo, add_and_commit_file):
    assert is_pristine(repo)

    add_and_commit_file(repo, "file_1")
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
