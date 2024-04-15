import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

from git import Commit, GitCommandError, Repo, TagReference
from git.remote import Remote

from ktd.logging import get_logger
from ktd.util.identifier import StringIdType

logger = get_logger(__name__)


def get_repo(path: str) -> Repo:
    return Repo(path, search_parent_directories=True)


def sync_tags(repo: str | Repo, remote: str | Remote | None = None) -> None:
    fetch_tags(repo, remote, prune=True, force=True)


def fetch_tags(
    repo: str | Repo,
    remote: str | Remote | None = None,
    prune: bool = False,
    force: bool = False,
) -> None:
    repo = repo if isinstance(repo, Repo) else get_repo(repo)
    if remote:
        remotes = [remote] if isinstance(remote, Remote) else [repo.remote()]
    else:
        remotes = repo.remotes
    kwargs = {"force": True} if force else {}
    kwargs.update({"prune": True, "prune_tags": True} if prune else {})
    for remote in remotes:
        remote.fetch("refs/tags/*:refs/tags/*", **kwargs)  # type: ignore


def is_synced(repo: str | Repo, branch: str | None = None) -> bool:
    """Returns True iff the local branch is synced with the remote branch."""
    repo = repo if isinstance(repo, Repo) else get_repo(repo)
    if repo.head.is_detached:
        return False
    local_branch = repo.branches[branch] if branch else repo.active_branch
    remote_branch = local_branch.tracking_branch()
    return remote_branch is not None and local_branch.commit == remote_branch.commit


def get_first_remote_ancestor(
    repo: str | Repo,
    commit: Commit | str = "HEAD",
    remote: Remote | str = "origin",
    branch: str = "main",
) -> Commit:
    repo = repo if isinstance(repo, Repo) else get_repo(repo)
    repo.remotes[str(remote)].fetch()
    commit = commit if isinstance(commit, Commit) else repo.commit(commit)
    try:
        return repo.commit(repo.git.merge_base(commit.hexsha, f"{remote}/{branch}"))
    except GitCommandError as e:
        logger.error(
            f"No remote ancestor on {remote}/{branch} found for {commit.hexsha}"
        )
        raise e


def get_tags(
    repo: str | Repo,
    commit: Commit | None = None,
    tag_type: StringIdType | None = None,
) -> list[TagReference]:
    repo = repo if isinstance(repo, Repo) else get_repo(repo)
    return [
        tag
        for tag in sorted(repo.tags, key=lambda tag: tag.commit.committed_datetime)
        if (commit is None or tag.commit == commit)
        and (tag_type is None or tag_type.is_valid(tag.name))
    ]


def create_tag_with_type(
    repo: str | Repo,
    tag_type: StringIdType,
    message: str | None = None,
    remote: str | Remote | None = None,
) -> TagReference:
    repo = repo if isinstance(repo, Repo) else get_repo(repo)
    if remote:
        fetch_tags(repo, remote)
    existing = [str(t) for t in repo.tags if tag_type.is_valid(str(t))]
    tag_name = tag_type.next(existing=existing)
    tag = repo.create_tag(tag_name, message=message)
    if remote:
        remote = remote if isinstance(remote, Remote) else repo.remotes[remote]
        remote.push(tag.path)
    return tag


def update_to_ref(repo: str | Repo, ref: str) -> None:
    repo = repo if isinstance(repo, Repo) else get_repo(repo)
    logger.info(f"Fetching {repo.working_tree_dir}")
    for remote in repo.remotes:
        remote.fetch()
    logger.info(f"Checking out {ref}")
    repo.git.checkout(ref)


def get_remote_branches_for_commit(repo: str | Repo, commit: Commit | str) -> list[str]:
    repo = repo if isinstance(repo, Repo) else get_repo(repo)
    commit = commit if isinstance(commit, Commit) else repo.commit(commit)
    branches = repo.git.branch("-r", "--contains", commit.hexsha).splitlines()
    return [branch.strip() for branch in branches]


def is_commit_in_remote(repo: str | Repo, commit: Commit | str) -> bool:
    return bool(get_remote_branches_for_commit(repo, commit))


def get_staged_files(repo: str | Repo) -> list[str]:
    repo = repo if isinstance(repo, Repo) else get_repo(repo)
    return repo.git.diff(cached=True, name_only=True).splitlines()


def diff_vs_commit(
    repo: str | Repo,
    ref: str = "HEAD",
    include_staged_untracked: bool = True,
    include_untracked: bool = False,
    *args,
    **kwargs,
) -> str:
    """Returns the diff between the working dir and the given ref

    The result is as if the staging area had been unstaged prior to
    calling `diff`, but previously staged files, regardless of whether
    or not they're untracked, are included in the diff.
    """
    repo = repo if isinstance(repo, Repo) else get_repo(repo)
    staged_files = get_staged_files(repo)

    # use a temporary index file to avoid modifying the actual index and
    # add all files, including untracked files
    # https://git.vger.kernel.narkive.com/yH88SS4R/diff-new-files-without-using-index
    with NamedTemporaryFile(prefix="/tmp/") as temp_index_file:
        temp_index_file.close()
        shutil.copyfile(repo.index.path, temp_index_file.name)
        env = {"GIT_INDEX_FILE": temp_index_file.name}
        if include_staged_untracked:
            repo.git.add([f for f in staged_files if Path(f).exists()], env=env)
        repo.git.add(all=include_untracked, update=not include_untracked, env=env)
        return repo.git.diff("--cached", ref, *args, **kwargs, env=env)


def is_pristine(repo: str | Repo) -> bool:
    repo = repo if isinstance(repo, Repo) else get_repo(repo)
    return not (
        repo.is_dirty()
        or bool(repo.untracked_files)
        or repo.head.is_detached
        or not is_synced(repo)
    )
