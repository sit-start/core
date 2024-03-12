from tempfile import NamedTemporaryFile

from git import Repo, Commit, TagReference
from git.remote import Remote
from ktd.logging import get_logger
from ktd.util.identifier import RUN_ID, StringIdType

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


def get_tags(
    repo: str | Repo,
    remote: str | Remote | None = None,
    commit: Commit | None = None,
    remote_only: bool = False,
) -> list[TagReference]:
    """Returns all tags sorted by increasing commit date"""
    repo = repo if isinstance(repo, Repo) else get_repo(repo)
    if remote:
        fetch_tags(repo, remote, prune=remote_only)
    tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)
    if commit:
        tags = filter(lambda t: t.commit == commit, tags)
    return list(tags) if tags else []


def get_tag_with_type(
    repo: str | Repo,
    tag_type: StringIdType,
    remote: str | Remote | None = None,
    commit: Commit | None = None,
    remote_only: bool = False,
) -> TagReference | None:
    """Returns the most recent tag of the given type or None."""
    all_tags = get_tags(repo, remote, commit, remote_only=remote_only)
    return next((t for t in reversed(all_tags) if tag_type.is_valid(t.name)), None)


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
    logger.info(f"Fetching {repo.working_dir}")
    for remote in repo.remotes:
        remote.fetch()
    logger.info(f"Checking out {ref}")
    repo.git.checkout(ref)


def get_repo_state(repo: str | Repo) -> dict[str, str | list[str] | None]:
    repo = repo if isinstance(repo, Repo) else get_repo(repo)
    local_branch = repo.active_branch if not repo.head.is_detached else None
    commit = local_branch.commit if local_branch else repo.head.commit
    remote_branch = local_branch.tracking_branch() if local_branch else None
    remote = remote_branch.remote_name if remote_branch else "origin"
    # Note: run and tags below may include local-only tags; can specify
    # remote_only=True to exclude these in both invocations below
    run = get_tag_with_type(repo, RUN_ID, remote, commit)
    tags = [t.name for t in get_tags(repo, remote, commit)]

    return {
        "path": str(repo.working_dir),
        "url": repo.remotes[remote].url if remote else None,
        "branch": local_branch.name if local_branch else None,
        "commit": commit.hexsha,
        "tags": tags or None,
        "run": run.name if run else None,
        "remote": remote,
        "remote_branch": remote_branch.name if remote_branch else None,
        "remote_commit": remote_branch.commit.hexsha if remote_branch else None,
        "status": repo.git.status("--porcelain") or None,
        "diff": diff_vs_commit(repo, include_untracked=True) or None,
    }


def diff_vs_commit(
    repo: str | Repo,
    ref: str = "HEAD",
    include_untracked: bool = False,
    *args,
    **kwargs,
) -> str:
    repo = repo if isinstance(repo, Repo) else get_repo(repo)
    """Returns the diff between the working dir and the given ref
    
    The result is as if the staging area had been restored prior to
    calling diff.
    """
    # use a temporary index file to avoid modifying the actual index and
    # add all files, including untracked files
    # https://git.vger.kernel.narkive.com/yH88SS4R/diff-new-files-without-using-index
    with NamedTemporaryFile(prefix="/tmp/") as temp_index_file:
        temp_index_file.close()
        env = {"GIT_INDEX_FILE": temp_index_file.name}
        repo.git.add("--all" if include_untracked else "--update", env=env)
        return repo.git.diff("--cached", ref, *args, **kwargs, env=env)


def get_short_repo_description(
    repo: str | Repo, state: dict[str, str | list[str] | None] | None = None
) -> str:
    repo = repo if isinstance(repo, Repo) else get_repo(repo)
    state = state if state else get_repo_state(repo)
    tag = state["tags"][-1] if state["tags"] else None
    commit = state["commit"]
    assert commit, "No commit found"

    name = state["run"] or state["branch"] or tag or commit[:6]

    dirty = repo.is_dirty()
    has_untracked = bool(repo.untracked_files)
    detached = repo.head.is_detached
    not_synced = not detached and not is_synced(repo)

    suffix = dirty * "*" + has_untracked * "?" + detached * "^" + (not_synced) * "!"
    return str(name) + suffix


def is_pristine(repo: str | Repo) -> bool:
    repo = repo if isinstance(repo, Repo) else get_repo(repo)
    return not (
        repo.is_dirty()
        or bool(repo.untracked_files)
        or repo.head.is_detached
        or not is_synced(repo)
    )
