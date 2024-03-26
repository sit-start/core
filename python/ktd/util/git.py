import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from git import Commit, GitCommandError, Repo, TagReference
from git.remote import Remote
from ktd.logging import get_logger
from ktd.util.identifier import StringIdType
from ktd.util.run import run

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


def get_remote_branches_for_commit(repo: str | Repo, commit: Commit | str) -> list[str]:
    repo = repo if isinstance(repo, Repo) else get_repo(repo)
    commit = commit if isinstance(commit, Commit) else repo.commit(commit)
    try:
        return repo.git.branch("-r", "--contains", commit.hexsha).splitlines()
    except GitCommandError:
        return []


def is_commit_in_remote(repo: str | Repo, commit: Commit | str) -> bool:
    return bool(get_remote_branches_for_commit(repo, commit))


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


def is_pristine(repo: str | Repo) -> bool:
    repo = repo if isinstance(repo, Repo) else get_repo(repo)
    return not (
        repo.is_dirty()
        or bool(repo.untracked_files)
        or repo.head.is_detached
        or not is_synced(repo)
    )


@dataclass
class RepoState:
    path: str
    url: str
    remote: str
    branch: str
    commit: str
    remote_commit: str
    uncommitted_changes: str
    local_commits: str

    @classmethod
    def from_repo(
        cls,
        repo: str | Repo,
        remote: str | Remote = "origin",
        branch: str = "main",
    ) -> "RepoState":
        repo = repo if isinstance(repo, Repo) else get_repo(repo)
        commit = repo.head.commit
        remote_commit = get_first_remote_ancestor(repo, commit, remote, branch)
        local_commits = repo.git.format_patch("--stdout", remote_commit)

        assert commit == remote_commit or bool(
            local_commits
        ), "No local commits, but commit != remote_commit"

        return cls(
            path=str(repo.working_dir),
            url=repo.remotes[str(remote)].url,
            remote=str(remote),
            branch=branch,
            commit=commit.hexsha,
            remote_commit=remote_commit.hexsha,
            uncommitted_changes=diff_vs_commit(repo, include_untracked=True),
            local_commits=local_commits,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "RepoState":
        return cls(**d)

    @property
    def summary(self) -> str:
        is_local = self.commit != self.remote_commit
        is_dirty_or_has_untracked_files = bool(self.uncommitted_changes)
        hash_len = 7
        return (
            is_local * f"{self.remote_commit[:hash_len]}.."
            + self.commit[:hash_len]
            + is_dirty_or_has_untracked_files * "+"
        )

    def replay(
        self, repo: str | Repo, replay_branch_name: str = "repo-state-replay"
    ) -> None:
        repo = repo if isinstance(repo, Repo) else get_repo(repo)
        replay_branch = repo.create_head(replay_branch_name, commit=self.remote_commit)
        replay_branch.checkout()

        if not (self.uncommitted_changes or self.local_commits):
            return

        with NamedTemporaryFile(prefix="/tmp/") as f:
            f.close()
            patch_path = Path(f.name)

            if self.local_commits:
                patch_path.write_text(self.local_commits)
                repo.git.am(
                    "--3way",
                    "--keep-non-patch",
                    "--committer-date-is-author-date",
                    str(patch_path),
                )

            if self.uncommitted_changes:
                patch_path.write_text(self.uncommitted_changes)
                repo.git.apply(str(patch_path))


def get_github_user():
    return json.loads(run(shlex.split("gh api user"), output="capture").stdout)["login"]


def get_github_ssh_url(account: str, repo: str):
    return f"git@github.com:{account}/{repo}"
