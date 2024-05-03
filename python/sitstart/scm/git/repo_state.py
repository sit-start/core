from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from git import Repo
from git.remote import Remote

from sitstart.logging import get_logger
from sitstart.scm.git.util import diff_vs_commit, get_first_remote_ancestor, get_repo

logger = get_logger(__name__)


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
                patch_path.write_text(f"{self.uncommitted_changes}\n")
                repo.git.apply(str(patch_path))
