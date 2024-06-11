import re
import sys
import time
from pathlib import Path
from typing import Any

import git
import yaml
from ray.job_submission import JobStatus, JobSubmissionClient

from sitstart.logging import get_logger
from sitstart.scm.git.util import DOTFILES_REPO_PATH, list_tracked_dotfiles
from sitstart.util.run import run

DASHBOARD_PORT = 8265

logger = get_logger(__name__)


def get_job_submission_client() -> JobSubmissionClient:
    return JobSubmissionClient(f"http://127.0.0.1:{DASHBOARD_PORT}")


def wait_for_job_status(
    client: JobSubmissionClient,
    sub_id: str,
    target_status: JobStatus,
    timeout_sec=60,
) -> JobStatus:
    start_time_sec = time.time()
    while (
        client.get_job_status(sub_id) != target_status
        and time.time() - start_time_sec < timeout_sec
    ):
        time.sleep(1)
    return client.get_job_status(sub_id)


def get_file_mounts(config_path: Path, user_root: str = "/home") -> dict[Path, Path]:
    config = yaml.load(config_path.read_text(), Loader=yaml.SafeLoader)
    user = config["auth"]["ssh_user"]
    mounts = config["file_mounts"]

    def expand_user(path: str, user: str) -> Path:
        return Path(re.sub(r"^~/", f"{user_root}/{user}/", path))

    def resolve_path(path: str) -> Path:
        return Path(path).expanduser().resolve()

    return {expand_user(dst, user): resolve_path(src) for dst, src in mounts.items()}


def stop_all_jobs() -> None:
    client = get_job_submission_client()
    for job in client.list_jobs():
        if job.status == "RUNNING":
            if not job.submission_id:
                logger.warning(
                    f"Running {job.job_id} has no submission ID and cannot be stopped"
                )
                continue
            logger.info(f"Stopping job {job.job_id} / {job.submission_id}")
            client.stop_job(job.submission_id)


def submit_job(
    script_path: Path,
    config_path: Path,
    cluster_name: str,
    job_config_path: Path | None = None,
    restart: bool = False,
    do_sync_dotfiles: bool = False,
) -> str:
    # get the script path's containing repo
    try:
        repo = git.Repo(script_path, search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        logger.error(f"No git repository found for {str(script_path)!r}; exiting")
        sys.exit(-1)
    repo_path = Path(repo.working_dir)

    # make sure the job config is also in the repo
    if job_config_path and not job_config_path.is_relative_to(repo.working_dir):
        logger.error(f"Job config {str(job_config_path)!r} not in repository; exiting")
        sys.exit(-1)

    # ensure the repo is in the cluster config's `file_mounts`, which
    # map cluster paths to local paths for syncing local -> head ->
    # worker
    mounts = get_file_mounts(config_path)
    mount = next((m for m in mounts.items() if repo_path.is_relative_to(m[1])), None)
    if not mount:
        msg = f"Repo {repo.working_dir!r} not in file_mounts and cannot be synced."
        logger.error(msg)
        sys.exit(-1)

    # setup remote command, w/ hydra config args if job_config was provided
    cluster_script_path = Path(mount[0]) / script_path.relative_to(mount[1])
    cmd = ["python", str(cluster_script_path)]
    if job_config_path:
        job_config_path = Path(mount[0]) / job_config_path.relative_to(mount[1])
        cmd.append(f"--config-path={str(job_config_path.parent)}")
        cmd.append(f"--config-name={job_config_path.stem}")

    # invoke ray-up, syncing file mounts and running setup commands
    # even if the config hasn't changed
    logger.info("Running 'ray up' to sync files and run setup commands")
    cluster_up(
        config_path=config_path,
        cluster_name=cluster_name or Path(config_path).stem,
        no_restart=not restart,
        no_config_cache=True,
        do_sync_dotfiles=do_sync_dotfiles,
    )

    # submit the job; note that disallowing user-specified parameters,
    # aside from an optional job config that's part of the repository,
    # goes a long way to ensuring reproducibility from only the cached
    # repository state
    # TODO: control env vars here as well w/ ray envs
    client = get_job_submission_client()
    entrypoint = " ".join(cmd)
    logger.info(f"Submitting job with entrypoint {entrypoint!r}")
    sub_id = client.submit_job(entrypoint=entrypoint)
    logger.info(f"Job {sub_id} submitted")
    logger.info("See logs at http://localhost:3000/d/ray_logs_dashboard")
    logger.info("See Ray dashboard at http://localhost:8265")

    return sub_id


def sync_dotfiles(config_path: Path, cluster_name: str) -> None:
    files = list_tracked_dotfiles()
    if files:
        logger.info(f"[{cluster_name}] Syncing dotfiles to the cluster head")
        config = yaml.load(config_path.read_text(), Loader=yaml.SafeLoader)
        ssh_user = config["auth"]["ssh_user"]

        # we'll also sync the dotfiles .git directory separately,
        # since it's not in the repo root directory
        home = str(Path.home())
        repo_path = str(Path(DOTFILES_REPO_PATH).expanduser().relative_to(home))

        # get mappings from source to destination
        src_to_dst = {f"{home}/{f}": f"~{ssh_user}/{f}" for f in files}
        src_to_dst[f"{home}/{repo_path}/"] = f"~{ssh_user}/{repo_path}"
        dst_dirs = list(set(str(Path(f).parent) for f in src_to_dst.values()))
        cluster_args = [str(config_path), "--cluster-name", cluster_name]
        kwargs: dict[str, Any] = {"output": "capture"}

        # create directories at the destination and sync files; use `ray exec`
        # here and below since we may not have updated the SSH config
        cmd = f"mkdir -p {' '.join(dst_dirs)}"
        run(["ray", "exec", *cluster_args, "--verbose", cmd], **kwargs)
        for src, dest in src_to_dst.items():
            run(["ray", "rsync-up", *cluster_args, src, dest], **kwargs)

        # update the git/yadm config to use the ssh user's home directory
        conf_path = f"~{ssh_user}/{repo_path}/config"
        cmd = f"git config --file {conf_path} core.worktree /home/{ssh_user}"
        run(["ray", "exec", *cluster_args, "--verbose", cmd], **kwargs)


def cluster_up(
    config_path: Path,
    cluster_name: str,
    min_workers: int | None = None,
    max_workers: int | None = None,
    no_restart: bool = False,
    restart_only: bool = False,
    prompt: bool = False,
    verbose: bool = False,
    show_output: bool = False,
    no_config_cache: bool = False,
    do_sync_dotfiles: bool = False,
) -> None:
    cmd = [
        "ray",
        "up",
        str(config_path),
        "--verbose",
        "--disable-usage-stats",
        "--cluster-name",
        cluster_name,
    ]
    if min_workers is not None:
        cmd += ["--min-workers", str(min_workers)]
    if max_workers is not None:
        cmd += ["--max-workers", str(max_workers)]
    if no_restart:
        cmd.append("--no-restart")
    if restart_only:
        cmd.append("--restart-only")
    if not prompt:
        cmd.append("--yes")
    if verbose:
        cmd.append("--verbose")
    if no_config_cache:
        cmd.append("--no-config-cache")

    logger.info(f"[{cluster_name}] Creating or updating Ray cluster")
    if show_output:
        run(cmd, output="std")
    else:
        run(cmd + ["--log-color", "false"], output="file")

    if do_sync_dotfiles:
        sync_dotfiles(config_path, cluster_name)


def cluster_down(
    config_path: Path,
    cluster_name: str,
    workers_only: bool = False,
    keep_min_workers: bool = False,
    prompt: bool = False,
    verbose: bool = False,
    show_output: bool = False,
) -> None:
    pass
    cmd = ["ray", "down", str(config_path), "--cluster-name", cluster_name]
    if workers_only:
        cmd.append("--workers-only")
    if keep_min_workers:
        cmd.append("--keep-min-workers")
    if not prompt:
        cmd.append("--yes")
    if verbose:
        cmd.append("--verbose")

    logger.info(f"[{cluster_name}] Tearing down Ray cluster")

    if show_output:
        run(cmd, output="std")
    else:
        run(cmd + ["--log-color", "false"], output="file")
