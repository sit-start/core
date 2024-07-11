import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from ray.cluster_utils import ray_constants
from ray.job_submission import JobDetails, JobStatus, JobSubmissionClient

from sitstart import PYTHON_ROOT, REPO_ROOT
from sitstart.logging import get_logger
from sitstart.ml.experiments import CONFIG_ROOT
from sitstart.ml.experiments.util import (
    get_experiment_wandb_url,
    get_param_space_description,
    load_experiment_config,
)
from sitstart.scm.git.util import DOTFILES_REPO_PATH, list_tracked_dotfiles
from sitstart.util.run import run

DASHBOARD_PORT = 8265

WORKING_DIR = REPO_ROOT
REMOTE_WORKING_DIR = Path(
    f"'${{{ray_constants.RAY_RUNTIME_ENV_CREATE_WORKING_DIR_ENV_VAR}}}'"
)

logger = get_logger(__name__)


def get_job_submission_client(
    dashboard_port: int = DASHBOARD_PORT,
) -> JobSubmissionClient:
    return JobSubmissionClient(f"http://127.0.0.1:{dashboard_port}")


def get_job_runtime_env(clone_venv: bool = True) -> dict[str, Any]:
    return {
        "working_dir": WORKING_DIR,
        "pip": "requirements.txt" if clone_venv else None,
        "env_vars": {"PYTHONPATH": str(Path(PYTHON_ROOT).relative_to(WORKING_DIR))},
    }


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


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _stop_job(
    job: JobDetails, delete: bool = False, dashboard_port: int = DASHBOARD_PORT
) -> None:
    logger.info(f"Stopping job {job.submission_id} ({job.job_id})")
    if not job.submission_id:
        logger.warning(
            f"Running {job.job_id} has no submission ID and cannot be stopped"
        )
        return

    client = get_job_submission_client(dashboard_port)
    try:
        if job.status == JobStatus.RUNNING:
            client.stop_job(job.submission_id)
            if delete:
                wait_for_job_status(
                    client, job.submission_id, JobStatus.STOPPED, timeout_sec=10
                )
        if delete:
            client.delete_job(job.submission_id)
    except Exception as e:
        logger.error(f"Error stopping job {job.submission_id}: {e}")
        return


def get_file_mounts(config_path: Path, user_root: str = "/home") -> dict[Path, Path]:
    config = yaml.load(config_path.read_text(), Loader=yaml.SafeLoader)
    user = config["auth"]["ssh_user"]
    mounts = config["file_mounts"]

    def expand_user(path: str, user: str) -> Path:
        return Path(re.sub(r"^~/", f"{user_root}/{user}/", path))

    return {expand_user(dst, user): _resolve_path(src) for dst, src in mounts.items()}


def stop_job(
    sub_id: str, delete: bool = False, dashboard_port: int = DASHBOARD_PORT
) -> None:
    if sub_id == "all":
        stop_all_jobs(delete=delete, dashboard_port=dashboard_port)
        return

    client = get_job_submission_client(dashboard_port)
    try:
        job = client.get_job_info(sub_id)
    except Exception as e:
        logger.error(f"Error getting job info for {sub_id}: {e}")
    _stop_job(job, delete=delete, dashboard_port=dashboard_port)


def stop_all_jobs(delete: bool = False, dashboard_port: int = DASHBOARD_PORT) -> None:
    logger.info("Stopping all jobs.")
    client = get_job_submission_client(dashboard_port)
    for job in client.list_jobs():
        _stop_job(job, delete=delete, dashboard_port=dashboard_port)


def list_jobs(dashboard_port: int = DASHBOARD_PORT) -> None:
    client = get_job_submission_client(dashboard_port)
    start = "\n" + "-" * 50 + "\n"
    message = "Listing all jobs."
    for job in client.list_jobs():
        entrypoint = job.entrypoint.replace(" ", " \\\n    ")
        description = job.metadata.get("description", None) if job.metadata else None
        message += (
            f"{start}{job.job_id} / {job.submission_id}:"
            + (f"\n- description: {description}" if description else "")
            + (f"\n- status: {job.status}")
            + (f"\n- entrypoint: {entrypoint}")
        )
    logger.info(message)


def submit_job(
    script_path: Path,
    dashboard_port: int = DASHBOARD_PORT,
    description: str | None = None,
    clone_venv: bool = True,
    config_path: Path | None = None,
) -> str:
    """Submit a job to the Ray cluster.

    Args:
        script_path: The path to the script to run. Must be in `WORKING_DIR`.
        dashboard_port: The port for the Ray dashboard.
        description: A description of the job, shown in`list_jobs`.
        clone_venv: Whether to clone the current virtual environment.
        config_path: The path to the script's Hydra config.
    """
    script_path = _resolve_path(script_path)
    if not script_path.is_relative_to(WORKING_DIR):
        logger.error(f"Script path {script_path} not in working dir {WORKING_DIR}.")
        sys.exit(-1)
    script_path_in_working_dir = script_path.relative_to(WORKING_DIR)

    if config_path:
        config_path = _resolve_path(config_path)
        if not config_path.is_relative_to(WORKING_DIR):
            logger.error(f"Config path {config_path} not in working dir {WORKING_DIR}.")
            sys.exit(-1)
        config_path_in_working_dir = config_path.relative_to(script_path.parent)

    # setup the remote command
    cmd = ["python", str(script_path_in_working_dir)]
    if config_path:
        cmd.append(f"--config-path={config_path_in_working_dir.parent}")
        cmd.append(f"--config-name={config_path_in_working_dir.stem}")

    # grab extra info if the job config is an experiment config
    wandb_url = None
    if config_path and config_path == Path(CONFIG_ROOT) / config_path.name:
        exp_config = load_experiment_config(config_path.stem)
        param_space_desc = get_param_space_description(exp_config)
        prefix = f"{description}: " if description else ""
        description = prefix + param_space_desc
        wandb_url = get_experiment_wandb_url(exp_config)

    # submit the job
    # TODO: https://github.com/sit-start/core/issues/126
    client = get_job_submission_client(dashboard_port=dashboard_port)
    entrypoint = " ".join(cmd)
    metadata = {"description": description} if description else None
    runtime_env = get_job_runtime_env(clone_venv=clone_venv)
    logger.info(f"Submitting job with entrypoint {entrypoint!r}")
    sub_id = client.submit_job(
        entrypoint=entrypoint, metadata=metadata, runtime_env=runtime_env
    )

    logger.info(f"Job {sub_id} submitted")
    logger.info("Logs: http://localhost:3000/d/ray_logs_dashboard")
    logger.info(f"Ray dashboard: http://localhost:{dashboard_port}")
    if wandb_url:
        logger.info(f"W&B project: {wandb_url}")

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
