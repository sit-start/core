import os
import subprocess
import sys
from os.path import expanduser, expandvars, realpath
from pathlib import Path
from time import sleep

import git
import ray
import typer
import yaml
from ray.job_submission import JobSubmissionClient
from typer import Argument, Option

from ktd.aws.ec2.util import (
    kill_instances_with_name,
    update_ssh_config_for_instances_with_name,
)
from ktd.aws.util import get_aws_session
from ktd.logging import get_logger
from ktd.util.run import run
from ktd.util.ssh import close_ssh_connection, open_ssh_tunnel
from ktd.util.vscode import open_vscode_over_ssh

DEFAULT_CONFIG = "main"
CONFIG_ROOT = f"{os.environ['DEV']}/core/python/ktd/ray/config/cluster"
FORWARDED_PORTS = {
    "Ray Dashboard": 8265,
    "Prometheus": 9090,
    "Grafana": 3000,
    "TensorBoard": 6006,
}
SCRIPT_PATH_DEFAULT = "$DEV/core/python/ktd/ml/experiments/{script_name}.py"


app = typer.Typer()
logger = get_logger(__name__, format="simple")

# Arguments and options
_config_arg = Argument(
    help=f"The Ray cluster config in '{CONFIG_ROOT}'.",
    default=DEFAULT_CONFIG,
)
_profile_opt = Option(
    None,
    help="The AWS profile to use.",
    envvar="AWS_PROFILE",
    show_default=False,
)
_min_workers_opt = Option(
    None,
    help="The minimum number of workers. This overrides the config.",
    show_default=False,
)
_max_workers_opt = Option(
    None,
    help="The maximum number of workers. This overrides the config.",
    show_default=False,
)
_no_restart_opt = Option(
    False,
    "--no-restart",
    help="Do not restart Ray services during the update.",
    show_default=False,
)
_restart_only_opt = Option(
    False,
    "--restart-only",
    help="Skip running setup commands and only restart Ray.",
    show_default=False,
)
_cluster_name_opt = Option(
    None,
    help="The cluster name. This overrides the config.",
    show_default=False,
)
_prompt_opt = Option(
    False,
    "--prompt",
    help="Prompt for confirmation.",
    show_default=False,
)
_verbose_opt = Option(
    False,
    "--verbose",
    help="Display verbose output.",
    show_default=False,
)
_open_vscode_opt = Option(
    False,
    "--open-vscode",
    help="Open VS Code on the cluster head.",
    show_default=False,
)
_show_output_opt = Option(
    False,
    "--show-output",
    help="Display output from the 'ray' command.",
    show_default=False,
)
_no_config_cache_opt = Option(
    False,
    "--no-config-cache",
    help="Disable the local cluster config cache.",
    show_default=False,
)
_workers_only_opt = Option(
    False,
    "--workers-only",
    help="Only destroy workers.",
    show_default=False,
)
_keep_min_workers_opt = Option(
    False,
    "--keep-min-workers",
    help="Retain the minimum number of workers specified in the config.",
    show_default=False,
)
_kill_opt = Option(
    False,
    "--kill",
    help="Terminate all instances.",
    show_default=False,
)
_script_path_arg = Argument(
    SCRIPT_PATH_DEFAULT,
    help="The path to the script to run, required if --script-name is not specified.",
)
_script_name_opt = Option(
    None,
    help="The name of the script to run, required if script_path is not specified.",
    show_default=False,
)
_restart_opt = Option(
    False,
    "--restart",
    help="Restart Ray services. This stops any existing jobs.",
    show_default=False,
)


def _ray_up(
    config: str = "main",
    min_workers: int | None = None,
    max_workers: int | None = None,
    no_restart: bool = False,
    restart_only: bool = False,
    cluster_name: str = "",
    prompt: bool = False,
    verbose: bool = False,
    show_output: bool = False,
    no_config_cache: bool = False,
) -> None:
    cmd = ["ray", "up", f"{CONFIG_ROOT}/{config}.yaml"]
    if min_workers is not None:
        cmd += ["--min-workers", str(min_workers)]
    if max_workers is not None:
        cmd += ["--max-workers", str(max_workers)]
    if no_restart:
        cmd.append("--no-restart")
    if restart_only:
        cmd.append("--restart-only")
    if cluster_name:
        cmd += ["--cluster-name", cluster_name]
    else:
        cluster_name = config
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


@app.command()
def stop_jobs() -> None:
    """Stop all running jobs on the active Ray cluster."""
    client = ray.job_submission.JobSubmissionClient("http://127.0.0.1:8265")
    for job in client.list_jobs():
        if job.status == "RUNNING":
            if not job.submission_id:
                logger.warning(
                    f"Running {job.job_id} has no submission ID and cannot be stopped"
                )
                continue
            logger.info(f"Stopping job {job.job_id} / {job.submission_id}")
            client.stop_job(job.submission_id)


# TODO: control env vars here as well; use ray envs
# TODO: support additional user-specified repos on local+remote hosts
@app.command()
def submit(
    script_path: str = _script_path_arg,
    script_name: str = _script_name_opt,
    config: str = _config_arg,
    cluster_name: str = _cluster_name_opt,
    restart: bool = _restart_opt,
) -> None:
    """Run a job on a Ray cluster."""
    # get script name
    if not script_name and script_path == SCRIPT_PATH_DEFAULT:
        raise RuntimeError("exp_name or exp_path must be provided")
    script_path = script_path.format(script_name=script_name)

    # get repo
    try:
        repo = git.Repo(expandvars(script_path), search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        logger.error(f"No git repository found for {script_path!r}; exiting")
        sys.exit(-1)

    # ensure the repo is in cluster config's file mounts, as we'll use that
    # to sync repo state with head and worker nodes
    config_path = Path(CONFIG_ROOT) / f"{config}.yaml"
    config_data = yaml.load(config_path.read_text(), Loader=yaml.FullLoader)
    file_mounts = [realpath(expanduser(f)) for f in config_data["file_mounts"].values()]
    repo_path = Path(repo.working_dir).resolve()
    if not any(repo_path.is_relative_to(f) for f in file_mounts):
        msg = f"Repo {repo.working_dir!r} not in file_mounts and cannot be synced."
        raise RuntimeError(msg)

    # invoke ray-up, syncing file mounts and running setup commands
    # even if the config hasn't changed
    logger.info("Running 'ray up' to sync files and run setup commands")
    _ray_up(
        config=config,
        no_restart=not restart,
        cluster_name=cluster_name,
        no_config_cache=True,
    )

    # run a basic job that uses the native environment and existing file(s)
    client = JobSubmissionClient("http://127.0.0.1:8265")
    entrypoint = f"python {script_path}"
    try:
        logger.info(f"Submitting job with entrypoint {entrypoint!r}")
        sub_id = client.submit_job(entrypoint=entrypoint)
        logger.info(f"Job {sub_id} submitted")
        logger.info("See logs at http://localhost:3000/d/ray_logs_dashboard")
        logger.info("See Ray dashboard at http://localhost:8265")
    except RuntimeError as e:
        logger.info(f"Failed to submit job: {e}")
        sys.exit(-1)


@app.command()
def up(
    config: str = _config_arg,
    profile: str = _profile_opt,
    min_workers: int = _min_workers_opt,
    max_workers: int = _max_workers_opt,
    no_restart: bool = _no_restart_opt,
    restart_only: bool = _restart_only_opt,
    cluster_name: str = _cluster_name_opt,
    prompt: bool = _prompt_opt,
    verbose: bool = _verbose_opt,
    open_vscode: bool = _open_vscode_opt,
    show_output: bool = _show_output_opt,
    no_config_cache: bool = _no_config_cache_opt,
) -> None:
    """Create or update a Ray cluster."""
    # invoke ray up
    _ray_up(
        config=config,
        min_workers=min_workers,
        max_workers=max_workers,
        no_restart=no_restart,
        restart_only=restart_only,
        cluster_name=cluster_name,
        prompt=prompt,
        verbose=verbose,
        show_output=show_output,
        no_config_cache=no_config_cache,
    )

    # 5s is usually sufficient for the minimal workers to be in the
    # running state after the Ray cluster is up
    sleep(5)
    cluster_name = cluster_name or config
    session = get_aws_session(profile)
    update_ssh_config_for_instances_with_name(
        session, instance_name=f"ray-{cluster_name}-*"
    )

    cluster_head_name = f"ray-{cluster_name}-head"
    logger.info(
        f"[{cluster_head_name}] Forwarding ports for "
        f"{', '.join(FORWARDED_PORTS.keys())}"
    )

    close_ssh_connection(cluster_head_name)
    for port in FORWARDED_PORTS.values():
        open_ssh_tunnel(cluster_head_name, port)

    if open_vscode:
        open_vscode_over_ssh(cluster_head_name)


@app.command()
def down(
    cluster_name: str = _cluster_name_opt,
    profile: str = _profile_opt,
    workers_only: bool = _workers_only_opt,
    keep_min_workers: bool = _keep_min_workers_opt,
    prompt: bool = _prompt_opt,
    verbose: bool = _verbose_opt,
    kill: bool = _kill_opt,
    show_output: bool = _show_output_opt,
) -> None:
    """Tear down a Ray cluster."""
    cluster_name = cluster_name or DEFAULT_CONFIG

    cluster_head_name = f"ray-{cluster_name}-head"
    cluster_names = f"ray-{cluster_name}-*"
    worker_names = f"ray-{cluster_name}-workers"

    logger.info(f"Closing SSH connections to {cluster_head_name}")
    try:
        close_ssh_connection(cluster_head_name, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to close SSH connections: {e}")

    logger.info(f"[{cluster_name}] Tearing down Ray cluster")
    cmd = ["ray", "down", f"{CONFIG_ROOT}/{cluster_name}.yaml"]
    if workers_only:
        cmd.append("--workers-only")
    if keep_min_workers:
        cmd.append("--keep-min-workers")
    if not prompt:
        cmd.append("--yes")
    if verbose:
        cmd.append("--verbose")

    if show_output:
        run(cmd, output="std")
    else:
        run(cmd + ["--log-color", "false"], output="file")

    session = get_aws_session(profile)

    if kill and keep_min_workers:
        logger.warning("Killing instances with keep_min_workers is not implemented")
    elif kill:
        instance_name = worker_names if workers_only else cluster_names
        logger.info(f"[{instance_name}] Killing instances")
        kill_instances_with_name(session, instance_name, update_ssh_config=False)

    update_ssh_config_for_instances_with_name(session, instance_name=cluster_names)


@app.command()
def monitor(
    config: str = Argument(
        help=f"The Ray cluster config in '{CONFIG_ROOT}'.", default=DEFAULT_CONFIG
    ),
) -> None:
    """Monitor autoscaling on a Ray cluster."""
    log_path = "/tmp/ray/session_latest/logs/monitor*"
    cmd = [
        "ray",
        "exec",
        f"{CONFIG_ROOT}/{config}.yaml",
        "--",
        f"tail -n 100 -f {log_path}",
    ]
    run(cmd, check=False)
