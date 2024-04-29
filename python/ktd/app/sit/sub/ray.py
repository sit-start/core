import subprocess
import sys
from os.path import expanduser, realpath
from pathlib import Path
from time import sleep

import git
import ray
import typer
import yaml
from ray.job_submission import JobSubmissionClient
from typer import Argument, Option
from typing_extensions import Annotated, Optional

from ktd import PYTHON_ROOT
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
CONFIG_ROOT = f"{PYTHON_ROOT}/ktd/ray/config/cluster"
FORWARDED_PORTS = {
    "Ray Dashboard": 8265,
    "Prometheus": 9090,
    "Grafana": 3000,
    "TensorBoard": 6006,
}
SCRIPT_PATH_DEFAULT = f"{PYTHON_ROOT}/ktd/ml/experiments/{{script_name}}.py"


app = typer.Typer()
logger = get_logger(__name__, format="simple")

# Arguments and options
ConfigArg = Annotated[
    str,
    Argument(help=f"The Ray cluster config in '{CONFIG_ROOT}'."),
]
ProfileOpt = Annotated[
    Optional[str],
    Option(help="The AWS profile to use.", envvar="AWS_PROFILE", show_default=False),
]
MinWorkersOpt = Annotated[
    Optional[int],
    Option(
        help="The minimum number of workers. This overrides the config.",
        show_default=False,
    ),
]
MaxWorkersOpt = Annotated[
    Optional[int],
    Option(
        help="The maximum number of workers. This overrides the config.",
        show_default=False,
    ),
]
NoRestartOpt = Annotated[
    bool,
    Option(
        "--no-restart",
        help="Do not restart Ray services during the update.",
        show_default=False,
    ),
]
RestartOnlyOpt = Annotated[
    bool,
    Option(
        "--restart-only",
        help="Skip running setup commands and only restart Ray.",
        show_default=False,
    ),
]
ClusterNameOpt = Annotated[
    Optional[str],
    Option(help="The cluster name. This overrides the config.", show_default=False),
]
PromptOpt = Annotated[
    bool,
    Option("--prompt", help="Prompt for confirmation.", show_default=False),
]
VerboseOpt = Annotated[
    bool,
    Option("--verbose", help="Display verbose output.", show_default=False),
]
OpenVscodeOpt = Annotated[
    bool,
    Option(
        "--open-vscode", help="Open VS Code on the cluster head.", show_default=False
    ),
]
ShowOutputOpt = Annotated[
    bool,
    Option(
        "--show-output",
        help="Display output from the 'ray' command.",
        show_default=False,
    ),
]
NoConfigCacheOpt = Annotated[
    bool,
    Option(
        "--no-config-cache",
        help="Disable the local cluster config cache.",
        show_default=False,
    ),
]
WorkersOnlyOpt = Annotated[
    bool,
    Option("--workers-only", help="Only destroy workers.", show_default=False),
]
KeepMinWorkersOpt = Annotated[
    bool,
    Option(
        "--keep-min-workers",
        help="Retain the minimum number of workers specified in the config.",
        show_default=False,
    ),
]
KillOpt = Annotated[
    bool,
    Option("--kill", help="Terminate all instances.", show_default=False),
]
ScriptPathArg = Annotated[
    str,
    Argument(
        help="The path to the script to run, required if --script-name is not specified."
    ),
]
ScriptNameOpt = Annotated[
    Optional[str],
    Option(
        help="The name of the script to run, required if script_path is not specified.",
        show_default=False,
    ),
]
RestartOpt = Annotated[
    bool,
    Option(
        "--restart",
        help="Restart Ray services. This stops any existing jobs.",
        show_default=False,
    ),
]


def _ray_up(
    config: str = "main",
    min_workers: int | None = None,
    max_workers: int | None = None,
    no_restart: bool = False,
    restart_only: bool = False,
    cluster_name: str | None = None,
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
    script_path: ScriptPathArg = SCRIPT_PATH_DEFAULT,
    script_name: ScriptNameOpt = None,
    config: ConfigArg = DEFAULT_CONFIG,
    cluster_name: ClusterNameOpt = None,
    restart: RestartOpt = False,
) -> None:
    """Run a job on a Ray cluster."""
    # get script name
    if not script_name and script_path == SCRIPT_PATH_DEFAULT:
        raise RuntimeError("exp_name or exp_path must be provided")

    def _resolve(path: str) -> str:
        return realpath(expanduser(path))

    script_path = _resolve(script_path.format(script_name=script_name))

    # get the script path's containing repo
    try:
        repo = git.Repo(script_path, search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        logger.error(f"No git repository found for {script_path!r}; exiting")
        sys.exit(-1)

    # ensure the repo is in the cluster config's `file_mounts`, which maps
    # cluster paths to local paths for syncing local -> head -> worker
    config_path = Path(CONFIG_ROOT) / f"{config}.yaml"
    config_data = yaml.load(config_path.read_text(), Loader=yaml.FullLoader)
    repo_path = Path(repo.working_dir).resolve()
    mounts = {dst: _resolve(src) for dst, src in config_data["file_mounts"].items()}
    mount = next((m for m in mounts.items() if repo_path.is_relative_to(m[1])), None)
    if not mount:
        msg = f"Repo {repo.working_dir!r} not in file_mounts and cannot be synced."
        raise RuntimeError(msg)
    cluster_script_path = f"{mount[0]}/{str(Path(script_path).relative_to(mount[1]))}"

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
    entrypoint = f"python {cluster_script_path}"
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
    config: ConfigArg = DEFAULT_CONFIG,
    profile: ProfileOpt = None,
    min_workers: MinWorkersOpt = None,
    max_workers: MaxWorkersOpt = None,
    no_restart: NoRestartOpt = False,
    restart_only: RestartOnlyOpt = False,
    cluster_name: ClusterNameOpt = None,
    prompt: PromptOpt = False,
    verbose: VerboseOpt = False,
    open_vscode: OpenVscodeOpt = False,
    show_output: ShowOutputOpt = False,
    no_config_cache: NoConfigCacheOpt = False,
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
    cluster_name: ClusterNameOpt = None,
    profile: ProfileOpt = None,
    workers_only: WorkersOnlyOpt = False,
    keep_min_workers: KeepMinWorkersOpt = False,
    prompt: PromptOpt = False,
    verbose: VerboseOpt = False,
    kill: KillOpt = False,
    show_output: ShowOutputOpt = False,
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
