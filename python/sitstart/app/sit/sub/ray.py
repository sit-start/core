import re
import sys
from pathlib import Path
from time import sleep
from typing import Any

import git
import ray
import typer
import yaml
from ray.job_submission import JobSubmissionClient
from typer import Argument, Option
from typing_extensions import Annotated, Optional

from sitstart import PYTHON_ROOT
from sitstart.aws.ec2.util import (
    kill_instances_with_name,
    update_ssh_config_for_instances_with_name,
)
from sitstart.aws.util import get_aws_session
from sitstart.logging import get_logger
from sitstart.scm.git.util import DOTFILES_REPO_PATH, list_tracked_dotfiles
from sitstart.util.run import run
from sitstart.util.ssh import close_ssh_connection, open_ssh_tunnel
from sitstart.util.vscode import open_vscode_over_ssh

DEFAULT_CONFIG = "main"
CLUSTER_CONFIG_ROOT = f"{PYTHON_ROOT}/sitstart/ray/config/cluster"
SCRIPT_ROOT = f"{PYTHON_ROOT}/sitstart/ml/experiments"
JOB_CONFIG_DIR = "conf"
DASHBOARD_PORT = 8265
FORWARDED_PORTS = {
    "Ray Dashboard": DASHBOARD_PORT,
    "Prometheus": 9090,
    "Grafana": 3000,
    "TensorBoard": 6006,
}


app = typer.Typer()
logger = get_logger(__name__, format="simple")


# Arguments and options
ConfigOpt = Annotated[
    str,
    Option(
        help="The Ray cluster config path, or filename or stem in "
        f"{CLUSTER_CONFIG_ROOT!r}."
    ),
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
NoPortForwardingOpt = Annotated[
    bool,
    Option(
        "--no-port-forwarding",
        help=f"Disable port forwarding for {' '.join(FORWARDED_PORTS.keys())}.",
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
ScriptArg = Annotated[
    str,
    Argument(
        help=f"The script path, or filename or stem in {SCRIPT_ROOT!r}.",
        show_default=False,
    ),
]
JobConfigOpt = Annotated[
    Optional[str],
    Option(
        help="The job config path, or filename or stem in the "
        f"{JOB_CONFIG_DIR!r} directory adjacent to the script.",
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
NoSyncDotfilesOpt = Annotated[
    bool,
    Option(
        "--no-sync-dotfiles",
        help="Disable syncing dotfiles to the head node.",
        show_default=False,
    ),
]


def _job_submission_client() -> JobSubmissionClient:
    return ray.job_submission.JobSubmissionClient(f"http://127.0.0.1:{DASHBOARD_PORT}")


def _resolve_path(
    input: str, root: str | None = None, extensions: list[str] = []
) -> Path:
    candidates = [input]
    if root is not None:
        candidates += [f"{root}/{input}"]
    if extensions:
        candidates += [f"{root}/{input}.{ext}" for ext in extensions]
    for candidate in candidates:
        path = Path(candidate).expanduser().resolve()
        if path.exists():
            return path
    raise FileNotFoundError(
        f"File/stem {input!r} not found in {root!r} with extensions "
        f"{' '.join(f'{e!r}' for e in extensions)}."
    )


def _resolve_cluster_config_path(config: str) -> Path:
    return _resolve_path(config, CLUSTER_CONFIG_ROOT, ["yaml", "yml"])


def _expand_user(path: str, user: str, user_root: str = "/home") -> Path:
    return Path(re.sub(r"^~/", f"{user_root}/{user}/", path))


def _resolve_file_mounts(config_path: Path) -> dict[Path, Path]:
    config = yaml.load(config_path.read_text(), Loader=yaml.SafeLoader)
    user = config["auth"]["ssh_user"]
    mounts = config["file_mounts"]

    return {_expand_user(dst, user): _resolve_path(src) for dst, src in mounts.items()}


def _ray_up(
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
    sync_dotfiles: bool = False,
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

    if sync_dotfiles:
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


@app.command()
def stop_jobs() -> None:
    """Stop all running jobs on the active Ray cluster."""
    client = _job_submission_client()
    for job in client.list_jobs():
        if job.status == "RUNNING":
            if not job.submission_id:
                logger.warning(
                    f"Running {job.job_id} has no submission ID and cannot be stopped"
                )
                continue
            logger.info(f"Stopping job {job.job_id} / {job.submission_id}")
            client.stop_job(job.submission_id)


@app.command()
def submit(
    script: ScriptArg,
    config: ConfigOpt = DEFAULT_CONFIG,
    cluster_name: ClusterNameOpt = None,
    profile: ProfileOpt = None,
    job_config: JobConfigOpt = None,
    restart: RestartOpt = False,
    no_sync_dotfiles: NoSyncDotfilesOpt = False,
) -> str:
    """Run a job on a Ray cluster."""
    _ = get_aws_session(profile=profile)
    # resolve input paths
    config_path = _resolve_cluster_config_path(config)
    script_path = _resolve_path(script, SCRIPT_ROOT, ["py", "sh"])
    job_config_root = str(script_path.parent / JOB_CONFIG_DIR)
    job_config_path = (
        _resolve_path(job_config, job_config_root, ["yaml", "yml"])
        if job_config
        else None
    )

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
    mounts = _resolve_file_mounts(config_path)
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
    _ray_up(
        config_path=config_path,
        cluster_name=cluster_name or Path(config_path).stem,
        no_restart=not restart,
        no_config_cache=True,
        sync_dotfiles=not no_sync_dotfiles,
    )

    # submit the job; note that disallowing user-specified parameters,
    # aside from an optional job config that's part of the repository,
    # goes a long way to ensuring reproducibility from only the cached
    # repository state
    # TODO: control env vars here as well w/ ray envs
    client = JobSubmissionClient("http://127.0.0.1:8265")
    entrypoint = " ".join(cmd)
    try:
        logger.info(f"Submitting job with entrypoint {entrypoint!r}")
        sub_id = client.submit_job(entrypoint=entrypoint)
        logger.info(f"Job {sub_id} submitted")
        logger.info("See logs at http://localhost:3000/d/ray_logs_dashboard")
        logger.info("See Ray dashboard at http://localhost:8265")
    except RuntimeError as e:
        logger.info(f"Failed to submit job: {e}")
        sys.exit(-1)

    return sub_id


@app.command()
def up(
    config: ConfigOpt = DEFAULT_CONFIG,
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
    no_sync_dotfiles: NoSyncDotfilesOpt = False,
    no_port_forwarding: NoPortForwardingOpt = False,
) -> None:
    """Create or update a Ray cluster."""
    session = get_aws_session(profile)
    config_path = _resolve_cluster_config_path(config)
    cluster_name = cluster_name or config_path.stem

    # invoke ray up
    _ray_up(
        config_path=config_path,
        cluster_name=cluster_name,
        min_workers=min_workers,
        max_workers=max_workers,
        no_restart=no_restart,
        restart_only=restart_only,
        prompt=prompt,
        verbose=verbose,
        show_output=show_output,
        no_config_cache=no_config_cache,
        sync_dotfiles=not no_sync_dotfiles,
    )

    # 5s is usually sufficient for the minimal workers to be in the
    # running state after the Ray cluster is up
    sleep(5)
    update_ssh_config_for_instances_with_name(
        session, instance_name=f"ray-{cluster_name}-*"
    )

    cluster_head_name = f"ray-{cluster_name}-head"
    close_ssh_connection(cluster_head_name)
    if not no_port_forwarding:
        logger.info(
            f"[{cluster_head_name}] Forwarding ports for "
            f"{', '.join(FORWARDED_PORTS.keys())}"
        )
        for port in FORWARDED_PORTS.values():
            open_ssh_tunnel(cluster_head_name, port)

    if open_vscode:
        open_vscode_over_ssh(cluster_head_name)


@app.command()
def down(
    config: ConfigOpt = DEFAULT_CONFIG,
    profile: ProfileOpt = None,
    cluster_name: ClusterNameOpt = None,
    workers_only: WorkersOnlyOpt = False,
    keep_min_workers: KeepMinWorkersOpt = False,
    prompt: PromptOpt = False,
    verbose: VerboseOpt = False,
    kill: KillOpt = False,
    show_output: ShowOutputOpt = False,
) -> None:
    """Tear down a Ray cluster."""
    session = get_aws_session(profile)
    config_path = _resolve_cluster_config_path(config)
    cluster_name = cluster_name or config_path.stem
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

    instance_names = (
        f"ray-{cluster_name}-worker" if workers_only else f"ray-{cluster_name}-*"
    )

    if kill and keep_min_workers:
        logger.warning("Killing instances with keep_min_workers is not implemented")
    elif kill:
        logger.info(f"[{instance_names}] Killing instances")
        kill_instances_with_name(session, instance_names, update_ssh_config=False)

    update_ssh_config_for_instances_with_name(session, instance_name=instance_names)


@app.command()
def monitor(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Monitor autoscaling on a Ray cluster."""
    config_path = str(_resolve_cluster_config_path(config))
    log_path = "/tmp/ray/session_latest/logs/monitor*"
    cmd = [
        "ray",
        "exec",
        config_path,
        "--",
        f"tail -n 100 -f {log_path}",
    ]
    run(cmd, check=False)
