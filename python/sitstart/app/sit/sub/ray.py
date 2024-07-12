import sys
from pathlib import Path
from time import sleep

import typer
from typer import Argument, Option
from typing_extensions import Annotated, Optional

from sitstart import PYTHON_ROOT
from sitstart.aws.ec2.util import (
    kill_instances_with_name,
    update_ssh_config_for_instances_with_name,
)
from sitstart.aws.util import get_aws_session
from sitstart.logging import get_logger
from sitstart.ray.cluster import (
    DASHBOARD_PORT,
    cluster_down,
    cluster_up,
    list_jobs as _list_jobs,
    stop_job as _stop_job,
    submit_job,
)
from sitstart.util.run import run
from sitstart.util.ssh import close_ssh_connection, open_ssh_tunnel
from sitstart.util.vscode import open_vscode_over_ssh

DEFAULT_CONFIG = "main"
CLUSTER_CONFIG_ROOT = f"{PYTHON_ROOT}/sitstart/ray/config/cluster"
SCRIPT_ROOT = f"{PYTHON_ROOT}/sitstart/ml/experiments"
SCRIPT_CONFIG_DIR = "conf"
FORWARDED_PORTS = {  # (service, remote port) pairs
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
        help=f"Disable port forwarding for {', '.join(FORWARDED_PORTS.keys())}.",
        show_default=False,
    ),
]
ForwardPortOpt = Annotated[
    Optional[list[str]],
    Option(
        help=(
            "Forward a local port for one of "
            f"{', '.join(FORWARDED_PORTS.keys())}, "
            "e.g., 'Ray Dashboard,8266'. Overrides --no-port-forwarding. "
            "Use a negative port to disable."
        ),
        show_default=False,
    ),
]
DashboardPortOpt = Annotated[
    int,
    Option(help="The local port for the Ray dashboard used for submitting the job."),
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
ScriptConfigOpt = Annotated[
    Optional[str],
    Option(
        help="The script config path, or filename or stem in the "
        f"{SCRIPT_CONFIG_DIR!r} directory adjacent to the script.",
        show_default=False,
    ),
]
DescriptionOpt = Annotated[
    Optional[str],
    Option(help="A description for the job.", show_default=False),
]
SubmissionIdArg = Annotated[
    list[str],
    Argument(
        help="The submission ID of the job to stop or 'all' to stop all jobs.",
        show_default=False,
    ),
]
DeleteOpt = Annotated[
    bool,
    Option(
        "--delete",
        help="Delete the job after stopping it.",
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
SyncDotfilesOpt = Annotated[
    bool,
    Option(
        "--sync-dotfiles",
        help="Sync dotfiles to the head node.",
        show_default=False,
    ),
]
CloneVenvOpt = Annotated[
    bool,
    Option(
        "--clone-venv",
        help="Create a new virtual environment for the job "
        "from the requirements.txt",
        show_default=False,
    ),
]


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


@app.command()
def stop_job(
    submission_id: SubmissionIdArg,
    delete: DeleteOpt = False,
    dashboard_port: DashboardPortOpt = DASHBOARD_PORT,
) -> None:
    """Stops a job on the active Ray cluster."""
    for sub_id in submission_id:
        _stop_job(sub_id=sub_id, delete=delete, dashboard_port=dashboard_port)


@app.command()
def list_jobs(
    dashboard_port: DashboardPortOpt = DASHBOARD_PORT,
) -> None:
    """List all jobs on the active Ray cluster."""
    _list_jobs(dashboard_port=dashboard_port)


@app.command()
def submit(
    script: ScriptArg,
    profile: ProfileOpt = None,
    config: ScriptConfigOpt = None,
    dashboard_port: DashboardPortOpt = DASHBOARD_PORT,
    clone_venv: CloneVenvOpt = False,
    description: DescriptionOpt = None,
) -> str:
    """Run a job on a Ray cluster."""
    _ = get_aws_session(profile=profile)
    script_path = _resolve_path(script, SCRIPT_ROOT, ["py", "sh"])
    config_root = str(script_path.parent / SCRIPT_CONFIG_DIR)
    config_path = (
        _resolve_path(config, config_root, ["yaml", "yml"]) if config else None
    )
    try:
        return submit_job(
            script_path=script_path,
            clone_venv=clone_venv,
            config_path=config_path,
            dashboard_port=dashboard_port,
            description=description,
        )
    except RuntimeError as e:
        logger.error(f"Failed to submit job: {e}")
        sys.exit(-1)


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
    sync_dotfiles: SyncDotfilesOpt = False,
    no_port_forwarding: NoPortForwardingOpt = False,
    forward_port: ForwardPortOpt = None,
) -> None:
    """Create or update a Ray cluster."""
    session = get_aws_session(profile)
    config_path = _resolve_cluster_config_path(config)
    cluster_name = cluster_name or config_path.stem

    cluster_up(
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
        do_sync_dotfiles=sync_dotfiles,
    )

    # 5s is usually sufficient for the minimal workers to be in the
    # running state after the Ray cluster is up
    sleep(5)
    update_ssh_config_for_instances_with_name(
        session, instance_name=f"ray-{cluster_name}-*"
    )

    cluster_head_name = f"ray-{cluster_name}-head"
    close_ssh_connection(cluster_head_name)

    services = FORWARDED_PORTS.keys()
    local_forwarded_ports = {} if no_port_forwarding else FORWARDED_PORTS.copy()

    for el in forward_port if forward_port else []:
        service_port = el.split(",")
        if not (
            len(service_port) == 2 and service_port[0] and service_port[1].isdigit()
        ):
            logger.error(f"Invalid port forwarding specification: {el!r}.")
            sys.exit(-1)
        service, port = service_port
        if service not in services:
            logger.error(f"Invalid service name: {service!r}.")
            sys.exit(-1)
        local_forwarded_ports[service] = int(port)

    if local_forwarded_ports:
        logger.info(
            f"[{cluster_head_name}] Forwarding ports for "
            f"{', '.join(local_forwarded_ports.keys())}"
        )
        for service in local_forwarded_ports:
            open_ssh_tunnel(
                cluster_head_name,
                FORWARDED_PORTS[service],
                local_port=local_forwarded_ports[service],
            )

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

    cluster_down(
        config_path=config_path,
        cluster_name=cluster_name,
        workers_only=workers_only,
        keep_min_workers=keep_min_workers,
        prompt=prompt,
        verbose=verbose,
        show_output=show_output,
    )

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
