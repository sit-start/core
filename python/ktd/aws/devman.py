#!/usr/bin/env python3
import argparse
import inspect
import os
import re
import subprocess
import sys
import tempfile
from os.path import expanduser, expandvars, realpath
from pathlib import Path
from time import sleep

import boto3
import git
import json5
import ray.job_submission
import yaml
from boto3.resources.base import ServiceResource
from ktd.aws.ec2.util import get_instance_name, get_instances, wait_for_instance_with_id
from ktd.aws.util import sso_login
from ktd.cloudpathlib import CloudPath
from ktd.logging import get_logger
from ktd.util.ssh import (
    close_ssh_connection,
    open_ssh_tunnel,
    remove_from_ssh_config,
    update_ssh_config,
)
from ktd.util.string import strip_ansi_codes, truncate

logger = get_logger(__name__, format="simple")

# TODO: use subprocess.check_call() throughout

SSH_CONFIG_PATH = Path(os.environ["HOME"]) / ".ssh" / "config"
CF_TEMPLATE_PATH = Path(__file__).parent / "cloudformation" / "templates" / "dev.yaml"
RAY_CONFIG_ROOT = Path(__file__).parent.parent / "ray" / "config" / "cluster"
_PARAM_HELP = {
    # devman
    "profile": "the AWS SSO profile to use for the session",
    # common
    "instance_name": "the instance name / pattern",
    "open_vscode": "open VS Code to the `open` command's default target/path",
    # list
    "compact": "display compact output",
    "show_killed": "show killed instances",
    # create
    "instance_type": "the instance type",
    "no_clone_repos": "skip cloning repos",
    "repo_root": "the root directory for cloning repos",
    "repos": "comma-separate list of github repos to clone",
    "yadm_dotfiles_repo": "github repo to clone via yadm",
    # code
    "target": "target type to open in VS Code; one of 'file', 'folder'",
    "path": "absolute path to open in VS Code",
    # ray_up
    "config": f"the Ray cluster config in {RAY_CONFIG_ROOT}",
    "min_workers": "minimum number of workers; overrides the config",
    "max_workers": "maximum number of workers; overrides the config",
    "no_restart": "do not restart Ray services during the update",
    "restart_only": "skip running setup commands and only restart Ray",
    "cluster_name": "override the configured cluster name",
    "prompt": "prompt for confirmation",
    "verbose": "display verbose output",
    "show_output": "display output from the Ray command",
    # ray_down
    "workers_only": "only destroy workers",
    "keep_min_workers": "retain the minimal amount of workers specified in the config",
    "kill": "terminate all instances",
    # ray_submit_job
    "script_name": "name of the script to run, without the .py extension",
    "script_path": "path to the script; must exist on local and remote filesystems",
    "restart": "restart Ray services; this stops any existing jobs",
}
_INTERNAL_PARAMS = ["session"]
_CMD_PREFIX = "_cmd_"
_ALL_INSTANCE_STATES = [
    "pending",
    "running",
    "stopping",
    "stopped",
    "shutting-down",
    "terminated",
]
_FORWARDED_PORTS = {
    "Ray Dashboard": 8265,
    "Prometheus": 9090,
    "Grafana": 3000,
}
_SCRIPT_PATH_DEFAULT = "$DEV/core/python/ktd/ml/experiments/{exp_name}.py"
_RUNNER = "$DEV/core/python/ktd/util/run_from_ref.py"


def _github_ssh_keys(use_cached: bool = True) -> str:
    github_keys_path = Path(os.environ["HOME"]) / ".ssh" / "github_keys"
    if use_cached and github_keys_path.exists():
        return github_keys_path.read_text()

    path = CloudPath("https://api.github.com/meta")
    meta = json5.loads(path.read_text())
    if not isinstance(meta, dict) or "ssh_keys" not in meta:
        raise RuntimeError("Failed to fetch GitHub SSH keys")
    return "\n".join(f"github.com {v}" for v in meta["ssh_keys"])


def _open_vscode(
    session: boto3.Session,
    instance_name: str,
    target: str = "file",
    path: str = "/home/ec2-user/dev/dev.code-workspace",
) -> None:
    logger.info(f"[{instance_name}] Opening VS Code on instance")
    subprocess.call(
        [
            "code",
            "--",
            f"--{target}-uri",
            f"vscode-remote://ssh-remote+{instance_name}{path}",
        ]
    )


def _wait_for_ssh(instance_name: str, max_attempts: int = 15) -> None:
    logger.info(f"[{instance_name}] Waiting for SSH")
    subprocess.call(
        [
            "ssh",
            "-o",
            f"ConnectionAttempts {max_attempts}",
            "-o",
            "BatchMode yes",
            "-o",
            "StrictHostKeyChecking no",
            instance_name,
            "true",
        ]
    )


def _get_instance_name_for_ssh_config(instance: ServiceResource) -> str:
    """Attempts to return a unique name for the given instance"""
    name = get_instance_name(instance)
    id = instance.id  # type: ignore
    if name is None:
        logger.warning(f"Instance has no name; using instance ID {id}")
        return id
    expected_duplicate_patterns = ["^ray-.*-worker$"]
    for pattern in expected_duplicate_patterns:
        if re.match(pattern, name):
            return f"{name}-{id}"
    return name


def _update_hostname_in_ssh_config(instance: ServiceResource) -> None:
    instance_name = _get_instance_name_for_ssh_config(instance)
    assert instance.meta is not None, f"[{instance_name}] Instance has no metadata"
    host_name = instance.meta.data["PublicDnsName"]
    if not host_name:
        logger.info("No hostname to update in SSH config")
        return
    update_ssh_config(instance_name, HostName=host_name, path=str(SSH_CONFIG_PATH))


def _wait_for_stack_with_name(
    stack_name: str,
    session: boto3.Session | None = None,
    delay_sec: int = 15,
    max_attempts: int = 20,
) -> None:
    logger.info(f"[{stack_name}] Waiting for stack to be ready")
    session = session or boto3.Session()
    client = session.client("cloudformation")
    waiter = client.get_waiter("stack_create_complete")
    config = {"Delay": delay_sec, "MaxAttempts": max_attempts}
    waiter.wait(StackName=stack_name, WaiterConfig=config)


def _kill_instances(
    session: boto3.Session,
    instances: list[ServiceResource],
    update_ssh_config: bool = True,
    kill_stacks: bool = False,
) -> None:
    if not instances:
        logger.warning("No instances to kill")
        return
    instance_ids = [instance.id for instance in instances]  # type: ignore
    ec2_client = session.client("ec2")
    ec2_client.terminate_instances(InstanceIds=instance_ids)  # type: ignore

    # also terminate a stack if it exists, as is the case for
    # devservers, and remove from the SSH config
    if kill_stacks:
        cf_client = session.client("cloudformation")
        for instance in instances:
            if name := get_instance_name(instance):
                logger.info(f"[{name}] Killing stack")
                cf_client.delete_stack(StackName=name)  # type: ignore

    if update_ssh_config:
        for instance in instances:
            remove_from_ssh_config(_get_instance_name_for_ssh_config(instance))


def _kill_instances_by_name(
    session: boto3.Session,
    instance_name: str,
    states: list[str] | None = None,
    update_ssh_config: bool = True,
    kill_stacks: bool = False,
) -> None:
    states = states or list(set(_ALL_INSTANCE_STATES).difference(["terminated"]))
    instances = get_instances(name=instance_name, states=states, session=session)
    _kill_instances(
        session, instances, update_ssh_config=update_ssh_config, kill_stacks=kill_stacks
    )


def _add_subparser(
    subparsers: argparse._SubParsersAction, cmd: str, aliases: list[str] | None = None
) -> None:
    """Adds a subparser for the given command.

    Automatically generates the parser based on the function signature
    and type annotations. The corresponding function must be prefixed
    with _CMD_PREFIX. Function arguments must be annotated; default
    values cannot include `None`. Boolean arguments must default to
    False.
    """

    func = getattr(sys.modules[__name__], _CMD_PREFIX + cmd)
    parser = subparsers.add_parser(
        cmd,
        help=func.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        aliases=aliases or [],
    )
    for p in inspect.signature(func).parameters.values():
        if p.name in _INTERNAL_PARAMS:
            continue
        args = [p.name if p.default is p.empty else f"--{p.name}"]
        kwargs = {
            "default": None if p.default is p.empty else p.default,
            "help": _PARAM_HELP[p.name],
        }
        if p.annotation == bool:
            assert (
                p.default is p.empty or p.default is False
            ), "boolean param must be False by default"
            kwargs["action"] = "store_true"
        else:
            kwargs["type"] = p.annotation
        parser.add_argument(*args, **kwargs)
    parser.set_defaults(func=func)
    return parser


def _update_hostnames_in_ssh_config(
    session: boto3.Session, instance_name: str = "?*"
) -> None:
    """Update the SSH config for all running instances with the given name"""
    logger.info(f"[{instance_name}] Updating hostnames in SSH config")

    running_instances: list[ServiceResource] = []
    other_instance_names: list[str] = []

    for instance in get_instances(name=instance_name, session=session):
        this_instance_name = _get_instance_name_for_ssh_config(instance)
        if instance.state["Name"].strip() == "running":  # type: ignore
            running_instances.append(instance)
        else:
            other_instance_names.append(this_instance_name)

    for this_instance_name in set(other_instance_names):
        remove_from_ssh_config(this_instance_name)
    for instance in running_instances:
        _update_hostname_in_ssh_config(instance)


def _ray_up(
    config: str = "main",
    min_workers: int = -1,
    max_workers: int = -1,
    no_restart: bool = False,
    restart_only: bool = False,
    cluster_name: str = "",
    prompt: bool = False,
    verbose: bool = False,
    show_output: bool = False,
) -> None:
    cmd = ["ray", "up", str(RAY_CONFIG_ROOT / f"{config}.yaml")]
    if min_workers >= 0:
        cmd += ["--min-workers", str(min_workers)]
    if max_workers >= 0:
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

    logger.info(f"[{cluster_name}] Creating or updating Ray cluster")
    if show_output:
        subprocess.run(cmd, check=True)
    else:
        log_path = Path(tempfile.mkdtemp(prefix="/tmp/")) / f"ray_up_{cluster_name}.log"
        logger.info(f"[{cluster_name}] Ray output written to {log_path}")
        cmd += ["--log-color", "false"]
        with open(log_path, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        # even with --log-color false, the log still contains ANSI codes, so
        # strip them
        log_path.write_text(strip_ansi_codes(log_path.read_text()))


def _cmd_start(session: boto3.Session, instance_name: str) -> None:
    """Start instances with the given name"""
    logger.info(f"[{instance_name}] Starting instances")

    states = ["stopping", "stopped"]
    instances = get_instances(name=instance_name, states=states, session=session)
    if not instances:
        logger.info(f"[{instance_name}] Instance in {'/'.join(states)} state not found")
        return

    for instance in instances:
        ec2_client = session.client("ec2")
        ec2_client.start_instances(InstanceIds=[instance.id])  # type: ignore
        wait_for_instance_with_id(instance.id, session=session)  # type: ignore
        _update_hostname_in_ssh_config(instance)


def _cmd_stop(session: boto3.Session, instance_name: str) -> None:
    """Stop instances with the given name"""
    logger.info(f"[{instance_name}] Stopping instances")

    states = ["pending", "running"]
    instances = get_instances(name=instance_name, states=states, session=session)
    if not instances:
        logger.info(f"[{instance_name}] Instance in {'/'.join(states)} state not found")
        return

    instance_ids = [instance.id for instance in instances]  # type: ignore
    ec2_client = session.client("ec2")
    ec2_client.stop_instances(InstanceIds=instance_ids)  # type: ignore

    for instance in instances:
        remove_from_ssh_config(_get_instance_name_for_ssh_config(instance))


def _cmd_kill(session: boto3.Session, instance_name: str) -> None:
    """Kill/terminate instances with the given name"""
    logger.info(f"[{instance_name}] Killing instances")
    _kill_instances_by_name(session, instance_name, kill_stacks=True)


def _cmd_list(
    session: boto3.Session,
    show_killed: bool = False,
    compact: bool = False,
) -> None:
    """List instances"""
    # FIXME: track stacks w/ specific tag, not instances, since it's
    # possible for, e.g., stack creation to succeed but instance
    # creation to fail
    instance_info = {}
    for instance in session.resource("ec2").instances.all():
        info = {
            "state": instance.state["Name"].strip(),
            "dns_name": instance.public_dns_name.strip(),
            "ip": (instance.private_ip_address or "").strip(),
            "name": str(get_instance_name(instance)),
            "instance_type": instance.instance_type.strip(),
        }
        if show_killed or (
            info["state"] != "terminated" and info["state"] != "shutting-down"
        ):
            instance_info[instance.id] = info

    if len(instance_info) == 0:
        logger.info("No instances found")
        return

    name_width = max([len(i["name"]) for i in instance_info.values()])
    type_width = max([len(i["instance_type"]) for i in instance_info.values()])
    state_width = max([len(i["state"]) for i in instance_info.values()])
    ip_width = max([len(i["ip"]) for i in instance_info.values()])
    id_width = max([len(i) for i in instance_info.keys()])

    if compact:
        name_width = min(name_width, 8)
        type_width = min(type_width, 11)
        state_width = min(state_width, 7)

    for id, info in instance_info.items():
        if info["state"] == "terminated" and not show_killed:
            continue

        if compact:
            info["state"] = truncate(info["state"], state_width)
            info["instance_type"] = truncate(info["instance_type"], type_width)
            info["name"] = truncate(info["name"], name_width)

        logger.info(
            f"{info['state']:>{state_width}} {info['name']:<{name_width}} "
            f"{info['instance_type']:<{type_width}} {id:<{id_width}} "
            f"{info['ip']:<{ip_width}} {info['dns_name']}"
        )


def _cmd_refresh(session: boto3.Session) -> None:
    """Refresh hostnames in the SSH config for all running named instances"""
    _update_hostnames_in_ssh_config(session, instance_name="?*")


def _cmd_create(
    session: boto3.Session,
    instance_name: str,
    instance_type: str = "g5g.2xlarge",
    no_clone_repos: bool = False,
    repo_root: str = "/home/ec2-user",
    repos: str = "kevdale/dev",
    yadm_dotfiles_repo: str = "kevdale/dotfiles",
    open_vscode: bool = False,
) -> None:
    """Create a devserver with the given name and arguments"""
    # parameters should be kept in sync with ktd/aws/cloudformation/templates/dev.yaml
    logger.info(f"[{instance_name}] Creating devserver")

    states = ["pending", "running", "stopping", "stopped"]
    if get_instances(name=instance_name, states=states, session=session):
        logger.info(f"[{instance_name}] Instance name in use. Aborting.")
        return

    cf_client = session.client("cloudformation")
    cf_client.create_stack(  # type: ignore
        StackName=instance_name,
        TemplateBody=CF_TEMPLATE_PATH.read_text(),
        Capabilities=["CAPABILITY_IAM"],
        Parameters=[
            {"ParameterKey": "InstanceType", "ParameterValue": instance_type},
            {
                "ParameterKey": "CloneRepositories",
                "ParameterValue": "false" if no_clone_repos else "true",
            },
        ],
    )

    _wait_for_stack_with_name(instance_name, session=session)

    # the devserver stack creates an instance with the same name
    instances = get_instances(name=instance_name, states=["running"], session=session)
    assert instances is not None and len(instances) == 1

    _update_hostname_in_ssh_config(instances[0])

    _wait_for_ssh(instance_name)

    if open_vscode:
        _open_vscode(session, instance_name)


def _cmd_open(
    session: boto3.Session,
    instance_name: str,
    target: str = "file",
    path: str = "/home/ec2-user/dev/dev.code-workspace",
) -> None:
    """Open VS Code on the instance with the given name"""
    _open_vscode(session, instance_name, target, path)


def _cmd_ray_up(
    session: boto3.Session,
    config: str = "main",
    min_workers: int = -1,
    max_workers: int = -1,
    no_restart: bool = False,
    restart_only: bool = False,
    cluster_name: str = "",
    prompt: bool = False,
    verbose: bool = False,
    open_vscode: bool = False,
    show_output: bool = False,
) -> None:
    """Create or update a ray cluster"""
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
    )

    # 5s is usually sufficient for the minimal workers to be in the
    # running state after the Ray cluster is up
    sleep(5)
    cluster_name = cluster_name or config
    _update_hostnames_in_ssh_config(session, instance_name=f"ray-{cluster_name}-*")

    cluster_head_name = f"ray-{cluster_name}-head"
    logger.info(
        f"[{cluster_head_name}] Forwarding ports for "
        f"{', '.join(_FORWARDED_PORTS.keys())}"
    )

    close_ssh_connection(cluster_head_name)
    for port in _FORWARDED_PORTS.values():
        open_ssh_tunnel(cluster_head_name, port)

    if open_vscode:
        _open_vscode(session, cluster_head_name)


def _cmd_ray_down(
    session: boto3.Session,
    cluster_name: str = "main",
    workers_only: bool = False,
    keep_min_workers: bool = False,
    prompt: bool = False,
    verbose: bool = False,
    kill: bool = False,
    show_output: bool = False,
) -> None:
    """Tear down a Ray cluster"""
    logger.info(f"[{cluster_name}] Tearing down Ray cluster")

    cmd = ["ray", "down", str(RAY_CONFIG_ROOT / f"{cluster_name}.yaml")]
    if workers_only:
        cmd.append("--workers-only")
    if keep_min_workers:
        cmd.append("--keep-min-workers")
    if not prompt:
        cmd.append("--yes")
    if verbose:
        cmd.append("--verbose")

    if show_output:
        subprocess.call(cmd)
    else:
        log_path = (
            Path(tempfile.mkdtemp(prefix="/tmp/")) / f"ray_down_{cluster_name}.log"
        )
        logger.info(f"[{cluster_name}] Ray output written to {log_path}")
        cmd += ["--log-color", "false"]
        with open(log_path, "w") as f:
            subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        # even with --log-color false, the log still contains ANSI codes, so
        # strip them
        log_path.write_text(strip_ansi_codes(log_path.read_text()))

    cluster_head_name = f"ray-{cluster_name}-head"
    cluster_names = f"ray-{cluster_name}-*"
    worker_names = f"ray-{cluster_name}-workers"

    logger.info(f"Closing SSH connections to {cluster_head_name}")
    try:
        close_ssh_connection(cluster_head_name)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to close SSH connections: {e}")

    if kill and keep_min_workers:
        logger.warning("Killing instances with keep_min_workers is not implemented")
    elif kill:
        instance_name = worker_names if workers_only else cluster_names
        logger.info(f"[{instance_name}] Killing instances")
        _kill_instances_by_name(session, instance_name, update_ssh_config=False)

    _update_hostnames_in_ssh_config(session, instance_name=cluster_names)


def _cmd_ray_stop_all_jobs(session: boto3.Session) -> None:
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


def _cmd_ray_monitor(session: boto3.Session, cluster_name: str = "main") -> None:
    """Monitor autoscaling for a Ray cluster"""
    log_path = "/tmp/ray/session_latest/logs/monitor*"
    cmd = [
        "ray",
        "exec",
        str(RAY_CONFIG_ROOT / f"{cluster_name}.yaml"),
        "--",
        f"tail -n 100 -f {log_path}",
    ]
    logger.info(" ".join(cmd))
    subprocess.call(cmd)


# TODO: control env vars here as well; use ray envs
# TODO: support additional user-specified repos on local+remote hosts
def _cmd_ray_submit_job(
    session: boto3.Session,
    script_name: str = "",
    script_path: str = _SCRIPT_PATH_DEFAULT,
    config: str = "main",
    cluster_name: str = "",
    restart: bool = False,
) -> None:
    """Runs a job on the Ray cluster"""
    # get script name
    if not script_name and script_path == _SCRIPT_PATH_DEFAULT:
        raise RuntimeError("exp_name or exp_path must be provided")
    script_path = script_path.format(exp_name=script_name)

    # get repo
    try:
        repo = git.Repo(expandvars(script_path), search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        logger.error(f"No git repository found for {script_path!r}; exiting")
        sys.exit(-1)

    # ensure the repo is in cluster config's file mounts, as we'll use that
    # to sync repo state with head and worker nodes
    config_path = RAY_CONFIG_ROOT / f"{config}.yaml"
    config_data = yaml.load(config_path.read_text(), Loader=yaml.FullLoader)
    file_mounts = [realpath(expanduser(f)) for f in config_data["file_mounts"].values()]
    repo_path = Path(repo.working_dir).resolve()
    if not any(repo_path.is_relative_to(f) for f in file_mounts):
        msg = f"Repo {repo.working_dir!r} not in file_mounts and cannot be synced."
        raise RuntimeError(msg)

    # invoke ray-up
    logger.info("Running 'ray up' to sync files and run setup commands")
    _ray_up(config=config, no_restart=not restart, cluster_name=cluster_name)

    # run a basic job that uses the native environment and existing file(s)
    client = ray.job_submission.JobSubmissionClient("http://127.0.0.1:8265")
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


def main():
    """Manages devserver instances and cluster resources"""
    parser = argparse.ArgumentParser(
        description=main.__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--profile", help=_PARAM_HELP["profile"])
    subparsers = parser.add_subparsers(help="Sub-command help")
    subparsers.required = True

    commands = {
        "start": ["s"],
        "stop": ["h"],
        "kill": ["k"],
        "list": ["ls"],
        "refresh": ["r"],
        "create": ["c"],
        "open": ["o"],
        "ray_up": ["u", "up"],
        "ray_down": ["d", "down"],
        "ray_monitor": ["m", "monitor"],
        "ray_submit_job": ["submit_job", "sj"],
        "ray_stop_all_jobs": ["stop_jobs", "kj"],
    }
    for cmd, aliases in commands.items():
        _add_subparser(subparsers, cmd, aliases)

    args = parser.parse_args()

    profile = args.profile
    delattr(args, "profile")
    func = args.func
    delattr(args, "func")

    sso_login(profile_name=profile)
    session = boto3.Session(profile_name=profile)

    func(session, **vars(args))


if __name__ == "__main__":
    main()
