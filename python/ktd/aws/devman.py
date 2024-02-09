#!/usr/bin/env python3
import argparse
import inspect
import re
import subprocess
import sys
from pathlib import Path
from typing import Type
from time import sleep

import boto3
import json5
import ktd.logging
from boto3.resources.base import ServiceResource
from ktd.aws.ec2.util import (
    get_instance_name,
    get_instances_with_name,
    remove_from_ssh_config,
    update_ssh_config,
    wait_for_instance_with_id,
)
from ktd.aws.util import sso_login
from ktd.cloudpathlib import CloudPath

logger = ktd.logging.get_logger(__name__)

TEMPLATE_PATH = Path(__file__).parent / "cloudformation" / "templates" / "dev.yaml"
RAY_CONFIG_ROOT = Path(__file__).parent.parent / "ray" / "cluster" / "config"
PROJECT_PATH = "$HOME/projects"
_PARAM_HELP = {
    # devman
    "profile": "the AWS SSO profile to use for the session",
    # common
    "instance_name": "the instance name",
    # list
    "no_compact": "do not display compact output",
    "show_killed": "show killed instances",
    # create
    "instance_type": "the instance type",
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
    "prompt": "prompt for confirmation",
    "verbose": "display verbose output",
    # ray_down
    "workers_only": "only destroy workers",
    "keep_min_workers": "retain the minimal amount of workers specified in the config",
}
_INTERNAL_PARAMS = ["session"]
_CMD_PREFIX = "_cmd_"


# TODO: maintain .ssh/config: remove old instances, update new ones
# TODO: can simplify the code if we always access instances via their
# stack's unique name
# TODO: add a tag to the stack and/or instance to make filtering easy


def _github_ssh_keys(use_cached: bool = False) -> str:
    github_keys_path = Path(os.environ["HOME"]) / ".ssh" / "github_keys"
    if use_cached and github_keys_path.exists():
        return github_keys_path.read_text()

    path = CloudPath("https://api.github.com/meta")
    meta = json5.loads(path.read_text())
    if not isinstance(meta, dict) or "ssh_keys" not in meta:
        raise RuntimeError("Failed to fetch GitHub SSH keys")
    return "\n".join(f"github.com {v}" for v in meta["ssh_keys"])


def _clone_repos(
    instance_name: str,
    repos: list[str] | None,
    yadm_dotfiles_repo: str | None,
) -> None:
    # NOTE - yadm and git commands assume identity forwarding is setup in the SSH config

    logger.info(f"[{instance_name}] Updating known hosts over initial SSH connection")
    # add github to known hosts, with retries on the initial connection
    github_keys = _github_ssh_keys()
    subprocess.call(
        [
            "ssh",
            "-o",
            "ConnectionAttempts 10",
            "-o",
            "StrictHostKeyChecking no",
            instance_name,
            f"echo '{github_keys}' >> ~/.ssh/known_hosts",
        ]
    )

    if yadm_dotfiles_repo:
        logger.info(f"[{instance_name}] Cloning dotfiles")
        subprocess.call(
            [
                "ssh",
                instance_name,
                f"yadm clone git@github.com:{yadm_dotfiles_repo} --no-checkout; "
                "yadm checkout ~",
            ]
        )

    if repos:
        logger.info(f"[{instance_name}] Cloning repos")
        clone_cmds = [f"git clone git@github.com:{r}" for r in repos]
        cmd = f"mkdir -p {PROJECT_PATH} && cd $_ && {'; '.join(clone_cmds)}"
        subprocess.call(["ssh", instance_name, cmd])


def _get_instance_with_name(
    instance_name: str, states: list[str] | None = None
) -> Type[ServiceResource] | None:
    instances = get_instances_with_name(instance_name, states=states)
    if not instances:
        return None
    if len(instances) > 1:
        logger.warning(
            f"[{instance_name}] Found multiple instances with this name; "
            "using the first one"
        )
    return instances[0]


def _trunc(s: str, max_len: int = 50) -> str:
    suffix = "..."
    return s if len(s) <= max_len else s[: max_len - len(suffix)] + suffix


def _update_hostname_in_ssh_config(instance_name: str) -> None:
    logger.info(f"[{instance_name}] Updating hostname in SSH config")

    states = ["running"]
    instance = _get_instance_with_name(instance_name, states=states)
    if not instance:
        logger.info(f"[{instance_name}] Instance in {'/'.join(states)} state not found")
        return

    assert instance.meta is not None, f"[{instance_name}] Instance has no metadata"
    host_name = instance.meta.data["PublicDnsName"]
    if not host_name:
        logger.info("No hostname to update in SSH config")
        return
    update_ssh_config(instance_name, HostName=host_name)


def _remove_from_ssh_config(instance_name: str) -> None:
    instance = _get_instance_with_name(instance_name)
    if not instance:
        return
    if remove_from_ssh_config(instance_name):
        logger.info(f"[{instance_name}] Removing entry in SSH config")


def _wait_for_stack_with_name(
    stack_name: str,
    session: boto3.Session | None = None,
    delay_sec: int = 15,
    max_attempts: int = 20,
) -> None:
    session = session or boto3.Session()
    client = session.client("cloudformation")
    waiter = client.get_waiter("stack_create_complete")
    config = {"Delay": delay_sec, "MaxAttempts": max_attempts}
    waiter.wait(StackName=stack_name, WaiterConfig=config)


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


def _update_ssh_config(session: boto3.Session, regex: str = ".*") -> None:
    """Update the SSH config for all running instances that match the given regex"""
    logger.info(
        f"Updating the SSH config for all running instances that match {repr(regex)}"
    )

    ec2 = session.resource("ec2")

    running_instances: list[str] = []
    other_instances: list[str] = []

    for instance in ec2.instances.all():
        if (instance_name := get_instance_name(instance)) is None:
            continue
        if not re.match(regex, instance_name):
            continue
        if instance.state["Name"].strip() == "running":
            running_instances.append(instance_name)
        else:
            other_instances.append(instance_name)

    for instance in other_instances:
        _remove_from_ssh_config(instance)
    for instance in running_instances:
        _update_hostname_in_ssh_config(instance)


def _cmd_start(session: boto3.Session, instance_name: str) -> None:
    """Start the instance with the given name"""
    logger.info(f"[{instance_name}] Starting instance")

    states = ["stopping", "stopped"]
    instance = _get_instance_with_name(instance_name, states=states)
    if not instance:
        logger.info(f"[{instance_name}] Instance in {'/'.join(states)} state not found")
        return

    ec2_client = session.client("ec2")
    ec2_client.start_instances(InstanceIds=[instance.id])  # type: ignore
    wait_for_instance_with_id(instance.id, session=session)  # type: ignore
    _update_hostname_in_ssh_config(instance_name)


def _cmd_stop(session: boto3.Session, instance_name: str) -> None:
    """Stop the instance with the given name"""
    logger.info(f"[{instance_name}] Stopping instance")

    states = ["pending", "running"]
    instance = _get_instance_with_name(instance_name, states=states)
    if not instance:
        logger.info(f"[{instance_name}] Instance in {'/'.join(states)} state not found")
        return

    ec2_client = session.client("ec2")
    ec2_client.stop_instances(InstanceIds=[instance.id])  # type: ignore

    _remove_from_ssh_config(instance_name)


def _cmd_kill(session: boto3.Session, instance_name: str) -> None:
    """Kill/terminate the instance with the given name"""
    logger.info(f"[{instance_name}] Killing instance")
    # instance = _get_instance_with_name(instance_name)
    # ec2_client = session.client("ec2")
    # ec2_client.terminate_instances(InstanceIds=[instance.id])  # type: ignore

    cf_client = session.client("cloudformation")
    cf_client.delete_stack(StackName=instance_name)

    _remove_from_ssh_config(instance_name)


def _cmd_list(
    session: boto3.Session, show_killed: bool = False, no_compact: bool = False
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

    if not no_compact:
        name_width = min(name_width, 8)
        type_width = min(type_width, 11)
        state_width = min(state_width, 7)

    for id, info in instance_info.items():
        if info["state"] == "terminated" and not show_killed:
            continue

        if not no_compact:
            info["state"] = _trunc(info["state"], state_width)
            info["instance_type"] = _trunc(info["instance_type"], type_width)
            info["name"] = _trunc(info["name"], name_width)

        logger.info(
            f"{info['state']:>{state_width}} {info['name']:<{name_width}} "
            f"{info['instance_type']:<{type_width}} {id} {info['dns_name']}"
        )


def _cmd_refresh(session: boto3.Session) -> None:
    """Refresh the SSH config for all running named instances"""
    _update_ssh_config(session)


# TODO: we can't really pass in kwargs given how this is currently invoked
def _cmd_create(
    session: boto3.Session,
    instance_name: str,
    instance_type: str = "g5g.2xlarge",
    repos: str = "kevdale/core,kevdale/study",
    yadm_dotfiles_repo: str = "kevdale/dotfiles",
) -> None:
    """Create a devserver with the given name and arguments"""
    # parameters should be kept in sync with ktd/aws/cloudformation/templates/dev.yaml
    logger.info(f"[{instance_name}] Creating devserver")

    states = ["pending", "running", "stopping", "stopped"]
    if _get_instance_with_name(instance_name, states=states):
        logger.info(f"[{instance_name}] Instance name in use. Aborting.")
        return

    cf_client = session.client("cloudformation")
    cf_client.create_stack(
        StackName=instance_name,
        TemplateBody=TEMPLATE_PATH.read_text(),
        Parameters=[
            {"ParameterKey": "InstanceType", "ParameterValue": instance_type},
        ],
    )
    logger.info(f"[{instance_name}] Waiting for instance to be ready")
    _wait_for_stack_with_name(instance_name, session=session)
    _update_hostname_in_ssh_config(instance_name)
    _clone_repos(
        instance_name, repos=repos.split(","), yadm_dotfiles_repo=yadm_dotfiles_repo
    )


def _cmd_open(
    session: boto3.Session,
    instance_name: str,
    target: str = "file",
    path: str = "/home/ec2-user/projects/study/study.code-workspace",
) -> None:
    """Open VS Code on the instance with the given name"""
    logger.info(f"[{instance_name}] Opening VS Code on instance")
    subprocess.call(
        [
            "code",
            "--",
            f"--{target}-uri",
            f"vscode-remote://ssh-remote+{instance_name}{path}",
        ]
    )


def _cmd_ray_up(
    session: boto3.Session,
    config: str = "g5g",
    min_workers: int = -1,
    max_workers: int = -1,
    no_restart: bool = False,
    restart_only: bool = False,
    prompt: bool = False,
    verbose: bool = False,
) -> None:
    logger.info(f"[{config}] Creating or updating Ray cluster")

    cmd = ["ray", "up", str(RAY_CONFIG_ROOT / f"{config}.yaml")]
    if min_workers >= 0:
        cmd += ["--min-workers", str(min_workers)]
    if max_workers >= 0:
        cmd += ["--max-workers", str(max_workers)]
    if no_restart:
        cmd.append("--no-restart")
    if restart_only:
        cmd.append("--restart-only")
    if not prompt:
        cmd.append("--yes")
    if verbose:
        cmd.append("--verbose")

    subprocess.call(cmd)

    # 5s is usually sufficient for the minimal workers to be in the
    # running state after the Ray cluster is up
    sleep(5)
    _update_ssh_config(session, regex=f"ray-{config}.*")
    logger.info(
        f"[{config}] Use `refresh (r)` to update the SSH config for any workers "
        "not yet running."
    )


def _cmd_ray_down(
    session: boto3.Session,
    config: str = "g5g",
    workers_only: bool = False,
    keep_min_workers: bool = False,
    prompt: bool = False,
    verbose: bool = False,
) -> None:
    logger.info(f"[{config}] Tearing down Ray cluster")

    cmd = ["ray", "down", str(RAY_CONFIG_ROOT / f"{config}.yaml")]
    if workers_only:
        cmd.append("--workers-only")
    if keep_min_workers:
        cmd.append("--keep-min-workers")
    if not prompt:
        cmd.append("--yes")
    if verbose:
        cmd.append("--verbose")

    subprocess.call(cmd)

    _update_ssh_config(session, regex=f"ray-{config}.*")

    # TODO: add kill option, based on _get_instances_that_match(regex)


def main():
    """Manages devserver instances"""
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
        "ray_up": ["u"],
        "ray_down": ["d"],
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
