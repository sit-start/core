#!/usr/bin/env python3
import argparse
import inspect
import subprocess
import sys
from pathlib import Path
from typing import Type

import boto3
import ktd.logging
from boto3.resources.base import ServiceResource
from ktd.aws.ec2.util import (
    get_instance_name,
    get_instances_with_name,
    update_ssh_config,
    wait_for_instance_with_id,
)
from ktd.aws.util import sso_login

logger = ktd.logging.get_logger(__name__)

TEMPLATE_PATH = Path(__file__).parent / "cloudformation" / "templates" / "dev.yaml"
PROJECT_PATH = "$HOME/projects"

_REPO_ROOT = "git@github.com:kevdale"  # FIXME: hardcoded
_PROJECTS = ["infra", "study"]  # FIXME: hardcoded


_PARAM_HELP = {
    "profile": "the AWS SSO profile to use for the session",
    "no_compact": "do not display compact output",
    "instance_name": "the instance name",
    "instance_type": "the instance type",
    "show_killed": "show killed instances",
}

_INTERNAL_PARAMS = ["session"]

_CMD_PREFIX = "_cmd_"


# TODO: maintain .ssh/config: remove old instances, update new ones
# TODO: can simplify the code if we always access instances via their
# stack's unique name
# TODO: add a tag to the stack and/or instance to make filtering easy


def _clone_dotfiles_and_repos(instance_name: str) -> None:
    logger.info(f"[{instance_name}] Making initial SSH connection")
    # yadm and git commands assume identity forwarding is setup in the SSH config
    # FIXME: the ssh-keyscan method is frowned upon.
    _ = subprocess.call(
        [
            "ssh",
            "-o",
            "ConnectionAttempts 10",
            "-o",
            "StrictHostKeyChecking no",
            instance_name,
            "ssh-keyscan -H github.com >> ~/.ssh/known_hosts",
        ],
        stderr=subprocess.DEVNULL,
    )

    logger.info(f"[{instance_name}] Cloning dotfiles and repos")
    # clone yadm
    subprocess.call(
        [
            "ssh",
            instance_name,
            f"yadm clone {_REPO_ROOT}/dotfiles --no-checkout; yadm checkout ~",
        ]
    )

    # clone projects
    # TODO: for this to work on first connection, I had to do
    # StrictHostChecking no. This might be fine for all aws hosts, but
    # reconsider
    if _PROJECTS:
        clone_cmds = [f"git clone {_REPO_ROOT}/{p}" for p in _PROJECTS]
        cmd = f"mkdir -p {PROJECT_PATH} && cd $_ && {'; '.join(clone_cmds)}"
        subprocess.call(["ssh", instance_name, cmd])


def _get_instance_with_name(instance_name: str) -> Type[ServiceResource]:
    instances = get_instances_with_name(instance_name)
    assert len(instances) > 0, f"[{instance_name}] No instances found with this name"
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
    instance = _get_instance_with_name(instance_name)
    assert instance.meta is not None, f"[{instance_name}] Instance has no metadata"
    host_name = instance.meta.data["PublicDnsName"]
    if not host_name:
        logger.info("No hostname to update in SSH config")
        return
    update_ssh_config(instance_name, HostName=host_name)


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


def _add_subparser(subparsers: argparse._SubParsersAction, cmd: str) -> None:
    func = getattr(sys.modules[__name__], _CMD_PREFIX + cmd)
    parser = subparsers.add_parser(cmd, help=func.__doc__)
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


def _cmd_start(session: boto3.Session, instance_name: str) -> None:
    """Start the instance with the given name"""
    logger.info(f"Starting instance {instance_name}")
    instance = _get_instance_with_name(instance_name)
    ec2_client = session.client("ec2")
    ec2_client.start_instances(InstanceIds=[instance.id])  # type: ignore
    wait_for_instance_with_id(instance.id, session=session)  # type: ignore
    _update_hostname_in_ssh_config(instance_name)


def _cmd_stop(session: boto3.Session, instance_name: str) -> None:
    """Stop the instance with the given name"""
    logger.info(f"[{instance_name}] Stopping instance")
    instance = _get_instance_with_name(instance_name)
    ec2_client = session.client("ec2")
    ec2_client.stop_instances(InstanceIds=[instance.id])  # type: ignore


def _cmd_kill(session: boto3.Session, instance_name: str) -> None:
    """Kill/terminate the instance with the given name"""
    logger.info(f"[{instance_name}] Killing instance")
    # instance = _get_instance_with_name(instance_name)
    # ec2_client = session.client("ec2")
    # ec2_client.terminate_instances(InstanceIds=[instance.id])  # type: ignore

    cf_client = session.client("cloudformation")
    cf_client.delete_stack(StackName=instance_name)

    # TODO: cleanup SSH config


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
        if show_killed or info["state"] != "terminated":
            instance_info[instance.id] = info

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
    """Refresh the SSH config for all named instances"""
    logger.info("Refreshing SSH config for all named instances")
    ec2 = session.resource("ec2")
    for instance in ec2.instances.all():
        if instance_name := get_instance_name(instance):
            _update_hostname_in_ssh_config(instance_name)


# TODO: we can't really pass in kwargs given how this is currently invoked
def _cmd_create(
    session: boto3.Session, instance_name: str, instance_type: str = "g5.2xlarge"
) -> None:
    """Create a devserver with the given name and arguments"""
    # parameters should be kept in sync with ktd/aws/cloudformation/templates/dev.yaml
    logger.info(f"[{instance_name}] Creating devserver")
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
    _clone_dotfiles_and_repos(instance_name)


def _cmd_code(session: boto3.Session, instance_name: str) -> None:
    """Open VS Code on the instance with the given name"""
    logger.info(f"[{instance_name}] Opening VS Code on instance")
    subprocess.call(
        [
            "code",
            "--",
            "--folder-uri",
            f"vscode-remote://ssh-remote+{instance_name}/home/ec2-user",
        ]
    )


def main():
    """Manages devserver instances"""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--profile", help=_PARAM_HELP["profile"])
    subparsers = parser.add_subparsers(help="Sub-command help")
    subparsers.required = True

    commands = ["start", "stop", "kill", "list", "refresh", "create", "code"]
    for cmd in commands:
        _add_subparser(subparsers, cmd)

    args = parser.parse_args()

    profile = args.profile
    delattr(args, "profile")
    func = args.func
    delattr(args, "func")

    if profile is not None:
        sso_login(profile_name=profile)
    session = boto3.Session(profile_name=profile)

    func(session, **vars(args))


if __name__ == "__main__":
    main()
