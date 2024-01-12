#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, Type

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


# TODO: maintain .ssh/config: remove old instances, update new ones
# TODO: can simplify the code if we always access instances via their stack's unique name
# TODO: add a tag to the stack and/or instance to make filtering easy


def _clone_dotfiles_and_repos(instance_name: str) -> None:
    logger.info(f"Cloning dotfiles and repos on instance {instance_name}")
    # yadm and git commands assume identity forwarding is setup in the SSH config
    _ = subprocess.call(
        ["ssh", instance_name, "ssh-keyscan -H github.com >> ~/.ssh/known_hosts"],
        stderr=subprocess.DEVNULL,
    )

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
    assert len(instances) > 0, f"No instances found with name {instance_name}"
    if len(instances) > 1:
        logger.warning(
            f"Found multiple instances with name {instance_name}; "
            "using the first one"
        )
    return instances[0]


def _trunc(s: str, max_len: int = 50) -> str:
    suffix = "..."
    max_len -= len(suffix)
    return s if len(s) <= max_len else s[:max_len] + suffix


def _update_hostname_in_ssh_config(instance_name: str) -> None:
    logger.info(f"Updating hostname in SSH config for instance {instance_name}")
    instance = _get_instance_with_name(instance_name)
    assert instance.meta is not None, f"Instance {instance_name} has no metadata"
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


def _get_command(name: str) -> Callable:
    return getattr(sys.modules[__name__], "_cmd_" + name)


def _cmd_start(session: boto3.Session, instance_name: str) -> None:
    logger.info(f"Starting instance {instance_name}")
    instance = _get_instance_with_name(instance_name)
    ec2_client = session.client("ec2")
    ec2_client.start_instances(InstanceIds=[instance.id])  # type: ignore
    wait_for_instance_with_id(instance.id, session=session)  # type: ignore
    _update_hostname_in_ssh_config(instance_name)


def _cmd_stop(session: boto3.Session, instance_name: str) -> None:
    logger.info(f"Stopping instance {instance_name}")
    instance = _get_instance_with_name(instance_name)
    ec2_client = session.client("ec2")
    ec2_client.stop_instances(InstanceIds=[instance.id])  # type: ignore


def _cmd_kill(session: boto3.Session, instance_name: str) -> None:
    logger.info(f"Killing instance {instance_name}")
    # instance = _get_instance_with_name(instance_name)
    # ec2_client = session.client("ec2")
    # ec2_client.terminate_instances(InstanceIds=[instance.id])  # type: ignore

    cf_client = session.client("cloudformation")
    cf_client.delete_stack(StackName=instance_name)


def _cmd_list(session: boto3.Session, show_terminated: bool = True) -> None:
    # FIXME: track stacks w/ specific tag, not instances, since it's
    # possible for, e.g., stack creation to succeed but instance
    # creation to fail
    ec2 = session.resource("ec2")
    name_width = 25
    for instance in ec2.instances.all():
        state = instance.state["Name"].strip()
        if state == "terminated" and not show_terminated:
            continue
        dns_name = instance.public_dns_name.strip()
        name = _trunc(str(get_instance_name(instance)), name_width)
        logger.info(f"{state:13} {instance.id} {name:<{name_width}} {dns_name}")


def _cmd_refresh(session: boto3.Session) -> None:
    logger.info("Refreshing SSH config for all named instances")
    ec2 = session.resource("ec2")
    for instance in ec2.instances.all():
        if instance_name := get_instance_name(instance):
            _update_hostname_in_ssh_config(instance_name)


# TODO: we can't really pass in kwargs given how this is currently invoked
def _cmd_create(session: boto3.Session, instance_name: str, **kwargs) -> None:
    """Create a devserver with the given name and arguments

    See ktd/aws/cloudformation/templates/dev.yaml for parameters and
    defaults.
    """
    logger.info(f"Creating devserver with name {instance_name}")
    cf_client = session.client("cloudformation")
    cf_client.create_stack(
        StackName=instance_name,
        TemplateBody=TEMPLATE_PATH.read_text(),
        Parameters=[
            {"ParameterKey": k, "ParameterValue": v} for k, v in kwargs.items()
        ],
    )
    logger.info(f"Created instance and stack with name {instance_name}")
    _wait_for_stack_with_name(instance_name, session=session)
    _update_hostname_in_ssh_config(instance_name)

    # FIXME: this tends to result in 'connection refused', so we need to
    # add proper return values and retry logic
    _clone_dotfiles_and_repos(instance_name)

    # TODO: can optionally open with:
    # $ code -- --folder-uri vscode-remote://ssh-remote+dev-main/home/ec2-user


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="the command to run", nargs="*")
    parser.add_argument(
        "--profile",
        help="the AWS SSO profile to use for the session",
        default=None,
    )
    # TODO: Generate doc for all commands, esp. create, and add to
    # help. Reference dev.yaml and expected format for kwargs

    args, cmd_args = parser.parse_known_args()
    cmd_pargs = args.command[1:]
    cmd_kwargs = {k: v for k, v in [a[2:].split("=") for a in cmd_args]}

    if args.profile is not None:
        sso_login(profile_name=args.profile)

    session = boto3.Session(profile_name=args.profile)

    cmd = _get_command(args.command[0])
    cmd(session, *cmd_pargs, **cmd_kwargs)


if __name__ == "__main__":
    main()
