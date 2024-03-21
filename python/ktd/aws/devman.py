#!/usr/bin/env python3
import argparse
import inspect
import sys
from pathlib import Path

import boto3
from ktd.aws.ec2.util import (
    get_instance_name,
    get_instances,
    get_unique_instance_name,
    kill_instances_with_name,
    update_ssh_config_for_instance,
    update_ssh_config_for_instances_with_name,
    wait_for_instance_with_id,
    wait_for_stack_with_name,
)
from ktd.aws.util import sso_login
from ktd.logging import get_logger
from ktd.util.ssh import remove_from_ssh_config, wait_for_connection
from ktd.util.string import truncate
from ktd.util.vscode import open_vscode_over_ssh

logger = get_logger(__name__, format="simple")

CF_TEMPLATE_PATH = Path(__file__).parent / "cloudformation" / "templates" / "dev.yaml"
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
}
_INTERNAL_PARAMS = ["session"]
_CMD_PREFIX = "_cmd_"


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
        update_ssh_config_for_instance(instance)


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
        remove_from_ssh_config(get_unique_instance_name(instance))


def _cmd_kill(session: boto3.Session, instance_name: str) -> None:
    """Kill/terminate instances with the given name"""
    logger.info(f"[{instance_name}] Killing instances")
    kill_instances_with_name(session, instance_name, kill_stacks=True)


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
    update_ssh_config_for_instances_with_name(session, instance_name="?*")


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

    wait_for_stack_with_name(instance_name, session=session)

    # the devserver stack creates an instance with the same name
    instances = get_instances(name=instance_name, states=["running"], session=session)
    assert instances is not None and len(instances) == 1

    update_ssh_config_for_instance(instances[0])

    logger.info(f"[{instance_name}] Waiting for SSH")
    wait_for_connection(instance_name)

    if open_vscode:
        open_vscode_over_ssh(instance_name)


def _cmd_open(
    session: boto3.Session,
    instance_name: str,
    target: str = "file",
    path: str = "/home/ec2-user/dev/dev.code-workspace",
) -> None:
    """Open VS Code on the instance with the given name"""
    open_vscode_over_ssh(instance_name, target, path)


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
