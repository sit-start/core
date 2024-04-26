import getpass
from pathlib import Path

import typer
from typer import Argument, Option

from ktd import PYTHON_ROOT
from ktd.aws.ec2.util import (
    get_instance_name,
    get_instances,
    get_unique_instance_name,
    kill_instances_with_name,
    update_ssh_config_for_instance,
    update_ssh_config_for_instances_with_name,
    wait_for_cloud_init,
    wait_for_instance_with_id,
    wait_for_stack_with_name,
)
from ktd.aws.util import get_aws_session
from ktd.logging import get_logger
from ktd.util.ssh import remove_from_ssh_config, wait_for_connection
from ktd.util.string import truncate
from ktd.util.system import (
    DEFAULT_DOTFILES_REPO_URL,
    deploy_dotfiles,
    get_system_config,
)
from ktd.util.vscode import (
    DEFAULT_FOLDER,
    DEFAULT_TARGET,
    VSCodeTarget,
    open_vscode_over_ssh,
)

CF_TEMPLATE_PATH = f"{PYTHON_ROOT}/ktd/aws/cloudformation/templates/dev.yaml"
DEFAULT_INSTANCE_TYPE = "g5.xlarge"

app = typer.Typer()
logger = get_logger(__name__, format="simple")

# Arguments and options
_instance_name_arg = Argument(
    ...,
    help="The instance name or name pattern.",
    show_default=False,
)
_profile_opt = Option(
    None,
    help="The AWS profile to use.",
    envvar="AWS_PROFILE",
    show_default=False,
)
_instance_type_opt = Option(DEFAULT_INSTANCE_TYPE, help="The instance type to create.")
_open_vscode_opt = Option(
    False,
    "--open-vscode",
    help=f"Open VS Code to the default {DEFAULT_TARGET} ({DEFAULT_FOLDER}).",
    show_default=False,
)
_show_killed_opt = Option(
    False,
    "--show-killed",
    help="Show killed instances.",
    show_default=False,
)
_compact_opt = Option(
    False,
    "--compact",
    help="Display compact output.",
    show_default=False,
)
_target_opt = Option(
    DEFAULT_TARGET,
    help="Target type to open in VS Code; one of 'file', 'folder'.",
    show_choices=True,
)
_path_opt = Option(
    DEFAULT_FOLDER,
    help="Absolute path to open in VS Code.",
)
_no_dotfiles_opt = Option(
    False,
    "--no-dotfiles",
    help="Do not install user dotfiles.",
    show_default=False,
)
_dotfiles_repo_opt = Option(
    DEFAULT_DOTFILES_REPO_URL.format(user=getpass.getuser()),
    help="The git repo from which to install dotfiles with yadm.",
)


@app.command()
def create(
    instance_name: str = _instance_name_arg,
    profile: str = _profile_opt,
    instance_type: str = _instance_type_opt,
    open_vscode: bool = _open_vscode_opt,
    no_dotfiles: bool = _no_dotfiles_opt,
    dotfiles_repo: str = _dotfiles_repo_opt,
) -> None:
    """Create a devserver with the given name and arguments."""
    # parameters should be kept in sync with ktd/aws/cloudformation/templates/dev.yaml
    logger.info(f"[{instance_name}] Creating devserver")

    session = get_aws_session(profile)
    states = ["pending", "running", "stopping", "stopped"]
    if get_instances(name=instance_name, states=states, session=session):
        logger.info(f"[{instance_name}] Instance name in use. Aborting.")
        return

    system_files_url = get_system_config()["archive_url"]

    cf_client = session.client("cloudformation")
    cf_client.create_stack(  # type: ignore
        StackName=instance_name,
        TemplateBody=Path(CF_TEMPLATE_PATH).read_text(),
        Capabilities=["CAPABILITY_AUTO_EXPAND", "CAPABILITY_IAM"],
        Parameters=[
            {"ParameterKey": "InstanceType", "ParameterValue": instance_type},
            {"ParameterKey": "SystemFilesUrl", "ParameterValue": system_files_url},
        ],
    )

    wait_for_stack_with_name(instance_name, session=session)

    # the devserver stack creates an instance with the same name
    instances = get_instances(name=instance_name, states=["running"], session=session)
    assert instances is not None and len(instances) == 1
    instance = instances[0]

    update_ssh_config_for_instance(instance)

    logger.info(f"[{instance_name}] Waiting for SSH")
    wait_for_connection(instance_name)

    logger.info(f"[{instance_name}] Waiting for cloud-init to complete")
    wait_for_cloud_init(instance)

    if not no_dotfiles:
        logger.info(f"[{instance_name}] Deploying dotfiles from {dotfiles_repo}")
        deploy_dotfiles(instance_name, dotfiles_repo)

    if open_vscode:
        open_vscode_over_ssh(instance_name)


@app.command()
def start(
    instance_name: str = _instance_name_arg,
    profile: str = _profile_opt,
    open_vscode: bool = _open_vscode_opt,
) -> None:
    """Start instances with the given name."""
    logger.info(f"[{instance_name}] Starting instances")

    session = get_aws_session(profile)
    states = ["stopping", "stopped"]
    instances = get_instances(name=instance_name, states=states, session=session)
    if not instances:
        logger.info(f"[{instance_name}] Instance in {'/'.join(states)} state not found")
        return

    for instance in instances:
        ec2_client = session.client("ec2")
        ec2_client.start_instances(InstanceIds=[instance.id])  # type: ignore
        wait_for_instance_with_id(instance.id, session=session)  # type: ignore

    update_ssh_config_for_instances_with_name(session, instance_name)

    if open_vscode:
        logger.info(f"[{instance_name}] Waiting for SSH")
        wait_for_connection(instance_name)
        open_vscode_over_ssh(instance_name)


@app.command()
def stop(instance_name: str = _instance_name_arg, profile: str = _profile_opt) -> None:
    """Stop instances with the given name."""
    logger.info(f"[{instance_name}] Stopping instances")

    session = get_aws_session(profile)
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


@app.command()
def kill(instance_name: str = _instance_name_arg, profile: str = _profile_opt) -> None:
    """Terminate instances with the given name."""
    logger.info(f"[{instance_name}] Killing instances")
    session = get_aws_session(profile)
    kill_instances_with_name(session, instance_name, kill_stacks=True)


@app.command()
def list(
    profile: str = _profile_opt,
    show_killed: bool = _show_killed_opt,
    compact: bool = _compact_opt,
) -> None:
    """List instances."""
    # FIXME: track stacks w/ specific tag, not instances, since it's
    # possible for, e.g., stack creation to succeed but instance
    # creation to fail
    session = get_aws_session(profile)
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


@app.command()
def refresh(profile: str = _profile_opt) -> None:
    """Refresh hostnames in the SSH config for all running named instances."""
    session = get_aws_session(profile)
    update_ssh_config_for_instances_with_name(session, instance_name="?*")


@app.command()
def open(
    instance_name: str = _instance_name_arg,
    target: VSCodeTarget = _target_opt,
    path: str = _path_opt,
) -> None:
    """Open VS Code on the instance with the given name."""
    open_vscode_over_ssh(instance_name, target, path)
