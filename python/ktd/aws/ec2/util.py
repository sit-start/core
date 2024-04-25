import os
import shlex
from os.path import expanduser, expandvars

import boto3
from boto3.resources.base import ServiceResource

from ktd.logging import get_logger
from ktd.util.run import run, Output
from ktd.util.ssh import remove_from_ssh_config, update_ssh_config

INSTANCE_STATES = [
    "pending",
    "running",
    "stopping",
    "stopped",
    "shutting-down",
    "terminated",
]

logger = get_logger(__name__)


def get_instance_name(instance: ServiceResource) -> str | None:
    if instance.meta is None:
        return None
    tags = instance.meta.data.get("Tags", {})
    return next((el["Value"] for el in tags if el["Key"] == "Name"), None)


def get_unique_instance_name(instance: ServiceResource) -> str:
    """Returns a name for the given EC2 instance unique among running instances."""
    name = get_instance_name(instance)
    id = instance.id  # type: ignore
    if name is None:
        logger.warning(f"Instance has no name; using instance ID {id}")
        return id

    instances_with_name = get_instances(name, states=["pending", "running"])
    if len(instances_with_name) == 1:
        return name

    return f"{name}-{id}"


def get_instances(
    name: str | None = None,
    ids: list[str] = [],
    states: list[str] | None = None,
    session: boto3.Session | None = None,
) -> list[ServiceResource]:
    """Returns a list of EC2 instances with the given name and state(s)

    Supports the same wildcards as the AWS CLI, e.g. "web*" or "web-1?".
    """
    session = session or boto3.Session()
    ec2 = session.resource("ec2")
    filters = []
    if name is not None:
        filters.append({"Name": "tag:Name", "Values": [name]})
    if states is not None:
        filters.append({"Name": "instance-state-name", "Values": states})
    return list(ec2.instances.filter(InstanceIds=ids, Filters=filters))


def wait_for_instance_with_id(
    instance_id: str,
    session: boto3.Session | None = None,
    delay_sec: int = 15,
    max_attempts: int = 20,
) -> None:
    session = session or boto3.Session()
    ec2_client = session.client("ec2")
    waiter = ec2_client.get_waiter("instance_status_ok")
    waiter.wait(
        InstanceIds=[instance_id],
        WaiterConfig={
            "Delay": delay_sec,
            "MaxAttempts": max_attempts,
        },  # timeout of 15s x 20 = 5 minutes w/ default values
    )


def wait_for_cloud_init(instance: ServiceResource):
    instance_id = instance.id  # type: ignore
    assert instance.meta is not None, f"Instance {instance_id} has no metadata"
    host_name = instance.meta.data["PublicDnsName"]
    cmd = shlex.split(f"ssh {host_name} sudo cloud-init status --wait")
    output = run(cmd, output=Output.CAPTURE)
    status = output.stdout.decode("utf-8").strip(".").strip()
    logger.info(f"Cloud-init {status}")


def wait_for_stack_with_name(
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


def kill_instances_with_name(
    session: boto3.Session,
    instance_name: str,
    states: list[str] | None = None,
    update_ssh_config: bool = True,
    kill_stacks: bool = False,
) -> None:
    states = states or list(set(INSTANCE_STATES).difference(["terminated"]))
    instances = get_instances(name=instance_name, states=states, session=session)
    kill_instances(
        session, instances, update_ssh_config=update_ssh_config, kill_stacks=kill_stacks
    )


def kill_instances(
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
            remove_from_ssh_config(get_unique_instance_name(instance))


def update_ssh_config_for_instance(
    instance: ServiceResource, ssh_config_path: str | os.PathLike = "$HOME/.ssh/config"
) -> None:
    instance_name = get_unique_instance_name(instance)
    assert instance.meta is not None, f"[{instance_name}] Instance has no metadata"
    host_name = instance.meta.data["PublicDnsName"]
    ssh_config_path = expanduser(expandvars(ssh_config_path))
    if not host_name:
        logger.info("No hostname to update in SSH config")
        return
    update_ssh_config(instance_name, HostName=host_name, path=ssh_config_path)


def update_ssh_config_for_instances_with_name(
    session: boto3.Session, instance_name: str = "?*"
) -> None:
    """Update the SSH config for all running instances with the given name"""
    logger.info(f"[{instance_name}] Updating hostnames in SSH config")

    running_instances: list[ServiceResource] = []
    other_instance_names: list[str] = []

    for instance in get_instances(name=instance_name, session=session):
        this_instance_name = get_unique_instance_name(instance)
        if instance.state["Name"].strip() == "running":  # type: ignore
            running_instances.append(instance)
        else:
            other_instance_names.append(this_instance_name)

    for this_instance_name in set(other_instance_names):
        remove_from_ssh_config(this_instance_name)
    for instance in running_instances:
        update_ssh_config_for_instance(instance)
