from os.path import exists, expanduser

import boto3
from boto3.resources.base import ServiceResource
from ktd.logging import get_logger
from sshconf import empty_ssh_config_file, read_ssh_config

logger = get_logger(__name__)


def get_instance_name(instance: ServiceResource) -> str | None:
    if instance.meta is None:
        return None
    tags = instance.meta.data.get("Tags", {})
    return next((el["Value"] for el in tags if el["Key"] == "Name"), None)


def get_instances(
    name: str | None = None,
    session: boto3.Session | None = None,
    states: list[str] | None = None,
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
    return list(ec2.instances.filter(Filters=filters))


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


def remove_from_ssh_config(host: str, path="~/.ssh/config") -> bool:
    """Removes the given host from the SSH config at the given path"""
    conf_path = expanduser(path)
    if not exists(conf_path):
        return False
    conf = read_ssh_config(conf_path)
    if host not in conf.hosts():
        return False
    logger.info(f"Removing host '{host}' from SSH config")
    conf.remove(host)
    conf.write(conf_path)
    return True


def update_ssh_config(host: str, reset=False, path="~/.ssh/config", **kwargs) -> None:
    """Updates the SSH config for the given host"""
    logger.info(f"Updating SSH config for host '{host}'")
    conf_path = expanduser(path)
    conf = read_ssh_config(conf_path) if exists(conf_path) else empty_ssh_config_file()

    if host in conf.hosts():
        if reset:
            # TODO: fix issue with spacing in the SSH config file for reset=True
            conf.remove(host)
        else:
            conf.set(host, **kwargs)

    if host not in conf.hosts():
        conf.add(host, **kwargs)

    conf.write(conf_path)
