from os.path import exists, expanduser
from typing import Type

import boto3
from boto3.resources.base import ServiceResource
from sshconf import empty_ssh_config_file, read_ssh_config


def get_instance_name(instance: Type[ServiceResource]) -> str | None:
    if instance.meta is None:
        return None
    tags = instance.meta.data.get("Tags", {})
    return next((el["Value"] for el in tags if el["Key"] == "Name"), None)


def get_instances_with_name(
    name: str,
    session: boto3.Session | None = None,
    states: list[str] | None = None,
) -> list[Type[ServiceResource]]:
    session = session or boto3.Session()
    ec2 = session.resource("ec2")
    filters = [{"Name": "tag:Name", "Values": [name]}]
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


def update_ssh_config(host: str, reset=False, path="~/.ssh/config", **kwargs) -> None:
    """Updates the SSH config for the given host"""
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
