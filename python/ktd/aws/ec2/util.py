from os.path import exists, expanduser
from typing import Optional, Type

import boto3
from boto3.resources.base import ServiceResource
from sshconf import empty_ssh_config_file, read_ssh_config

session = boto3.Session(profile_name="kevdale-sso")
ec2 = session.resource("ec2")


def get_instance_name(instance: Type[ServiceResource]) -> Optional[str]:
    tags = instance.meta.data.get("Tags", {})
    return next((el["Value"] for el in tags if el["Key"] == "Name"), None)


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
