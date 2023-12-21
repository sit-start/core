#!/usr/bin/env python3
"""
Launch and configure an EC2 instance for development.
"""

import logging
import subprocess
import sys
from pathlib import Path

import boto3
from ktd.aws.ec2.util import update_ssh_config
from ktd.aws.util import sso_login

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

# TODO store dev instance config in, e.g., a hydra yaml file
# TODO: handle instance name collisions if it's not done by AWS
_INSTANCE_NAME = "dev-main"
_SESSION_PROFILE = "kevdale-sso"
_IMAGE_ID = "ami-0e186acd30b9cf6a7"
_INSTANCE_TYPE = "g5g.xlarge"
_KEY_NAME = "rsa"
_SECURITY_GROUP_IDS = ["sg-07301ebcb97e0124c"]
_STARTUP_SCRIPT_PATH = f"{Path(__file__).parents[4]}/scripts/ec2_dev_setup.sh"

_PROJECT_PATH = "$HOME/projects"
_PROJECTS = ["study", "infra"]
_PROJECT_REPO_ROOT = "git@github.com:kevdale"


def main() -> int:
    # TODO only attempt login if session creation fails
    logger.info("Logging in to AWS with SSO")
    sso_login(profile_name=_SESSION_PROFILE)
    session = boto3.Session(profile_name=_SESSION_PROFILE)
    ec2 = session.resource("ec2")

    logger.info("Creating EC2 instance")
    instances = ec2.create_instances(
        InstanceType=_INSTANCE_TYPE,
        ImageId=_IMAGE_ID,
        MinCount=1,
        MaxCount=1,
        UserData=Path(_STARTUP_SCRIPT_PATH).read_text(),
        KeyName=_KEY_NAME,
        SecurityGroupIds=_SECURITY_GROUP_IDS,
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": _INSTANCE_NAME}],
            }
        ],
    )
    if len(instances) != 1:
        raise RuntimeError(f"Failed to create EC2 instance ({len(instances)})")
    instance_id = instances[0].id
    logger.info(f"Successfully created instance {instance_id}")

    logger.info("Waiting for initialization to complete")
    ec2_client = session.client("ec2")
    waiter = ec2_client.get_waiter("instance_status_ok")
    waiter.wait(
        InstanceIds=[instance_id],
        WaiterConfig={
            "Delay": 15,
            "MaxAttempts": 20,
        },  # timeout of 15s x 20 = 5 minutes
    )
    # get updated instance info after initialization
    instance = next(i for i in ec2.instances.all() if i.id == instance_id)
    host_name = instance.meta.data["PublicDnsName"]

    logger.info("Updating local SSH config")
    update_ssh_config(_INSTANCE_NAME, HostName=host_name)
    logger.info("Cloning git repositories and dotfiles")

    # yadm and git commands assume identity forwarding is setup in the SSH config
    # TODO: update .bash_profile and .bashrc to for the remote environment
    # subprocess.call(["ssh", _INSTANCE_NAME, f"yadm clone {_REPO_ROOT}/dotfiles"])

    # TODO: for this to work on first connection, I had to do
    # StrictHostChecking no. This might be fine for all aws hosts, but
    # reconsider
    if _PROJECTS:
        _ = subprocess.call(
            ["ssh", _INSTANCE_NAME, "ssh-keyscan -H github.com >> ~/.ssh/known_hosts"],
            stderr=subprocess.DEVNULL,
        )
        clone_cmds = [f"git clone {_PROJECT_REPO_ROOT}/{p}" for p in _PROJECTS]
        cmd = f"mkdir -p {_PROJECT_PATH} && cd $_ && {'; '.join(clone_cmds)}"
        subprocess.call(["ssh", _INSTANCE_NAME, cmd])


if __name__ == "__main__":
    sys.exit(main())
