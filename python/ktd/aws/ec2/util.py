import boto3
from boto3.resources.base import ServiceResource
from ktd.logging import get_logger

logger = get_logger(__name__)


def get_instance_name(instance: ServiceResource) -> str | None:
    if instance.meta is None:
        return None
    tags = instance.meta.data.get("Tags", {})
    return next((el["Value"] for el in tags if el["Key"] == "Name"), None)


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
