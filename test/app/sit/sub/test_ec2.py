import string
import subprocess
from typing import Any

import pytest

from sitstart.app.sit.sub import ec2
from sitstart.aws.ec2.util import (
    INSTANCE_STATES,
    get_instance_name,
    get_instances,
    wait_for_instance_with_id,
)
from sitstart.logging import get_logger
from sitstart.util.run import run
from sitstart.util.ssh import remove_from_ssh_config
from sitstart.util.string import rand_str

INSTANCE_NAME = "test-dev-" + rand_str(8, string.digits + string.ascii_letters)
INSTANCE_TYPE = "m6a.xlarge"
TERMINAL_STATES = ["shutting-down", "terminated"]
NON_TERMINAL_STATES = list(set(INSTANCE_STATES) - set(TERMINAL_STATES))

logger = get_logger(__name__)


def _get_unique_instance_with_name(
    name: str, states: list[str] | None = NON_TERMINAL_STATES, require=True
) -> Any:
    instances = get_instances(name=name, states=states)
    assert not require or len(instances) == 1
    return instances[0] if instances else None


def _get_instance_name(instance: Any) -> str:
    instance_name = get_instance_name(instance)
    assert instance_name is not None
    return instance_name


def _get_state_for_instance_with_name(
    instance_name: str, states: list[str] | None = NON_TERMINAL_STATES
) -> str:
    instance = _get_unique_instance_with_name(instance_name, states)
    assert hasattr(instance, "state")
    instance_state = instance.state["Name"]
    assert instance_state in INSTANCE_STATES
    return instance_state


@pytest.fixture(scope="module")
def running_instance(ssh_config):
    if not (instance := _get_unique_instance_with_name(INSTANCE_NAME, require=False)):
        logger.info(f"Creating instance {INSTANCE_NAME}")
        ec2.create(INSTANCE_NAME, instance_type=INSTANCE_TYPE, no_dotfiles=True)
        instance = _get_unique_instance_with_name(INSTANCE_NAME)
        assert instance is not None

    instance_name = _get_instance_name(instance)
    if _get_state_for_instance_with_name(instance_name) != "running":
        ec2.start(instance_name)

    yield instance

    logger.info(f"Killing instance {INSTANCE_NAME}")
    ec2.kill(instance_name)
    wait_for_instance_with_id(instance.id, wait_on="instance_terminated")
    state = _get_state_for_instance_with_name(instance_name, states=TERMINAL_STATES)
    assert state == "terminated"


@pytest.mark.integration
@pytest.mark.slow
def test_create(running_instance):
    assert running_instance


@pytest.mark.integration
@pytest.mark.slow
def test_list(running_instance, caplog):
    ec2.list()
    expected_strings = ["running", _get_instance_name(running_instance)]
    assert any(all(s in msg for s in expected_strings) for msg in caplog.messages)


@pytest.mark.integration
@pytest.mark.slow
def test_refresh(running_instance):
    instance_name = _get_instance_name(running_instance)
    remove_from_ssh_config(instance_name)
    with pytest.raises(subprocess.CalledProcessError):
        run(["ssh", instance_name, "echo"])

    ec2.refresh()
    assert run(["ssh", instance_name, "echo"], output="capture").returncode == 0


@pytest.mark.integration
@pytest.mark.slow
def test_stop(running_instance):
    instance_name = _get_instance_name(running_instance)
    assert _get_state_for_instance_with_name(instance_name) == "running"

    ec2.stop(instance_name)
    wait_for_instance_with_id(running_instance.id, wait_on="instance_stopped")
    assert _get_state_for_instance_with_name(instance_name) == "stopped"


@pytest.mark.integration
@pytest.mark.slow
def test_start(running_instance):
    instance_name = _get_instance_name(running_instance)
    ec2.stop(instance_name)
    wait_for_instance_with_id(running_instance.id, wait_on="instance_stopped")
    assert _get_state_for_instance_with_name(instance_name) == "stopped"

    ec2.start(instance_name)
    wait_for_instance_with_id(running_instance.id, wait_on="instance_running")
    assert _get_state_for_instance_with_name(instance_name) == "running"
