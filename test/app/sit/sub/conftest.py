import os
from pathlib import Path

import pytest

from sitstart.logging import get_logger
from sitstart.aws.util import get_aws_session

SSH_CONFIG = """CanonicalizeHostname yes
Host *.compute.amazonaws.com
    User ec2-user
    IdentityFile ~/.ssh/rsa.pem
    StrictHostKeyChecking no
"""

logger = get_logger(__name__)


@pytest.fixture(scope="module")
def is_local_test():
    return not os.getenv("GITHUB_ACTIONS", False)


@pytest.fixture(scope="module")
def ssh_config(is_local_test):
    ssh_config_path = Path("~/.ssh/config").expanduser()
    use_existing_config = is_local_test and ssh_config_path.exists()

    if use_existing_config:
        logger.info(
            "Skipping SSH config setup for local test and using the existing config "
            f"'{ssh_config_path}'."
        )
    else:
        assert not ssh_config_path.exists()
        logger.info(f"Creating SSH config '{ssh_config_path}'.")
        ssh_config_path.parent.mkdir(mode=0o700, exist_ok=True)
        ssh_config_path.write_text(SSH_CONFIG)
        ssh_config_path.chmod(0o600)

    yield

    if not use_existing_config:
        logger.info(f"Removing SSH config '{ssh_config_path}'.")
        ssh_config_path.unlink()


@pytest.fixture(scope="module")
def aws_session():
    # ensure we have an active session for local testing; uses the default profile
    return get_aws_session(profile=None)
