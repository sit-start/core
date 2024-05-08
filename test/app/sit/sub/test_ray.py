import string
import subprocess
import tempfile
import time
from pathlib import Path

import git
import pytest
import yaml
from ray.job_submission import JobStatus, JobSubmissionClient

from sitstart.app.sit.sub import ec2, ray
from sitstart.logging import get_logger
from sitstart.util.run import run
from sitstart.util.string import rand_str

RAY_CONFIG_NAME = f"test-{ray.DEFAULT_CONFIG}-" + rand_str(
    8, string.digits + string.ascii_letters
)
TEST_SCRIPT = "image_multiclass_smoketest"
SSH_CONFIG = """CanonicalizeHostname yes
Host *.compute.amazonaws.com
    User ec2-user
    IdentityFile ~/.ssh/rsa.pem
    StrictHostKeyChecking no
"""

logger = get_logger(__name__)


def _wait_for_job_status(
    client: JobSubmissionClient,
    sub_id: str,
    target_status: JobStatus,
    timeout_sec=60,
) -> JobStatus:
    start_time_sec = time.time()
    while (
        client.get_job_status(sub_id) != target_status
        and time.time() - start_time_sec < timeout_sec
    ):
        time.sleep(1)
    return client.get_job_status(sub_id)


@pytest.fixture(scope="module")
def ray_config(is_local_test):
    with tempfile.TemporaryDirectory() as temp_dir:
        original_config_path = ray._resolve_config_path(ray.DEFAULT_CONFIG)
        config = yaml.load(
            Path(original_config_path).read_text(), Loader=yaml.SafeLoader
        )
        config["cluster_name"] = RAY_CONFIG_NAME

        config_path = f"{temp_dir}/{RAY_CONFIG_NAME}.yaml"

        if not is_local_test:
            # `ray submit` requires that this repo is in the config's
            # file mounts, but, since we can't control the absolute path
            # of the repo in GH Actions, or how Ray rsyncs symlinked
            # directories, we need to update file mounts in the config.
            # For consistency, we do it here for all tests.

            # sanity check that file mounts correspond to the current
            # values of $DEV and $CORE.
            repo = git.Repo(__file__, search_parent_directories=True)
            assert any(v.startswith("~/dev") for v in config["file_mounts"])
            assert Path(repo.working_dir).stem == "core"

            for mount in config["file_mounts"].copy():
                if mount.startswith("~/dev"):
                    config["file_mounts"].pop(mount)
            config["file_mounts"]["~/dev/core"] = repo.working_dir

            Path(config_path).write_text(yaml.dump(config))
        else:
            Path(config_path).write_text(Path(original_config_path).read_text())

        yield config_path


@pytest.fixture(scope="module")
def ray_cluster(ssh_config, ray_config, aws_session, is_local_test):
    try:
        logger.info(f"Starting Ray cluster with config {ray_config!r}.")
        ray.up(config=ray_config, show_output=True, no_port_forwarding=is_local_test)
    except Exception as e:
        logger.error(f"Failed to start Ray cluster: {e}")
        ec2.kill(f"*{RAY_CONFIG_NAME}*")
        raise e
    yield
    logger.info(f"Cleaning up Ray cluster for config {ray_config!r}.")
    ec2.kill(f"*{RAY_CONFIG_NAME}*")


@pytest.fixture(scope="module")
def run_on_head(ray_config):
    def impl(cmd: str, **kwargs) -> subprocess.CompletedProcess[bytes]:
        return run(["ray", "exec", ray_config, cmd], output="capture", **kwargs)

    return impl


@pytest.mark.integration
@pytest.mark.slow
def test_up(ray_cluster, run_on_head):
    assert run_on_head("ray status").returncode == 0


@pytest.mark.integration
@pytest.mark.slow
def test_submit(ray_cluster, ray_config):
    client = ray._job_submission_client()

    sub_id = ray.submit(TEST_SCRIPT, config=ray_config)
    status = _wait_for_job_status(client, sub_id, JobStatus.RUNNING)
    assert status == JobStatus.RUNNING

    status = _wait_for_job_status(client, sub_id, JobStatus.SUCCEEDED, timeout_sec=600)
    assert status == JobStatus.SUCCEEDED


@pytest.mark.integration
@pytest.mark.slow
def test_stop_jobs(ray_cluster, ray_config):
    client = ray._job_submission_client()

    sub_id = ray.submit(TEST_SCRIPT, config=ray_config)
    status = _wait_for_job_status(client, sub_id, JobStatus.RUNNING)
    assert status == JobStatus.RUNNING

    ray.stop_jobs()
    status = _wait_for_job_status(client, sub_id, JobStatus.STOPPED)
    assert status == JobStatus.STOPPED


@pytest.mark.integration
@pytest.mark.slow
def test_down(ray_cluster, ray_config, run_on_head):
    ray.down(config=ray_config, show_output=True)
    with pytest.raises(subprocess.CalledProcessError):
        run_on_head("echo OK")
