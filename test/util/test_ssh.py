import shlex
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from sitstart.util.ssh import (
    _get_control_path,
    close_ssh_connection,
    get_github_ssh_keys,
    open_ssh_tunnel,
    remove_from_ssh_config,
    update_ssh_config,
    wait_for_connection,
)


def test_remove_from_ssh_config():
    with tempfile.TemporaryDirectory() as temp_dir:
        assert not remove_from_ssh_config("host", path=f"{temp_dir}/nonexistent_file")

    with tempfile.NamedTemporaryFile() as conf_file:
        conf_file.close()

        update_ssh_config(
            "host", path=conf_file.name, User="user", StrictHostKeyChecking="no"
        )
        remove_from_ssh_config("host", path=conf_file.name)
        assert Path(conf_file.name).read_text().strip() == ""

        assert not remove_from_ssh_config("nonexistent_host", path=conf_file.name)


def test_update_ssh_config():
    with tempfile.NamedTemporaryFile() as conf_file:
        conf_file.close()
        update_ssh_config(
            "host", path=conf_file.name, User="user", StrictHostKeyChecking="no"
        )
        assert Path(conf_file.name).read_text().strip() == "\n".join(
            [
                "Host host",
                "  User user",
                "  StrictHostKeyChecking no",
            ]
        )

        update_ssh_config(
            "host", path=conf_file.name, User="user", AddKeysToAgent="yes"
        )
        assert Path(conf_file.name).read_text().strip() == "\n".join(
            [
                "Host host",
                "  User user",
                "  StrictHostKeyChecking no",
                "  AddKeysToAgent yes",
            ]
        )

        update_ssh_config(
            "host", path=conf_file.name, reset=True, User="user", AddKeysToAgent="yes"
        )
        assert Path(conf_file.name).read_text().strip() == "\n".join(
            [
                "Host host",
                "  User user",
                "  AddKeysToAgent yes",
            ]
        )

        update_ssh_config("host", path=conf_file.name, no_overwrite=True, User="user-1")
        update_ssh_config("host", path=conf_file.name, no_overwrite=True, User="user-2")
        assert Path(conf_file.name).read_text().strip() == "\n".join(
            [
                "Host host",
                "  User user",
                "  AddKeysToAgent yes",
                "",
                "Host host-1",
                "  User user-1",
                "  AddKeysToAgent yes",
                "",
                "Host host-2",
                "  User user-2",
                "  AddKeysToAgent yes",
            ]
        )


@mock.patch("sitstart.util.ssh.run")
def test_open_ssh_tunnel(run_mock):
    control_path = _get_control_path(make_dir=False)
    ssh_addr = "dest"
    remote_port = 80
    local_port = 8080

    open_ssh_tunnel(ssh_addr, remote_port, local_port=local_port)
    run_mock.assert_called_once_with(
        shlex.split(
            f"ssh '-o ControlMaster=auto' '-o ControlPath={control_path}' "
            f"'-o ExitOnForwardFailure=yes' -fN -L localhost:{local_port}:"
            f"localhost:{remote_port} {ssh_addr}"
        ),
        output="capture",
    )


@mock.patch("sitstart.util.ssh.run")
def test_close_ssh_connection(run_mock):
    control_path = _get_control_path(make_dir=False)
    dest = "dest"

    close_ssh_connection(dest)
    run_mock.assert_called_once_with(
        shlex.split(f"ssh '-o ControlPath={control_path}' -O exit {dest}"),
        output="capture",
        check=False,
    )


@mock.patch("sitstart.util.ssh.run")
def test_wait_for_connection(run_mock):
    wait_for_connection("dest", max_attempts=10)
    run_mock.assert_called_once_with(
        shlex.split(
            "ssh '-o ConnectionAttempts=10' '-o BatchMode=yes' '-o "
            "StrictHostKeyChecking=no' dest true"
        ),
        check=True,
    )


@mock.patch("sitstart.util.ssh.CloudPath")
def test_get_github_ssh_keys(cloud_path_mock):
    cloud_path_instance = cloud_path_mock.return_value
    cloud_path_instance.read_text.return_value = '{"ssh_keys": ["key1", "key2"]}'
    assert get_github_ssh_keys() == ["key1", "key2"]

    cloud_path_instance.read_text.return_value = "{}"
    with pytest.raises(RuntimeError):
        get_github_ssh_keys()
