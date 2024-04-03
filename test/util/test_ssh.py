import shlex
import tempfile
from pathlib import Path
from unittest import mock

from ktd.util.ssh import (
    _get_control_path,
    close_ssh_connection,
    get_github_ssh_keys,
    open_ssh_tunnel,
    remove_from_ssh_config,
    update_ssh_config,
    wait_for_connection,
)


def test_remove_from_ssh_config():
    with tempfile.NamedTemporaryFile() as conf_file:
        conf_file.close()
        update_ssh_config(
            "host", path=conf_file.name, User="user", StrictHostKeyChecking="no"
        )
        remove_from_ssh_config("host", path=conf_file.name)
        assert Path(conf_file.name).read_text().strip() == ""


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

        update_ssh_config(
            "host",
            path=conf_file.name,
            no_overwrite=True,
            User="user",
            AddKeysToAgent="no",
        )
        assert Path(conf_file.name).read_text().strip() == "\n".join(
            [
                "Host host",
                "  User user",
                "  AddKeysToAgent yes",
                "",
                "Host host-1",
                "  User user",
                "  AddKeysToAgent no",
            ]
        )


@mock.patch("ktd.util.ssh.run")
def test_open_ssh_tunnel(run_mock):
    control_path = _get_control_path(make_dir=False)
    dest = "dest"
    port = 8000

    open_ssh_tunnel(dest, port)
    run_mock.assert_called_once_with(
        shlex.split(
            f"ssh '-o ControlMaster=auto' '-o ControlPath={control_path}' "
            f"'-o ExitOnForwardFailure=yes' -fN -L {port}:localhost:{port} {dest}"
        ),
        output="capture",
    )


@mock.patch("ktd.util.ssh.run")
def test_close_ssh_connection(run_mock):
    control_path = _get_control_path(make_dir=False)
    dest = "dest"

    close_ssh_connection(dest)
    run_mock.assert_called_once_with(
        shlex.split(f"ssh '-o ControlPath={control_path}' -O exit {dest}"),
        output="capture",
        check=False,
    )


@mock.patch("ktd.util.ssh.run")
def test_wait_for_connection(run_mock):
    wait_for_connection("dest", max_attempts=10)
    run_mock.assert_called_once_with(
        shlex.split(
            "ssh '-o ConnectionAttempts=10' '-o BatchMode=yes' '-o "
            "StrictHostKeyChecking=no' dest true"
        ),
        check=True,
    )


@mock.patch("ktd.util.ssh.CloudPath")
def test_get_github_ssh_keys(cloud_path_mock):
    cloud_path_instance = cloud_path_mock.return_value
    cloud_path_instance.read_text.return_value = '{"ssh_keys": ["key1", "key2"]}'
    assert get_github_ssh_keys() == ["key1", "key2"]
