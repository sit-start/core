from os import chmod, makedirs
from os.path import exists, expanduser

import json5
from sshconf import empty_ssh_config_file, read_ssh_config

from sitstart.cloudpathlib import CloudPath
from sitstart.logging import get_logger
from sitstart.util.run import run

logger = get_logger(__name__)


def _write_ssh_config(conf, path):
    permissions = 0o644 if path == "/etc/ssh/ssh_config" else 0o600
    conf.write(path)
    chmod(path, permissions)


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
    _write_ssh_config(conf, conf_path)
    return True


def update_ssh_config(
    host: str, reset=False, no_overwrite: bool = False, path="~/.ssh/config", **kwargs
) -> None:
    """Updates the SSH config for the given host"""
    logger.info(f"Updating SSH config for host '{host}'")
    assert not (reset and no_overwrite), "Cannot use reset and no_overwrite together"

    conf_path = expanduser(path)
    conf = read_ssh_config(conf_path) if exists(conf_path) else empty_ssh_config_file()

    if host in conf.hosts():
        if reset:
            # TODO: fix issue with spacing in the SSH config file for reset=True
            conf.remove(host)
        elif no_overwrite:
            i = 1
            while (alt_host := f"{host}-{i}") in conf.hosts():
                i += 1
            logger.warning(
                f"Host '{host}' already exists in SSH config, "
                f"using alternative name {repr(alt_host)}",
            )
            conf.add(alt_host, **conf.host(host))
            conf.set(alt_host, **kwargs)
        else:
            conf.set(host, **kwargs)

    if host not in conf.hosts():
        conf.add(host, **kwargs)

    _write_ssh_config(conf, conf_path)


def _get_control_path(make_dir: bool = True) -> str:
    socket_dir = expanduser("~/.ssh/sockets")
    if make_dir:
        makedirs(socket_dir, exist_ok=True)
    return f"{socket_dir}/%r@%n:%p"


def open_ssh_tunnel(
    ssh_addr: str,
    remote_port: int,
    remote_addr: str = "localhost",
    local_port: int | None = None,
    local_addr: str = "localhost",
    local: bool = True,
    quiet: bool = True,
) -> None:
    """Open an SSH tunnel.

    Args:
        ssh_addr: The address of the SSH server, in the form
            '[user@]host'.
        remote_port: The port on the remote server.
        remote_addr: The address of the remote server.
        local_port: The port on the local machine. Defaults to
            `remote_port`.
        local_addr: The address of the local machine.
        local: Whether to open a local or remote tunnel.
        quiet: Whether to suppress output.
    """
    if local_port is None:
        local_port = remote_port
    local_conn_str = f"{local_addr}:{local_port}"
    remote_conn_str = f"{remote_addr}:{remote_port}"
    if local:
        flag_conn_str = ["-L", f"{local_conn_str}:{remote_conn_str}"]
    else:
        flag_conn_str = ["-R", f"{remote_conn_str}:{local_conn_str}"]

    # Note - this should be reentrant
    cmd = (
        [
            "ssh",
            # use the existing connection if it exists
            "-o ControlMaster=auto",
            # path to the control socket
            f"-o ControlPath={_get_control_path()}",
            # terminate the connection if the forwarding fails
            "-o ExitOnForwardFailure=yes",
            # run in background, don't execute a remote command
            "-fN",
        ]
        + flag_conn_str
        + [ssh_addr]
    )
    run(cmd, output="capture" if quiet else "std")


def close_ssh_connection(
    ssh_addr: str, quiet: bool = True, check: bool = False
) -> None:
    """Close an SSH connection.

    Args:
        ssh_addr: The address of the SSH server, in the form
            '[user@]host'.
        quiet: Whether to suppress output.
        check: Whether to raise an exception on failure.
    """
    cmd = [
        "ssh",
        # ControlPath: path to the control socket
        f"-o ControlPath={_get_control_path()}",
        # Control an active connection with the given command
        "-O",
        "exit",
        ssh_addr,
    ]
    run(cmd, output="capture" if quiet else "std", check=check)


def wait_for_connection(ssh_addr: str, max_attempts: int = 15) -> None:
    """Wait for an SSH connection to be established."""
    run(
        [
            "ssh",
            f"-o ConnectionAttempts={max_attempts}",
            "-o BatchMode=yes",
            "-o StrictHostKeyChecking=no",
            ssh_addr,
            "true",
        ],
        check=True,
    )


def get_github_ssh_keys():
    """Fetch the GitHub SSH keys via the GitHub API."""
    path = CloudPath("https://api.github.com/meta")
    meta = json5.loads(path.read_text())
    if not isinstance(meta, dict) or "ssh_keys" not in meta:
        raise RuntimeError("Failed to fetch GitHub SSH keys")
    return meta["ssh_keys"]
