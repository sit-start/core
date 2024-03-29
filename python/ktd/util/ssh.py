from os import chmod, makedirs
from os.path import exists, expanduser

import json5
from sshconf import empty_ssh_config_file, read_ssh_config

from ktd.cloudpathlib import CloudPath
from ktd.logging import get_logger
from ktd.util.run import run

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


def _get_control_path() -> str:
    socket_dir = expanduser("~/.ssh/sockets")
    makedirs(socket_dir, exist_ok=True)
    return f"{socket_dir}/%r@%n:%p"


def open_ssh_tunnel(
    dest: str,
    port: int,
    bind_address: str | None = None,
    host: str = "localhost",
    host_port: int | None = None,  # defaults to `port`
    local: bool = True,
    quiet: bool = True,
) -> None:
    if host_port is None:
        host_port = port
    connection_str = f"{port}:{host}:{host_port}"
    if bind_address:
        connection_str = f"{bind_address}:{connection_str}"

    # Note - this should be reentrant
    cmd = [
        "ssh",
        # ControlMaster=auto: use the existing connection if it exists
        "-o ControlMaster=auto",
        # ControlPath: path to the control socket
        f"-o ControlPath={_get_control_path()}",
        # ExitOnForwardFailure=yes: terminate the connection if the forwarding fails
        "-o ExitOnForwardFailure=yes",
        # -f: run in background
        # -N: don't execute a remote command
        "-fN",
        # -L: local port forwarding
        # -R: remote port forwarding
        "-L" if local else "-R",
        connection_str,
        dest,
    ]
    run(cmd, output="capture" if quiet else "std")


def close_ssh_connection(dest: str, quiet: bool = True, check: bool = False) -> None:
    cmd = [
        "ssh",
        # ControlPath: path to the control socket
        f"-o ControlPath={_get_control_path()}",
        # Control an active connection with the given command
        "-O",
        "exit",
        dest,
    ]
    run(cmd, output="capture" if quiet else "std", check=check)


def wait_for_connection(dest: str, max_attempts: int = 15) -> None:
    run(
        [
            "ssh",
            "-o",
            f"ConnectionAttempts {max_attempts}",
            "-o",
            "BatchMode yes",
            "-o",
            "StrictHostKeyChecking no",
            dest,
            "true",
        ],
        check=True,
    )


def get_github_ssh_keys():
    path = CloudPath("https://api.github.com/meta")
    meta = json5.loads(path.read_text())
    if not isinstance(meta, dict) or "ssh_keys" not in meta:
        raise RuntimeError("Failed to fetch GitHub SSH keys")
    return meta["ssh_keys"]
