import subprocess
import sys
from os import makedirs
from os.path import exists, expanduser

from ktd.logging import get_logger
from sshconf import empty_ssh_config_file, read_ssh_config

logger = get_logger(__name__)


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
    conf.write(conf_path)
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

    conf.write(conf_path)


def _get_control_path() -> str:
    socket_dir = expanduser("~/.ssh/sockets")
    makedirs(socket_dir, exist_ok=True)
    return f"{socket_dir}/%r@%n:%p"


def _run_cmd(cmd: list[str], quiet: bool = True, check: bool = True) -> None:
    if not quiet:
        subprocess.run(cmd, check=check)
        return

    stdout = stderr = subprocess.PIPE
    try:
        subprocess.run(cmd, stdout=stdout, stderr=stderr, check=check)
    except subprocess.CalledProcessError as e:
        logger.error(
            "{exception}\nstdout:\n{stdout}\nstderr:\n{stderr}".format(
                exception=e,
                stdout=e.stdout.decode(sys.stdout.encoding),
                stderr=e.stderr.decode(sys.stderr.encoding),
            )
        )
        raise e


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
    _run_cmd(cmd, quiet)


def close_ssh_connection(dest: str, quiet: bool = True, noexcept: bool = True) -> None:
    cmd = [
        "ssh",
        # ControlPath: path to the control socket
        f"-o ControlPath={_get_control_path()}",
        # Control an active connection with the given command
        "-O",
        "exit",
        dest,
    ]
    _run_cmd(cmd, quiet, check=False)
