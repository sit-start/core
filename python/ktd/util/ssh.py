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
