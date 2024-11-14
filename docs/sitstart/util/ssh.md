# ssh

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [util](./index.md#util) / ssh

> Auto-generated documentation for [sitstart.util.ssh](../../../python/sitstart/util/ssh.py) module.

- [ssh](#ssh)
  - [close_ssh_connection](#close_ssh_connection)
  - [get_github_ssh_keys](#get_github_ssh_keys)
  - [open_ssh_tunnel](#open_ssh_tunnel)
  - [remove_from_ssh_config](#remove_from_ssh_config)
  - [update_ssh_config](#update_ssh_config)
  - [wait_for_connection](#wait_for_connection)

## close_ssh_connection

[Show source in ssh.py:124](../../../python/sitstart/util/ssh.py#L124)

Close an SSH connection.

#### Arguments

- `ssh_addr` - The address of the SSH server, in the form
    '[user@]host'.
- `quiet` - Whether to suppress output.
- `check` - Whether to raise an exception on failure.

#### Signature

```python
def close_ssh_connection(
    ssh_addr: str, quiet: bool = True, check: bool = False
) -> None: ...
```



## get_github_ssh_keys

[Show source in ssh.py:162](../../../python/sitstart/util/ssh.py#L162)

Fetch the GitHub SSH keys via the GitHub API.

#### Signature

```python
def get_github_ssh_keys(): ...
```



## open_ssh_tunnel

[Show source in ssh.py:74](../../../python/sitstart/util/ssh.py#L74)

Open an SSH tunnel.

#### Arguments

- `ssh_addr` - The address of the SSH server, in the form
    '[user@]host'.
- `remote_port` - The port on the remote server.
- `remote_addr` - The address of the remote server.
- `local_port` - The port on the local machine. Defaults to
    `remote_port`.
- `local_addr` - The address of the local machine.
- `local` - Whether to open a local or remote tunnel.
- `quiet` - Whether to suppress output.

#### Signature

```python
def open_ssh_tunnel(
    ssh_addr: str,
    remote_port: int,
    remote_addr: str = "localhost",
    local_port: int | None = None,
    local_addr: str = "localhost",
    local: bool = True,
    quiet: bool = True,
) -> None: ...
```



## remove_from_ssh_config

[Show source in ssh.py:20](../../../python/sitstart/util/ssh.py#L20)

Removes the given host from the SSH config at the given path

#### Signature

```python
def remove_from_ssh_config(host: str, path="~/.ssh/config") -> bool: ...
```



## update_ssh_config

[Show source in ssh.py:34](../../../python/sitstart/util/ssh.py#L34)

Updates the SSH config for the given host

#### Signature

```python
def update_ssh_config(
    host: str, reset=False, no_overwrite: bool = False, path="~/.ssh/config", **kwargs
) -> None: ...
```



## wait_for_connection

[Show source in ssh.py:147](../../../python/sitstart/util/ssh.py#L147)

Wait for an SSH connection to be established.

#### Signature

```python
def wait_for_connection(ssh_addr: str, max_attempts: int = 15) -> None: ...
```
