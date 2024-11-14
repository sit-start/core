# system

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [util](./index.md#util) / system

> Auto-generated documentation for [sitstart.util.system](../../../python/sitstart/util/system.py) module.

- [system](#system)
  - [deploy_dotfiles](#deploy_dotfiles)
  - [deploy_system_files](#deploy_system_files)
  - [deploy_system_files_from_filesystem](#deploy_system_files_from_filesystem)
  - [get_system_config](#get_system_config)
  - [hash_system_files](#hash_system_files)
  - [push_system_files](#push_system_files)
  - [system_file_archive_url](#system_file_archive_url)

## deploy_dotfiles

[Show source in system.py:181](../../../python/sitstart/util/system.py#L181)

#### Signature

```python
def deploy_dotfiles(host: str, repo_url: str | None = None) -> None: ...
```



## deploy_system_files

[Show source in system.py:159](../../../python/sitstart/util/system.py#L159)

Deploy system files from the S3 archive to the local filesystem.

For final deployment, use `dest_dir = "/", as_root = True`.

#### Signature

```python
def deploy_system_files(dest_dir: str, as_root: bool = False) -> None: ...
```



## deploy_system_files_from_filesystem

[Show source in system.py:150](../../../python/sitstart/util/system.py#L150)

Deploy system files to the local filesystem as symlinks. For development.

#### Signature

```python
def deploy_system_files_from_filesystem(
    dest_dir: str, src_dir: str = SYSTEM_FILE_ROOT, as_root: bool = False
) -> None: ...
```

#### See also

- [SYSTEM_FILE_ROOT](#system_file_root)



## get_system_config

[Show source in system.py:114](../../../python/sitstart/util/system.py#L114)

#### Signature

```python
def get_system_config() -> dict: ...
```



## hash_system_files

[Show source in system.py:120](../../../python/sitstart/util/system.py#L120)

#### Signature

```python
def hash_system_files() -> str: ...
```



## push_system_files

[Show source in system.py:128](../../../python/sitstart/util/system.py#L128)

Push repo system files to S3.

#### Signature

```python
def push_system_files() -> None: ...
```



## system_file_archive_url

[Show source in system.py:124](../../../python/sitstart/util/system.py#L124)

#### Signature

```python
def system_file_archive_url() -> str: ...
```
