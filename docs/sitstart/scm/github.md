# github

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [scm](./index.md#scm) / github

> Auto-generated documentation for [sitstart.scm.github](../../../python/sitstart/scm/github.py) module.

- [github](#github)
  - [create_private_fork](#create_private_fork)
  - [get_ssh_url](#get_ssh_url)
  - [get_user](#get_user)

## create_private_fork

[Show source in github.py:26](../../../python/sitstart/scm/github.py#L26)

Create a private fork of a repository.

#### Arguments

- `repo_url` - The URL of the repository to fork.
- `fork_name` - Rename the forked repository.
- `clone` - Clone the forked repository.
- `org` - Create the fork in an organization.

#### Signature

```python
def create_private_fork(
    repo_url: str,
    fork_name: str | None = None,
    clone: bool = False,
    org: str | None = None,
) -> None: ...
```



## get_ssh_url

[Show source in github.py:22](../../../python/sitstart/scm/github.py#L22)

#### Signature

```python
def get_ssh_url(account: str, repo: str): ...
```



## get_user

[Show source in github.py:18](../../../python/sitstart/scm/github.py#L18)

#### Signature

```python
def get_user(): ...
```
