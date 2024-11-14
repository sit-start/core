# util

[Core Index](../../../README.md#core-index) / [sitstart](../../index.md#sitstart) / [scm](../index.md#scm) / [git](./index.md#git) / util

> Auto-generated documentation for [sitstart.scm.git.util](../../../../python/sitstart/scm/git/util.py) module.

- [util](#util)
  - [create_tag_with_type](#create_tag_with_type)
  - [diff_vs_commit](#diff_vs_commit)
  - [fetch_tags](#fetch_tags)
  - [get_first_remote_ancestor](#get_first_remote_ancestor)
  - [get_remote_branches_for_commit](#get_remote_branches_for_commit)
  - [get_repo](#get_repo)
  - [get_staged_files](#get_staged_files)
  - [get_tags](#get_tags)
  - [is_commit_in_remote](#is_commit_in_remote)
  - [is_pristine](#is_pristine)
  - [is_synced](#is_synced)
  - [list_tracked_dotfiles](#list_tracked_dotfiles)
  - [sync_tags](#sync_tags)
  - [update_to_ref](#update_to_ref)

## create_tag_with_type

[Show source in util.py:87](../../../../python/sitstart/scm/git/util.py#L87)

#### Signature

```python
def create_tag_with_type(
    repo: str | Repo,
    tag_type: StringIdType,
    message: str | None = None,
    remote: str | Remote | None = None,
) -> TagReference: ...
```

#### See also

- [StringIdType](../../util/identifier.md#stringidtype)



## diff_vs_commit

[Show source in util.py:130](../../../../python/sitstart/scm/git/util.py#L130)

Returns the diff between the working dir and the given ref

The result is as if the staging area had been unstaged prior to
calling `diff`, but previously staged files, regardless of whether
or not they're untracked, are included in the diff.

#### Signature

```python
def diff_vs_commit(
    repo: str | Repo,
    ref: str = "HEAD",
    include_staged_untracked: bool = True,
    include_untracked: bool = False,
    *args,
    **kwargs
) -> str: ...
```



## fetch_tags

[Show source in util.py:28](../../../../python/sitstart/scm/git/util.py#L28)

#### Signature

```python
def fetch_tags(
    repo: str | Repo,
    remote: str | Remote | None = None,
    prune: bool = False,
    force: bool = False,
) -> None: ...
```



## get_first_remote_ancestor

[Show source in util.py:55](../../../../python/sitstart/scm/git/util.py#L55)

#### Signature

```python
def get_first_remote_ancestor(
    repo: str | Repo,
    commit: Commit | str = "HEAD",
    remote: Remote | str = "origin",
    branch: str = "main",
) -> Commit: ...
```



## get_remote_branches_for_commit

[Show source in util.py:114](../../../../python/sitstart/scm/git/util.py#L114)

#### Signature

```python
def get_remote_branches_for_commit(
    repo: str | Repo, commit: Commit | str
) -> list[str]: ...
```



## get_repo

[Show source in util.py:19](../../../../python/sitstart/scm/git/util.py#L19)

#### Signature

```python
def get_repo(path: str | None = None) -> Repo: ...
```



## get_staged_files

[Show source in util.py:125](../../../../python/sitstart/scm/git/util.py#L125)

#### Signature

```python
def get_staged_files(repo: str | Repo) -> list[str]: ...
```



## get_tags

[Show source in util.py:73](../../../../python/sitstart/scm/git/util.py#L73)

#### Signature

```python
def get_tags(
    repo: str | Repo, commit: Commit | None = None, tag_type: StringIdType | None = None
) -> list[TagReference]: ...
```



## is_commit_in_remote

[Show source in util.py:121](../../../../python/sitstart/scm/git/util.py#L121)

#### Signature

```python
def is_commit_in_remote(repo: str | Repo, commit: Commit | str) -> bool: ...
```



## is_pristine

[Show source in util.py:160](../../../../python/sitstart/scm/git/util.py#L160)

#### Signature

```python
def is_pristine(repo: str | Repo) -> bool: ...
```



## is_synced

[Show source in util.py:45](../../../../python/sitstart/scm/git/util.py#L45)

Returns True iff the local branch is synced with the remote branch.

#### Signature

```python
def is_synced(repo: str | Repo, branch: str | None = None) -> bool: ...
```



## list_tracked_dotfiles

[Show source in util.py:170](../../../../python/sitstart/scm/git/util.py#L170)

#### Signature

```python
def list_tracked_dotfiles() -> list[str]: ...
```



## sync_tags

[Show source in util.py:24](../../../../python/sitstart/scm/git/util.py#L24)

#### Signature

```python
def sync_tags(repo: str | Repo, remote: str | Remote | None = None) -> None: ...
```



## update_to_ref

[Show source in util.py:105](../../../../python/sitstart/scm/git/util.py#L105)

#### Signature

```python
def update_to_ref(repo: str | Repo, ref: str) -> None: ...
```
