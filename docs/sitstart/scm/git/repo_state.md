# RepoState

[Core Index](../../../README.md#core-index) / [sitstart](../../index.md#sitstart) / [scm](../index.md#scm) / [git](./index.md#git) / RepoState

> Auto-generated documentation for [sitstart.scm.git.repo_state](../../../../python/sitstart/scm/git/repo_state.py) module.

- [RepoState](#repostate)
  - [RepoState](#repostate-1)
    - [RepoState.from_dict](#repostatefrom_dict)
    - [RepoState.from_repo](#repostatefrom_repo)
    - [RepoState().replay](#repostate()replay)
    - [RepoState().summary](#repostate()summary)

## RepoState

[Show source in repo_state.py:15](../../../../python/sitstart/scm/git/repo_state.py#L15)

#### Signature

```python
class RepoState: ...
```

### RepoState.from_dict

[Show source in repo_state.py:52](../../../../python/sitstart/scm/git/repo_state.py#L52)

#### Signature

```python
@classmethod
def from_dict(cls, d: dict) -> "RepoState": ...
```

### RepoState.from_repo

[Show source in repo_state.py:25](../../../../python/sitstart/scm/git/repo_state.py#L25)

#### Signature

```python
@classmethod
def from_repo(
    cls, repo: str | Repo, remote: str | Remote = "origin", branch: str = "main"
) -> "RepoState": ...
```

### RepoState().replay

[Show source in repo_state.py:67](../../../../python/sitstart/scm/git/repo_state.py#L67)

#### Signature

```python
def replay(
    self, repo: str | Repo, replay_branch_name: str = "repo-state-replay"
) -> None: ...
```

### RepoState().summary

[Show source in repo_state.py:56](../../../../python/sitstart/scm/git/repo_state.py#L56)

#### Signature

```python
@property
def summary(self) -> str: ...
```
