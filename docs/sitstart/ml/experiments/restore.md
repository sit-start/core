# restore

[Core Index](../../../README.md#core-index) / [sitstart](../../index.md#sitstart) / [ml](../index.md#ml) / [experiments](./index.md#experiments) / restore

> Auto-generated documentation for [sitstart.ml.experiments.restore](../../../../python/sitstart/ml/experiments/restore.py) module.

- [restore](#restore)
  - [_get_run_group_from_trial](#_get_run_group_from_trial)
  - [get_checkpoint](#get_checkpoint)
  - [get_checkpoint_file_path](#get_checkpoint_file_path)
  - [get_checkpoint_from_config](#get_checkpoint_from_config)
  - [get_cloud_path](#get_cloud_path)
  - [get_experiment_state](#get_experiment_state)
  - [get_trial_params](#get_trial_params)
  - [get_trial_path_from_archived_trial](#get_trial_path_from_archived_trial)
  - [get_trial_path_from_trial](#get_trial_path_from_trial)
  - [to_local_checkpoint](#to_local_checkpoint)

## _get_run_group_from_trial

[Show source in restore.py:28](../../../../python/sitstart/ml/experiments/restore.py#L28)

Returns the run group for the given project and trial.

#### Signature

```python
def _get_run_group_from_trial(
    storage_path: str, project_name: str, trial_id: str
) -> str | None: ...
```



## get_checkpoint

[Show source in restore.py:147](../../../../python/sitstart/ml/experiments/restore.py#L147)

Get the checkpoint for the given trial.

Remote checkpoints not in the local cache are downloaded.
`storage_path` defaults to `get_storage_path()`. If `run_group`
isn't specified, the first run group in the storage path containing
the given trial ID is used.

Defaults to the last checkpoint.

#### Signature

```python
def get_checkpoint(
    project_name: str,
    trial_id: str,
    storage_path: str | None = None,
    run_group: str | None = None,
    select: Literal["best", "last"] = "last",
    select_metric: str | None = None,
    select_mode: str | None = None,
    to_local: bool = False,
    is_archived: bool = False,
) -> Checkpoint | None: ...
```



## get_checkpoint_file_path

[Show source in restore.py:228](../../../../python/sitstart/ml/experiments/restore.py#L228)

#### Signature

```python
def get_checkpoint_file_path(checkpoint: Checkpoint) -> str: ...
```



## get_checkpoint_from_config

[Show source in restore.py:205](../../../../python/sitstart/ml/experiments/restore.py#L205)

Get the checkpoint for the given config.

Remote checkpoints not in the local cache are downloaded.

#### Signature

```python
def get_checkpoint_from_config(
    config: DictConfig, to_local: bool = False
) -> Checkpoint | None: ...
```



## get_cloud_path

[Show source in restore.py:232](../../../../python/sitstart/ml/experiments/restore.py#L232)

Returns a `CloudPath` object for the given path string

#### Signature

```python
def get_cloud_path(
    path: str, local_cache_dir: str | None = RUN_ROOT, aws_profile: str | None = None
) -> CloudPath | Path: ...
```

#### See also

- [RUN_ROOT](./index.md#run_root)



## get_experiment_state

[Show source in restore.py:245](../../../../python/sitstart/ml/experiments/restore.py#L245)

Get the repository state and config from the given checkpoint.

If `checkpoint` was created at a remote storage path, the remote
`Checkpoint` should be provided here, not, e.g., the output of
`to_local_checkpoint()`.

#### Signature

```python
def get_experiment_state(
    checkpoint: Checkpoint,
) -> tuple[RepoState | None, DictConfig | None]: ...
```



## get_trial_params

[Show source in restore.py:124](../../../../python/sitstart/ml/experiments/restore.py#L124)

#### Signature

```python
def get_trial_params(
    project_name: str,
    trial_id: str,
    storage_path: str | None = None,
    run_group: str | None = None,
) -> dict[str, Any]: ...
```



## get_trial_path_from_archived_trial

[Show source in restore.py:98](../../../../python/sitstart/ml/experiments/restore.py#L98)

#### Signature

```python
def get_trial_path_from_archived_trial(
    project_name: str, trial_id: str
) -> str | None: ...
```



## get_trial_path_from_trial

[Show source in restore.py:80](../../../../python/sitstart/ml/experiments/restore.py#L80)

#### Signature

```python
def get_trial_path_from_trial(
    project_name: str, storage_path: str, trial_id: str, run_group: str | None = None
) -> str | None: ...
```



## to_local_checkpoint

[Show source in restore.py:107](../../../../python/sitstart/ml/experiments/restore.py#L107)

Converts a remote checkpoint to a local checkpoint.

Unlike `ray.train.Checkpoint.to_directory()`, this uses a local
cache.

#### Signature

```python
def to_local_checkpoint(checkpoint: Checkpoint) -> Checkpoint: ...
```
