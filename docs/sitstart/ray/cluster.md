# cluster

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [ray](./index.md#ray) / cluster

> Auto-generated documentation for [sitstart.ray.cluster](../../../python/sitstart/ray/cluster.py) module.

- [cluster](#cluster)
  - [cluster_down](#cluster_down)
  - [cluster_up](#cluster_up)
  - [get_file_mounts](#get_file_mounts)
  - [get_job_runtime_env](#get_job_runtime_env)
  - [get_job_submission_client](#get_job_submission_client)
  - [list_jobs](#list_jobs)
  - [stop_all_jobs](#stop_all_jobs)
  - [stop_job](#stop_job)
  - [submit_job](#submit_job)
  - [sync_dotfiles](#sync_dotfiles)
  - [wait_for_job_status](#wait_for_job_status)

## cluster_down

[Show source in cluster.py:282](../../../python/sitstart/ray/cluster.py#L282)

#### Signature

```python
def cluster_down(
    config_path: Path,
    cluster_name: str,
    workers_only: bool = False,
    keep_min_workers: bool = False,
    prompt: bool = False,
    verbose: bool = False,
    show_output: bool = False,
) -> None: ...
```



## cluster_up

[Show source in cluster.py:235](../../../python/sitstart/ray/cluster.py#L235)

#### Signature

```python
def cluster_up(
    config_path: Path,
    cluster_name: str,
    min_workers: int | None = None,
    max_workers: int | None = None,
    no_restart: bool = False,
    restart_only: bool = False,
    prompt: bool = False,
    verbose: bool = False,
    show_output: bool = False,
    no_config_cache: bool = False,
    do_sync_dotfiles: bool = False,
) -> None: ...
```



## get_file_mounts

[Show source in cluster.py:90](../../../python/sitstart/ray/cluster.py#L90)

#### Signature

```python
def get_file_mounts(config_path: Path, user_root: str = "/home") -> dict[Path, Path]: ...
```



## get_job_runtime_env

[Show source in cluster.py:38](../../../python/sitstart/ray/cluster.py#L38)

#### Signature

```python
def get_job_runtime_env(clone_venv: bool = True) -> dict[str, Any]: ...
```



## get_job_submission_client

[Show source in cluster.py:32](../../../python/sitstart/ray/cluster.py#L32)

#### Signature

```python
def get_job_submission_client(
    dashboard_port: int = DASHBOARD_PORT,
) -> JobSubmissionClient: ...
```

#### See also

- [DASHBOARD_PORT](#dashboard_port)



## list_jobs

[Show source in cluster.py:123](../../../python/sitstart/ray/cluster.py#L123)

#### Signature

```python
def list_jobs(dashboard_port: int = DASHBOARD_PORT) -> None: ...
```

#### See also

- [DASHBOARD_PORT](#dashboard_port)



## stop_all_jobs

[Show source in cluster.py:116](../../../python/sitstart/ray/cluster.py#L116)

#### Signature

```python
def stop_all_jobs(
    delete: bool = False, dashboard_port: int = DASHBOARD_PORT
) -> None: ...
```

#### See also

- [DASHBOARD_PORT](#dashboard_port)



## stop_job

[Show source in cluster.py:101](../../../python/sitstart/ray/cluster.py#L101)

#### Signature

```python
def stop_job(
    sub_id: str, delete: bool = False, dashboard_port: int = DASHBOARD_PORT
) -> None: ...
```

#### See also

- [DASHBOARD_PORT](#dashboard_port)



## submit_job

[Show source in cluster.py:139](../../../python/sitstart/ray/cluster.py#L139)

Submit a job to the Ray cluster.

#### Arguments

- `script_path` - The path to the script to run. Must be in [WORKING_DIR](#cluster).
- `dashboard_port` - The port for the Ray dashboard.
- `description` - A description of the job, shown in[list_jobs](#list_jobs).
- `clone_venv` - Whether to clone the current virtual environment.
- `config_path` - The path to the script's Hydra config.

#### Signature

```python
def submit_job(
    script_path: Path,
    dashboard_port: int = DASHBOARD_PORT,
    description: str | None = None,
    clone_venv: bool = True,
    config_path: Path | None = None,
) -> str: ...
```

#### See also

- [DASHBOARD_PORT](#dashboard_port)



## sync_dotfiles

[Show source in cluster.py:203](../../../python/sitstart/ray/cluster.py#L203)

#### Signature

```python
def sync_dotfiles(config_path: Path, cluster_name: str) -> None: ...
```



## wait_for_job_status

[Show source in cluster.py:46](../../../python/sitstart/ray/cluster.py#L46)

#### Signature

```python
def wait_for_job_status(
    client: JobSubmissionClient, sub_id: str, target_status: JobStatus, timeout_sec=60
) -> JobStatus: ...
```
