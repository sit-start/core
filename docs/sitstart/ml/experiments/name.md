# name

[Core Index](../../../README.md#core-index) / [sitstart](../../index.md#sitstart) / [ml](../index.md#ml) / [experiments](./index.md#experiments) / name

> Auto-generated documentation for [sitstart.ml.experiments.name](../../../../python/sitstart/ml/experiments/name.py) module.

- [name](#name)
  - [get_group_name](#get_group_name)
  - [get_group_name_from_run_name](#get_group_name_from_run_name)
  - [get_project_name](#get_project_name)
  - [get_run_name](#get_run_name)
  - [get_trial_dirname](#get_trial_dirname)
  - [get_trial_name](#get_trial_name)

## get_group_name

[Show source in name.py:15](../../../../python/sitstart/ml/experiments/name.py#L15)

#### Signature

```python
def get_group_name() -> str: ...
```



## get_group_name_from_run_name

[Show source in name.py:24](../../../../python/sitstart/ml/experiments/name.py#L24)

#### Signature

```python
def get_group_name_from_run_name(project_name: str, run_name: str) -> str: ...
```



## get_project_name

[Show source in name.py:11](../../../../python/sitstart/ml/experiments/name.py#L11)

#### Signature

```python
def get_project_name(config: DictConfig) -> str: ...
```



## get_run_name

[Show source in name.py:19](../../../../python/sitstart/ml/experiments/name.py#L19)

#### Signature

```python
def get_run_name(project_name: str, group_name: str | None = None) -> str: ...
```



## get_trial_dirname

[Show source in name.py:28](../../../../python/sitstart/ml/experiments/name.py#L28)

#### Signature

```python
def get_trial_dirname(trial: Trial) -> str: ...
```



## get_trial_name

[Show source in name.py:32](../../../../python/sitstart/ml/experiments/name.py#L32)

#### Signature

```python
def get_trial_name(trial: Trial, incl_params: bool = False) -> str: ...
```
