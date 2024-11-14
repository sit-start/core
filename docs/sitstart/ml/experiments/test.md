# test

[Core Index](../../../README.md#core-index) / [sitstart](../../index.md#sitstart) / [ml](../index.md#ml) / [experiments](./index.md#experiments) / test

> Auto-generated documentation for [sitstart.ml.experiments.test](../../../../python/sitstart/ml/experiments/test.py) module.

- [test](#test)
  - [test_checkpoint](#test_checkpoint)
  - [test_trial](#test_trial)

## test_checkpoint

[Show source in test.py:65](../../../../python/sitstart/ml/experiments/test.py#L65)

Test the given checkpoint.

#### Arguments

- `checkpoint` - The checkpoint to test. If the checkpoint was
    created at a remote storage path, the remote `Checkpoint` should be
    provided here.
- `config_overrides` - Additional overrides to apply to the
    experiment config. Useful when minor code changes are incompatible
    with the checkpoint config.
- `test_storage_path` - The root path for storing results, which are saved
to `{test_storage_path}/{config.name}/{trial_id}`.
- `trial_id` - The trial ID, used for storing results.

#### Signature

```python
def test_checkpoint(
    checkpoint: ray.train.Checkpoint,
    config_overrides: list[str] | None = None,
    test_storage_path: str | os.PathLike[str] = TEST_ROOT,
    trial_id: str = "unspecified",
) -> Any: ...
```

#### See also

- [TEST_ROOT](./index.md#test_root)



## test_trial

[Show source in test.py:24](../../../../python/sitstart/ml/experiments/test.py#L24)

Test the trial with the given project name and trial ID.

Loads the checkpoint for the project + trial and tests with
[test_checkpoint](#test_checkpoint). `select_*` arguments are supplied to `get_checkpoint`.

Defaults to testing the last checkpoint.

#### Signature

```python
def test_trial(
    project_name: str,
    trial_id: str,
    config_overrides: list[str] | None = None,
    storage_path: str | os.PathLike[str] = TEST_ROOT,
    select: Literal["best", "last"] = "last",
    select_metric: str | None = None,
    select_mode: str | None = None,
    is_archived: bool = False,
) -> Any: ...
```

#### See also

- [TEST_ROOT](./index.md#test_root)
