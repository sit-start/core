# train

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [ml](./index.md#ml) / train

> Auto-generated documentation for [sitstart.ml.train](../../../python/sitstart/ml/train.py) module.

- [train](#train)
  - [Trainer](#trainer)
    - [Trainer().fit](#trainer()fit)
  - [test](#test)
  - [train](#train-1)

## Trainer

[Show source in train.py:22](../../../python/sitstart/ml/train.py#L22)

#### Signature

```python
class Trainer(pl.Trainer): ...
```

### Trainer().fit

[Show source in train.py:23](../../../python/sitstart/ml/train.py#L23)

#### Signature

```python
def fit(
    self,
    model: pl.LightningModule,
    train_dataloaders: Any = None,
    val_dataloaders: Any = None,
    datamodule: pl.LightningDataModule | None = None,
    ckpt_path: Path | str | None = None,
) -> None: ...
```



## test

[Show source in train.py:165](../../../python/sitstart/ml/train.py#L165)

Test a model.

#### Arguments

- `data_module` - PyTorch Lightning data module, whose
    `test_dataloader` will be used.
- `training_module` - PyTorch Lightning training module.
- `checkpoint_path` - Path to a checkpoint from which model weights
    are loaded.
- `storage_path` - Path to save results.
- `accelerator` - Accelerator to use.

#### Signature

```python
def test(
    data_module: pl.LightningDataModule,
    training_module: pl.LightningModule,
    checkpoint_path: str | os.PathLike[str] | None = None,
    storage_path: str | os.PathLike[str] | None = None,
    accelerator: str | Accelerator = "auto",
) -> list[Mapping[str, float]]: ...
```



## train

[Show source in train.py:62](../../../python/sitstart/ml/train.py#L62)

Train a PyTorch Lightning model.

#### Arguments

- `data_module` - PyTorch Lightning data module.
- `training_module` - PyTorch Lightning training module.
- `ckpt_path` - Path to a local checkpoint from which to resume training.
- `float32_matmul_precision` - Precision for matrix multiplication.
- `logging_interval` - Logging interval in batches.
- `max_num_epochs` - Maximum number of epochs.
- `num_sanity_val_steps` - Number of sanity validation steps. See pl.Trainer.
- `project_name` - Name of the project.
- `seed` - Random seed.
- `storage_path` - Path to save results. Must be a local path if
    _with_ray=False.
- `use_gpu` - Whether to use the GPU.
- `wandb_enabled` - Whether to enable Weights & Biases logging.
- `with_ray` - Whether train() is invoked from a Ray training or
    tuning run.

#### Signature

```python
def train(
    data_module: pl.LightningDataModule,
    training_module: pl.LightningModule,
    ckpt_path: str | os.PathLike[str] | None = None,
    float32_matmul_precision: str = "default",
    gradient_clip_val: float | None = None,
    gradient_clip_algorithm: str | None = None,
    logging_interval: int = 100,
    max_num_epochs: int = 100,
    num_sanity_val_steps: int | None = None,
    project_name: str | None = None,
    seed: int | None = None,
    storage_path: os.PathLike[str] | None = None,
    use_gpu: bool = False,
    wandb_enabled: bool = False,
    with_ray: bool = False,
) -> None: ...
```
