# TrainingModule

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [ml](./index.md#ml) / TrainingModule

> Auto-generated documentation for [sitstart.ml.training_module](../../../python/sitstart/ml/training_module.py) module.

- [TrainingModule](#trainingmodule)
  - [TrainingModule](#trainingmodule-1)
    - [TrainingModule().configure_optimizers](#trainingmodule()configure_optimizers)
    - [TrainingModule().forward](#trainingmodule()forward)
    - [TrainingModule().on_test_epoch_end](#trainingmodule()on_test_epoch_end)
    - [TrainingModule().on_train_epoch_end](#trainingmodule()on_train_epoch_end)
    - [TrainingModule().on_validation_epoch_end](#trainingmodule()on_validation_epoch_end)
    - [TrainingModule().test_step](#trainingmodule()test_step)
    - [TrainingModule().to](#trainingmodule()to)
    - [TrainingModule().training_step](#trainingmodule()training_step)
    - [TrainingModule().validation_step](#trainingmodule()validation_step)

## TrainingModule

[Show source in training_module.py:19](../../../python/sitstart/ml/training_module.py#L19)

#### Signature

```python
class TrainingModule(pl.LightningModule):
    def __init__(
        self,
        loss_fn: nn.Module,
        lr_scheduler: LRScheduler | Callable[[Optimizer], LRScheduler] | None,
        train_metrics: dict[str, Metric | TorchMetric] | None,
        test_metrics: dict[str, Metric | TorchMetric] | None,
        model: nn.Module,
        optimizer: Optimizer | Callable[[Any], Optimizer],
        train_batch_transform: BatchTransform | None = None,
    ) -> None: ...
```

### TrainingModule().configure_optimizers

[Show source in training_module.py:92](../../../python/sitstart/ml/training_module.py#L92)

#### Signature

```python
def configure_optimizers(self) -> OptimizerLRSchedulerConfig: ...
```

### TrainingModule().forward

[Show source in training_module.py:59](../../../python/sitstart/ml/training_module.py#L59)

#### Signature

```python
def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

### TrainingModule().on_test_epoch_end

[Show source in training_module.py:89](../../../python/sitstart/ml/training_module.py#L89)

#### Signature

```python
def on_test_epoch_end(self) -> None: ...
```

### TrainingModule().on_train_epoch_end

[Show source in training_module.py:86](../../../python/sitstart/ml/training_module.py#L86)

#### Signature

```python
def on_train_epoch_end(self) -> None: ...
```

### TrainingModule().on_validation_epoch_end

[Show source in training_module.py:83](../../../python/sitstart/ml/training_module.py#L83)

#### Signature

```python
def on_validation_epoch_end(self) -> None: ...
```

### TrainingModule().test_step

[Show source in training_module.py:80](../../../python/sitstart/ml/training_module.py#L80)

#### Signature

```python
def test_step(self, batch: Any, batch_idx: int) -> None: ...
```

### TrainingModule().to

[Show source in training_module.py:102](../../../python/sitstart/ml/training_module.py#L102)

#### Signature

```python
def to(self, *args, **kwargs) -> "TrainingModule": ...
```

### TrainingModule().training_step

[Show source in training_module.py:62](../../../python/sitstart/ml/training_module.py#L62)

#### Signature

```python
def training_step(self, batch: Any, _: int) -> torch.Tensor: ...
```

### TrainingModule().validation_step

[Show source in training_module.py:77](../../../python/sitstart/ml/training_module.py#L77)

#### Signature

```python
def validation_step(self, batch: Any, batch_idx: int) -> None: ...
```
