# lr_schedulers

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [ml](./index.md#ml) / lr_schedulers

> Auto-generated documentation for [sitstart.ml.lr_schedulers](../../../python/sitstart/ml/lr_schedulers.py) module.

- [lr_schedulers](#lr_schedulers)
  - [LinearWarmup](#linearwarmup)
  - [LinearWarmupCosineAnnealingWarmRestarts](#linearwarmupcosineannealingwarmrestarts)
  - [SequentialLR](#sequentiallr)

## LinearWarmup

[Show source in lr_schedulers.py:28](../../../python/sitstart/ml/lr_schedulers.py#L28)

#### Signature

```python
class LinearWarmup(_SequentialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        scheduler: partial[LRScheduler],
        last_epoch: int = -1,
        warmup_iters: int = 5,
        warmup_factor: float = 1.0 / 3,
    ): ...
```



## LinearWarmupCosineAnnealingWarmRestarts

[Show source in lr_schedulers.py:50](../../../python/sitstart/ml/lr_schedulers.py#L50)

#### Signature

```python
class LinearWarmupCosineAnnealingWarmRestarts(_SequentialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        last_epoch: int = -1,
        warmup_iters: int = 5,
        warmup_factor: float = 1.0 / 3,
        *args: Any,
        **kwargs: Any
    ): ...
```



## SequentialLR

[Show source in lr_schedulers.py:11](../../../python/sitstart/ml/lr_schedulers.py#L11)

#### Signature

```python
class SequentialLR(_SequentialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        schedulers: list[partial[LRScheduler]],
        milestones: list[int],
        last_epoch: int = -1,
    ): ...
```
