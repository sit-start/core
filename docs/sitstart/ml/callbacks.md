# callbacks

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [ml](./index.md#ml) / callbacks

> Auto-generated documentation for [sitstart.ml.callbacks](../../../python/sitstart/ml/callbacks.py) module.

- [callbacks](#callbacks)
  - [LoggerCallback](#loggercallback)
    - [LoggerCallback().on_train_epoch_end](#loggercallback()on_train_epoch_end)

## LoggerCallback

[Show source in callbacks.py:7](../../../python/sitstart/ml/callbacks.py#L7)

#### Signature

```python
class LoggerCallback(Callback):
    def __init__(self, logger: logging.Logger): ...
```

### LoggerCallback().on_train_epoch_end

[Show source in callbacks.py:11](../../../python/sitstart/ml/callbacks.py#L11)

#### Signature

```python
def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None: ...
```
