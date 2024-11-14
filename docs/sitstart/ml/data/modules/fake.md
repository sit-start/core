# fake

[Core Index](../../../../README.md#core-index) / [sitstart](../../../index.md#sitstart) / [ml](../../index.md#ml) / [data](../index.md#data) / [modules](./index.md#modules) / fake

> Auto-generated documentation for [sitstart.ml.data.modules.fake](../../../../../python/sitstart/ml/data/modules/fake.py) module.

- [fake](#fake)
  - [Fake2d](#fake2d)
    - [Fake2d().prepare_data](#fake2d()prepare_data)

## Fake2d

[Show source in fake.py:9](../../../../../python/sitstart/ml/data/modules/fake.py#L9)

#### Signature

```python
class Fake2d(VisionDataModule):
    def __init__(
        self,
        batch_size: int = 10,
        train_split_size: float = 0.8,
        augment: Callable | None = None,
        transform: Callable | None = None,
        num_train: int = 40,
        num_test: int = 10,
        num_classes: int = 10,
        img_shape: tuple[int, int, int] = (3, 32, 32),
    ): ...
```

#### See also

- [VisionDataModule](./vision_data_module.md#visiondatamodule)

### Fake2d().prepare_data

[Show source in fake.py:35](../../../../../python/sitstart/ml/data/modules/fake.py#L35)

#### Signature

```python
def prepare_data(self) -> None: ...
```
