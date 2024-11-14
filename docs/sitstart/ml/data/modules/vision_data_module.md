# VisionDataModule

[Core Index](../../../../README.md#core-index) / [sitstart](../../../index.md#sitstart) / [ml](../../index.md#ml) / [data](../index.md#data) / [modules](./index.md#modules) / VisionDataModule

> Auto-generated documentation for [sitstart.ml.data.modules.vision_data_module](../../../../../python/sitstart/ml/data/modules/vision_data_module.py) module.

- [VisionDataModule](#visiondatamodule)
  - [VisionDataModule](#visiondatamodule-1)
    - [VisionDataModule().criteria_weight](#visiondatamodule()criteria_weight)
    - [VisionDataModule().get_sampler](#visiondatamodule()get_sampler)
    - [VisionDataModule().has_sampler](#visiondatamodule()has_sampler)
    - [VisionDataModule().prepare_data](#visiondatamodule()prepare_data)
    - [VisionDataModule().setup](#visiondatamodule()setup)
    - [VisionDataModule().test_as_val](#visiondatamodule()test_as_val)
    - [VisionDataModule().test_dataloader](#visiondatamodule()test_dataloader)
    - [VisionDataModule().test_dataset](#visiondatamodule()test_dataset)
    - [VisionDataModule().train_dataloader](#visiondatamodule()train_dataloader)
    - [VisionDataModule().train_dataset](#visiondatamodule()train_dataset)
    - [VisionDataModule().train_split](#visiondatamodule()train_split)
    - [VisionDataModule().train_split_size](#visiondatamodule()train_split_size)
    - [VisionDataModule().val_dataloader](#visiondatamodule()val_dataloader)
    - [VisionDataModule().val_split](#visiondatamodule()val_split)

## VisionDataModule

[Show source in vision_data_module.py:20](../../../../../python/sitstart/ml/data/modules/vision_data_module.py#L20)

#### Signature

```python
class VisionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_class: type,
        batch_size=128,
        data_dir: str | os.PathLike[str] | None = DEFAULT_DATASET_ROOT,
        train_dataset_size: float | int | None = None,
        train_split_size: float | int = 0.8,
        augment: Callable | None = None,
        collate: Callable | None = None,
        transform: Callable | None = None,
        n_workers: int = 8,
        shuffle: bool = True,
        sampler: Sampler | None = None,
        seed: int = 42,
        test_as_val: bool = False,
    ): ...
```

#### See also

- [DEFAULT_DATASET_ROOT](../index.md#default_dataset_root)

### VisionDataModule().criteria_weight

[Show source in vision_data_module.py:163](../../../../../python/sitstart/ml/data/modules/vision_data_module.py#L163)

Weight for the loss function, if applicable.

#### Signature

```python
@property
def criteria_weight(self) -> torch.Tensor | torch.nn.Module | None: ...
```

### VisionDataModule().get_sampler

[Show source in vision_data_module.py:109](../../../../../python/sitstart/ml/data/modules/vision_data_module.py#L109)

#### Signature

```python
@memoize
def get_sampler(self) -> Sampler | None: ...
```

#### See also

- [memoize](../../../util/decorators.md#memoize)

### VisionDataModule().has_sampler

[Show source in vision_data_module.py:113](../../../../../python/sitstart/ml/data/modules/vision_data_module.py#L113)

#### Signature

```python
@property
def has_sampler(self) -> bool: ...
```

### VisionDataModule().prepare_data

[Show source in vision_data_module.py:117](../../../../../python/sitstart/ml/data/modules/vision_data_module.py#L117)

#### Signature

```python
def prepare_data(self) -> None: ...
```

### VisionDataModule().setup

[Show source in vision_data_module.py:129](../../../../../python/sitstart/ml/data/modules/vision_data_module.py#L129)

#### Signature

```python
def setup(self, stage: str | None = None): ...
```

### VisionDataModule().test_as_val

[Show source in vision_data_module.py:101](../../../../../python/sitstart/ml/data/modules/vision_data_module.py#L101)

#### Signature

```python
@property
def test_as_val(self) -> bool: ...
```

### VisionDataModule().test_dataloader

[Show source in vision_data_module.py:155](../../../../../python/sitstart/ml/data/modules/vision_data_module.py#L155)

#### Signature

```python
def test_dataloader(self) -> DataLoader | None: ...
```

### VisionDataModule().test_dataset

[Show source in vision_data_module.py:89](../../../../../python/sitstart/ml/data/modules/vision_data_module.py#L89)

#### Signature

```python
@property
def test_dataset(self) -> datasets.VisionDataset | None: ...
```

### VisionDataModule().train_dataloader

[Show source in vision_data_module.py:132](../../../../../python/sitstart/ml/data/modules/vision_data_module.py#L132)

#### Signature

```python
def train_dataloader(self) -> DataLoader: ...
```

### VisionDataModule().train_dataset

[Show source in vision_data_module.py:84](../../../../../python/sitstart/ml/data/modules/vision_data_module.py#L84)

#### Signature

```python
@property
@memoize
def train_dataset(self) -> datasets.VisionDataset: ...
```

#### See also

- [memoize](../../../util/decorators.md#memoize)

### VisionDataModule().train_split

[Show source in vision_data_module.py:93](../../../../../python/sitstart/ml/data/modules/vision_data_module.py#L93)

#### Signature

```python
@property
def train_split(self) -> Subset: ...
```

### VisionDataModule().train_split_size

[Show source in vision_data_module.py:105](../../../../../python/sitstart/ml/data/modules/vision_data_module.py#L105)

#### Signature

```python
@property
def train_split_size(self) -> float | int: ...
```

### VisionDataModule().val_dataloader

[Show source in vision_data_module.py:149](../../../../../python/sitstart/ml/data/modules/vision_data_module.py#L149)

#### Signature

```python
def val_dataloader(self) -> DataLoader: ...
```

### VisionDataModule().val_split

[Show source in vision_data_module.py:97](../../../../../python/sitstart/ml/data/modules/vision_data_module.py#L97)

#### Signature

```python
@property
def val_split(self) -> Subset: ...
```
