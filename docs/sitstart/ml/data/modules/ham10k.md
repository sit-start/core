# HAM10k

[Core Index](../../../../README.md#core-index) / [sitstart](../../../index.md#sitstart) / [ml](../../index.md#ml) / [data](../index.md#data) / [modules](./index.md#modules) / HAM10k

> Auto-generated documentation for [sitstart.ml.data.modules.ham10k](../../../../../python/sitstart/ml/data/modules/ham10k.py) module.

- [HAM10k](#ham10k)
  - [HAM10k](#ham10k-1)
    - [HAM10k().class_count](#ham10k()class_count)
    - [HAM10k().criteria_weight](#ham10k()criteria_weight)
    - [HAM10k().get_sampler](#ham10k()get_sampler)
    - [HAM10k().has_sampler](#ham10k()has_sampler)

## HAM10k

[Show source in ham10k.py:16](../../../../../python/sitstart/ml/data/modules/ham10k.py#L16)

#### Signature

```python
class HAM10k(VisionDataModule):
    def __init__(
        self,
        criteria_gamma: float = 0.0,
        dedupe: bool = False,
        rebalance_gamma: float = 0.0,
        *args,
        **kwargs
    ): ...
```

#### See also

- [VisionDataModule](./vision_data_module.md#visiondatamodule)

### HAM10k().class_count

[Show source in ham10k.py:56](../../../../../python/sitstart/ml/data/modules/ham10k.py#L56)

#### Signature

```python
@property
def class_count(self) -> torch.Tensor: ...
```

### HAM10k().criteria_weight

[Show source in ham10k.py:50](../../../../../python/sitstart/ml/data/modules/ham10k.py#L50)

#### Signature

```python
@VisionDataModule.criteria_weight.getter
def criteria_weight(self) -> torch.Tensor | torch.nn.Module | None: ...
```

### HAM10k().get_sampler

[Show source in ham10k.py:30](../../../../../python/sitstart/ml/data/modules/ham10k.py#L30)

#### Signature

```python
@memoize
def get_sampler(self) -> Sampler | None: ...
```

#### See also

- [memoize](../../../util/decorators.md#memoize)

### HAM10k().has_sampler

[Show source in ham10k.py:46](../../../../../python/sitstart/ml/data/modules/ham10k.py#L46)

#### Signature

```python
@property
def has_sampler(self) -> bool: ...
```
