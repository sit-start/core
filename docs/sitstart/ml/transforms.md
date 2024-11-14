# transforms

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [ml](./index.md#ml) / transforms

> Auto-generated documentation for [sitstart.ml.transforms](../../../python/sitstart/ml/transforms.py) module.

- [transforms](#transforms)
  - [AdjustContrast](#adjustcontrast)
    - [AdjustContrast().forward](#adjustcontrast()forward)
  - [BatchTransform](#batchtransform)
    - [BatchTransform().forward](#batchtransform()forward)
  - [CutMixBatchTransform](#cutmixbatchtransform)
  - [CutMixCollateTransform](#cutmixcollatetransform)
  - [CutMixUp](#cutmixup)
    - [CutMixUp().forward](#cutmixup()forward)
  - [CutMixUpBatchTransform](#cutmixupbatchtransform)
  - [CutMixUpCollateTransform](#cutmixupcollatetransform)
  - [DefaultCollateTransform](#defaultcollatetransform)
    - [DefaultCollateTransform().forward](#defaultcollatetransform()forward)
  - [IdentityBatchTransform](#identitybatchtransform)
  - [ImageToFloat32](#imagetofloat32)
    - [ImageToFloat32().forward](#imagetofloat32()forward)
  - [MixUpBatchTransform](#mixupbatchtransform)
  - [MixUpCollateTransform](#mixupcollatetransform)

## AdjustContrast

[Show source in transforms.py:11](../../../python/sitstart/ml/transforms.py#L11)

#### Signature

```python
class AdjustContrast(torch.nn.Module):
    def __init__(self, factor: float): ...
```

### AdjustContrast().forward

[Show source in transforms.py:16](../../../python/sitstart/ml/transforms.py#L16)

#### Signature

```python
def forward(self, img: torch.Tensor) -> torch.Tensor: ...
```



## BatchTransform

[Show source in transforms.py:42](../../../python/sitstart/ml/transforms.py#L42)

#### Signature

```python
class BatchTransform(torch.nn.Module):
    def __init__(
        self,
        transform: Callable,
        requires_shuffle: bool = False,
        train_only: bool = False,
    ): ...
```

### BatchTransform().forward

[Show source in transforms.py:54](../../../python/sitstart/ml/transforms.py#L54)

#### Signature

```python
def forward(self, batch: Any) -> Any: ...
```



## CutMixBatchTransform

[Show source in transforms.py:63](../../../python/sitstart/ml/transforms.py#L63)

#### Signature

```python
class CutMixBatchTransform(BatchTransform):
    def __init__(self, **kwargs: Any): ...
```

#### See also

- [BatchTransform](#batchtransform)



## CutMixCollateTransform

[Show source in transforms.py:86](../../../python/sitstart/ml/transforms.py#L86)

#### Signature

```python
class CutMixCollateTransform(DefaultCollateTransform):
    def __init__(self, **kwargs: Any): ...
```

#### See also

- [DefaultCollateTransform](#defaultcollatetransform)



## CutMixUp

[Show source in transforms.py:31](../../../python/sitstart/ml/transforms.py#L31)

#### Signature

```python
class CutMixUp(torch.nn.Module):
    def __init__(self, p: list[float] | None = None, **kwargs: Any): ...
```

### CutMixUp().forward

[Show source in transforms.py:38](../../../python/sitstart/ml/transforms.py#L38)

#### Signature

```python
def forward(self, *inputs: Any) -> Any: ...
```



## CutMixUpBatchTransform

[Show source in transforms.py:73](../../../python/sitstart/ml/transforms.py#L73)

#### Signature

```python
class CutMixUpBatchTransform(BatchTransform):
    def __init__(self, **kwargs: Any): ...
```

#### See also

- [BatchTransform](#batchtransform)



## CutMixUpCollateTransform

[Show source in transforms.py:91](../../../python/sitstart/ml/transforms.py#L91)

#### Signature

```python
class CutMixUpCollateTransform(DefaultCollateTransform):
    def __init__(self, **kwargs: Any): ...
```

#### See also

- [DefaultCollateTransform](#defaultcollatetransform)



## DefaultCollateTransform

[Show source in transforms.py:78](../../../python/sitstart/ml/transforms.py#L78)

#### Signature

```python
class DefaultCollateTransform(BatchTransform):
    def __init__(self, *args: Any, **kwargs: Any): ...
```

#### See also

- [BatchTransform](#batchtransform)

### DefaultCollateTransform().forward

[Show source in transforms.py:82](../../../python/sitstart/ml/transforms.py#L82)

#### Signature

```python
def forward(self, batch: Any) -> Any: ...
```



## IdentityBatchTransform

[Show source in transforms.py:58](../../../python/sitstart/ml/transforms.py#L58)

#### Signature

```python
class IdentityBatchTransform(BatchTransform):
    def __init__(self): ...
```

#### See also

- [BatchTransform](#batchtransform)



## ImageToFloat32

[Show source in transforms.py:23](../../../python/sitstart/ml/transforms.py#L23)

#### Signature

```python
class ImageToFloat32(torch.nn.Module):
    def __init__(self): ...
```

### ImageToFloat32().forward

[Show source in transforms.py:27](../../../python/sitstart/ml/transforms.py#L27)

#### Signature

```python
def forward(self, img: torch.Tensor) -> torch.Tensor: ...
```



## MixUpBatchTransform

[Show source in transforms.py:68](../../../python/sitstart/ml/transforms.py#L68)

#### Signature

```python
class MixUpBatchTransform(BatchTransform):
    def __init__(self, **kwargs: Any): ...
```

#### See also

- [BatchTransform](#batchtransform)



## MixUpCollateTransform

[Show source in transforms.py:96](../../../python/sitstart/ml/transforms.py#L96)

#### Signature

```python
class MixUpCollateTransform(DefaultCollateTransform):
    def __init__(self, **kwargs: Any): ...
```

#### See also

- [DefaultCollateTransform](#defaultcollatetransform)
