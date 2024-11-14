# metrics

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [ml](./index.md#ml) / metrics

> Auto-generated documentation for [sitstart.ml.metrics](../../../python/sitstart/ml/metrics.py) module.

- [metrics](#metrics)
  - [AverageMulticlassRecall](#averagemulticlassrecall)
    - [AverageMulticlassRecall().compute](#averagemulticlassrecall()compute)
  - [MulticlassConfusionMatrix](#multiclassconfusionmatrix)
    - [MulticlassConfusionMatrix().plot](#multiclassconfusionmatrix()plot)

## AverageMulticlassRecall

[Show source in metrics.py:10](../../../python/sitstart/ml/metrics.py#L10)

Unweighted average multiclass recall.

Wraps torcheval.metrics.MulticlassRecall which, as of v0.0.7, errors
when average='macro', num_classes is specified, and the metric is
updated without a true or false positive in every class.

#### Signature

```python
class AverageMulticlassRecall(MulticlassRecall):
    def __init__(self, num_classes: int) -> None: ...
```

### AverageMulticlassRecall().compute

[Show source in metrics.py:21](../../../python/sitstart/ml/metrics.py#L21)

#### Signature

```python
def compute(self: "AverageMulticlassRecall") -> torch.Tensor: ...
```



## MulticlassConfusionMatrix

[Show source in metrics.py:25](../../../python/sitstart/ml/metrics.py#L25)

#### Signature

```python
class MulticlassConfusionMatrix(_MulticlassConfusionMatrix):
    def __init__(
        self, num_classes: int, labels: list[str] | None = None, **kwargs: Any
    ) -> None: ...
```

### MulticlassConfusionMatrix().plot

[Show source in metrics.py:34](../../../python/sitstart/ml/metrics.py#L34)

#### Signature

```python
def plot(self, labels: list[str] | None = None, **kwargs: Any) -> Any: ...
```
