# losses

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [ml](./index.md#ml) / losses

> Auto-generated documentation for [sitstart.ml.losses](../../../python/sitstart/ml/losses.py) module.

- [losses](#losses)
  - [FocalLoss](#focalloss)
    - [FocalLoss().forward](#focalloss()forward)
  - [_reduce](#_reduce)

## FocalLoss

[Show source in losses.py:16](../../../python/sitstart/ml/losses.py#L16)

Multi-class focal loss from https://arxiv.org/abs/1708.02002

#### Signature

```python
class FocalLoss(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
        gamma: float = 2.0,
    ) -> None: ...
```

### FocalLoss().forward

[Show source in losses.py:30](../../../python/sitstart/ml/losses.py#L30)

#### Signature

```python
def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor: ...
```



## _reduce

[Show source in losses.py:5](../../../python/sitstart/ml/losses.py#L5)

Reduce the given input. Reduction method is as in torch loss functions.

#### Signature

```python
def _reduce(input: torch.Tensor, reduction: str) -> torch.Tensor: ...
```
