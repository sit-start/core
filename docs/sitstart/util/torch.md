# torch

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [util](./index.md#util) / torch

> Auto-generated documentation for [sitstart.util.torch](../../../python/sitstart/util/torch.py) module.

- [torch](#torch)
  - [generator_from_seed](#generator_from_seed)
  - [int_hist](#int_hist)
  - [is_integer](#is_integer)
  - [overlay_text](#overlay_text)
  - [randint](#randint)
  - [unnormalize](#unnormalize)

## generator_from_seed

[Show source in torch.py:82](../../../python/sitstart/util/torch.py#L82)

Create a torch.Generator from the given seed.

Uses a random seed if `seed` is None.

#### Signature

```python
def generator_from_seed(
    seed: int | None = None, device: torch.device | str = "cpu"
) -> torch.Generator: ...
```



## int_hist

[Show source in torch.py:8](../../../python/sitstart/util/torch.py#L8)

#### Signature

```python
def int_hist(
    data: torch.Tensor | list, min: int | None = None, max: int | None = None
) -> torch.Tensor: ...
```



## is_integer

[Show source in torch.py:97](../../../python/sitstart/util/torch.py#L97)

Check if the input is an integer type.

#### Signature

```python
def is_integer(input: torch.Tensor | torch.dtype) -> bool: ...
```



## overlay_text

[Show source in torch.py:32](../../../python/sitstart/util/torch.py#L32)

#### Signature

```python
def overlay_text(
    images: torch.Tensor,
    text: str | list[str],
    position: tuple[int, int] = (0, 0),
    font_name: str = DEFAULT_FONT_NAME,
    font_size: int = 10,
    fill: str | tuple[int, int, int] = "yellow",
): ...
```

#### See also

- [DEFAULT_FONT_NAME](#default_font_name)



## randint

[Show source in torch.py:59](../../../python/sitstart/util/torch.py#L59)

Generate random integers.

Same as torch.randint with default values for `low`, `high`, and `dtype`.

#### Signature

```python
def randint(
    low: int | None = None,
    high: int | None = None,
    size: tuple[int, ...] | None = None,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor: ...
```



## unnormalize

[Show source in torch.py:20](../../../python/sitstart/util/torch.py#L20)

#### Signature

```python
def unnormalize(
    images: torch.Tensor,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> torch.Tensor: ...
```
