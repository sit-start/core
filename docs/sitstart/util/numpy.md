# numpy

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [util](./index.md#util) / numpy

> Auto-generated documentation for [sitstart.util.numpy](../../../python/sitstart/util/numpy.py) module.

- [numpy](#numpy)
  - [implay](#implay)
  - [imresize](#imresize)
  - [imshow](#imshow)
  - [imtile](#imtile)

## implay

[Show source in numpy.py:63](../../../python/sitstart/util/numpy.py#L63)

#### Signature

```python
def implay(arr: np.ndarray, fps: float = 30.0, scale: float = 1.0) -> None: ...
```



## imresize

[Show source in numpy.py:31](../../../python/sitstart/util/numpy.py#L31)

#### Signature

```python
def imresize(arr: np.ndarray, size: tuple[int, int], **kwargs) -> np.ndarray: ...
```



## imshow

[Show source in numpy.py:18](../../../python/sitstart/util/numpy.py#L18)

#### Signature

```python
def imshow(img: np.ndarray, height: float = 4.0) -> None: ...
```



## imtile

[Show source in numpy.py:36](../../../python/sitstart/util/numpy.py#L36)

#### Signature

```python
def imtile(images: np.ndarray | torch.Tensor, cols: int | None = None): ...
```
