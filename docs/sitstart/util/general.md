# general

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [util](./index.md#util) / general

> Auto-generated documentation for [sitstart.util.general](../../../python/sitstart/util/general.py) module.

- [general](#general)
  - [caller](#caller)
  - [hasarg](#hasarg)
  - [is_iterable](#is_iterable)
  - [is_valid_url](#is_valid_url)
  - [rgetattr](#rgetattr)
  - [rhasattr](#rhasattr)
  - [rsetattr](#rsetattr)

## caller

[Show source in general.py:35](../../../python/sitstart/util/general.py#L35)

Returns the name of the calling function.

#### Signature

```python
def caller() -> str: ...
```



## hasarg

[Show source in general.py:40](../../../python/sitstart/util/general.py#L40)

Returns True iff the given function has the given argument.

#### Signature

```python
def hasarg(func: Callable, arg: str, arg_type: type) -> bool: ...
```



## is_iterable

[Show source in general.py:47](../../../python/sitstart/util/general.py#L47)

Returns True iff the given object is iterable.

#### Signature

```python
def is_iterable(obj: Any, exclude: tuple[type] = (str)) -> bool: ...
```



## is_valid_url

[Show source in general.py:29](../../../python/sitstart/util/general.py#L29)

Returns True iff the given url is valid.

#### Signature

```python
def is_valid_url(url: str) -> bool: ...
```



## rgetattr

[Show source in general.py:15](../../../python/sitstart/util/general.py#L15)

getattr() with keylist notation

#### Signature

```python
def rgetattr(obj: object, attr: str, *args: Any, **kwargs: Any) -> Any: ...
```



## rhasattr

[Show source in general.py:24](../../../python/sitstart/util/general.py#L24)

hasattr() with keylist notation

#### Signature

```python
def rhasattr(obj: object, attr: str) -> bool: ...
```



## rsetattr

[Show source in general.py:9](../../../python/sitstart/util/general.py#L9)

setattr() with keylist notation

#### Signature

```python
def rsetattr(obj: object, attr: str, val: Any) -> None: ...
```
