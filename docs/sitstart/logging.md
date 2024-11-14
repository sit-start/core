# logging

[Core Index](../README.md#core-index) / [sitstart](./index.md#sitstart) / logging

> Auto-generated documentation for [sitstart.logging](../../python/sitstart/logging.py) module.

- [logging](#logging)
  - [Format](#format)
  - [get_logger](#get_logger)

## Format

[Show source in logging.py:5](../../python/sitstart/logging.py#L5)

#### Signature

```python
class Format(Enum): ...
```



## get_logger

[Show source in logging.py:22](../../python/sitstart/logging.py#L22)

#### Signature

```python
def get_logger(
    name: str | None = None,
    format: str | Format | None = Format.GLOG,
    level: int = logging.INFO,
    force: bool = False,
) -> logging.Logger: ...
```
