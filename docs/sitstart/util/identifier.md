# identifier

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [util](./index.md#util) / identifier

> Auto-generated documentation for [sitstart.util.identifier](../../../python/sitstart/util/identifier.py) module.

- [identifier](#identifier)
  - [StringIdType](#stringidtype)
    - [StringIdType().is_valid](#stringidtype()is_valid)
    - [StringIdType().next](#stringidtype()next)

## StringIdType

[Show source in identifier.py:12](../../../python/sitstart/util/identifier.py#L12)

#### Signature

```python
class StringIdType: ...
```

### StringIdType().is_valid

[Show source in identifier.py:20](../../../python/sitstart/util/identifier.py#L20)

#### Signature

```python
def is_valid(self, s: str) -> bool: ...
```

### StringIdType().next

[Show source in identifier.py:69](../../../python/sitstart/util/identifier.py#L69)

#### Signature

```python
def next(
    self,
    last: str | None = None,
    exists: Callable[[str], bool] | None = None,
    existing: list[str] | None = None,
    max_attempts: int = 100,
    seed: int | None = None,
) -> str: ...
```
