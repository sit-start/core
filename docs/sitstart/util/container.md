# container

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [util](./index.md#util) / container

> Auto-generated documentation for [sitstart.util.container](../../../python/sitstart/util/container.py) module.

- [container](#container)
  - [flatten](#flatten)
  - [get](#get)
  - [update](#update)
  - [walk](#walk)

## flatten

[Show source in container.py:4](../../../python/sitstart/util/container.py#L4)

Flatten nested dict- and list-like objects

#### Signature

```python
def flatten(
    obj: Any,
    parent_key: str = "",
    sep: str = ".",
    dict_types: list[type] = [dict],
    list_types: list[type] = [list],
    result_init: Callable[[], Any] = lambda: dict(),
) -> Any: ...
```



## get

[Show source in container.py:123](../../../python/sitstart/util/container.py#L123)

get() with keylist notation for nested dict- & list-like objects

#### Signature

```python
def get(
    obj: Any,
    keylist: str,
    default: Any = None,
    sep: str = ".",
    dict_types=[dict],
    list_types=[list],
) -> Any: ...
```



## update

[Show source in container.py:92](../../../python/sitstart/util/container.py#L92)

update() with keylist notation for nested dict- and list-like objects.

#### Signature

```python
def update(
    obj: Any,
    keylist: str,
    value: Any,
    sep: str = ".",
    dict_types=[dict],
    list_types=[list],
) -> None: ...
```



## walk

[Show source in container.py:40](../../../python/sitstart/util/container.py#L40)

Generate dotlist-style keys for nested dict- and list-like objects.

Analagous to os.walk.

#### Signature

```python
def walk(
    container: Any,
    top: str | None = "",
    topdown: bool = True,
    dict_types: list[type] = [dict],
    list_types: list[type] = [list],
) -> Iterator[tuple[str, list[str], list[str]]]: ...
```
