from typing import Any, Callable, Iterator


def flatten(
    obj: Any,
    parent_key: str = "",
    sep: str = ".",
    dict_types: list[type] = [dict],
    list_types: list[type] = [list],
    result_init: Callable[[], Any] = lambda: dict(),
) -> Any:
    """Flatten nested dict- and list-like objects"""
    kwargs = {
        "sep": sep,
        "dict_types": dict_types,
        "list_types": list_types,
        "result_init": result_init,
    }
    container_types = dict_types + list_types
    if type(obj) not in container_types:
        raise ValueError("object is not a container")

    result = result_init()
    if type(result) not in dict_types:
        raise ValueError("result type not a dict type")

    for key in obj.keys() if type(obj) in dict_types else range(len(obj)):
        if not isinstance(key, (str, int)):
            raise KeyError(f"Key {key!r} must be int or str.")
        new_key = (parent_key + sep + str(key)).lstrip(sep)
        val = obj[key]
        if type(val) in container_types:
            result.update(flatten(val, parent_key=new_key, **kwargs))
        else:
            result[new_key] = val

    return result


def walk(
    container: Any,
    top: str | None = "",
    topdown: bool = True,
    dict_types: list[type] = [dict],
    list_types: list[type] = [list],
) -> Iterator[tuple[str, list[str], list[str]]]:
    """Generate dotlist-style keys for nested dict- and list-like objects.

    Analagous to os.walk.
    """
    container_types = dict_types + list_types
    type_args: Any = dict(dict_types=dict_types, list_types=list_types)

    while top is not None:
        _top = top
        top = None
        container_keys, object_keys = [], []

        def _full_dotlist_key(key: str | int) -> str:
            return f"{_top}.{key}".strip(".")

        top_container = get(container, _top, **type_args)
        if type(top_container) in dict_types:
            keys = top_container.keys()
        elif type(top_container) in list_types:
            keys = range(len(top_container))
        else:
            raise ValueError(f"object is not a container: {_top!r}")

        for k in keys:
            full_key = _full_dotlist_key(k)
            value = get(container, full_key, **type_args)
            if type(value) in container_types:
                container_keys.append(str(k))
            else:
                object_keys.append(str(k))

        if topdown:
            yield _top, container_keys, object_keys
            for k in container_keys:
                yield from walk(
                    container, _full_dotlist_key(k), topdown=topdown, **type_args
                )
        else:
            for k in container_keys:
                yield from walk(
                    container, _full_dotlist_key(k), topdown=topdown, **type_args
                )
            yield _top, container_keys, object_keys


def update(
    obj: Any,
    keylist: str,
    value: Any,
    sep: str = ".",
    dict_types=[dict],
    list_types=[list],
) -> None:
    """update() with keylist notation for nested dict- and list-like objects."""
    keylist = keylist.strip(sep)

    keys = keylist.split(sep)
    if not keys:
        raise KeyError("key cannot be empty.")

    container_types = dict_types + list_types

    for i, k in enumerate(keys):
        try:
            if type(obj) not in container_types:
                raise ValueError("object is not a container")
            if type(obj) in list_types:
                k = int(k)
            if i < len(keys) - 1:
                obj = obj[k]
            else:
                obj[k] = value
        except Exception as e:
            raise KeyError(f"Invalid key {sep.join(keys[:i+1])!r}") from e


def get(
    obj: Any,
    keylist: str,
    default: Any = None,
    sep: str = ".",
    dict_types=[dict],
    list_types=[list],
) -> Any:
    """get() with keylist notation for nested dict- & list-like objects"""
    keylist = keylist.strip(sep)

    if not keylist:
        return obj

    for k in keylist.split(sep):
        try:
            if type(obj) in list_types:
                obj = obj[int(k)]
            elif type(obj) in dict_types:
                obj = obj[k]
            else:
                return default
        except (IndexError, KeyError):
            return default

    return obj
