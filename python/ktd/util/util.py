import functools
from typing import Any
from urllib.parse import urlparse


# https://stackoverflow.com/questions/31174295
def rsetattr(obj: object, attr: str, val: Any) -> None:
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj: object, attr: str, *args: Any, **kwargs: Any) -> Any:
    def _getattr(obj, attr):
        return getattr(obj, attr, *args, **kwargs)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rhasattr(obj: object, attr: str) -> bool:
    return rgetattr(obj, attr, None) is not None


def is_valid_url(url: str) -> bool:
    parsed_url = urlparse(url)
    return bool(parsed_url.scheme) and bool(parsed_url.netloc)


def flatten_dict(
    d: dict[str, Any], parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    result = {}
    for key, val in d.items():
        if not isinstance(key, str):
            raise KeyError(f"Key {key} is not a string.")
        new_key = (parent_key + sep + key).lstrip(sep)
        if isinstance(val, dict):
            result.update(flatten_dict(val, parent_key=new_key, sep=sep))
        else:
            result[new_key] = val
    return result
