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
