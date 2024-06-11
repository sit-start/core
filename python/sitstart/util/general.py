import functools
import inspect
from collections.abc import Iterable
from typing import Any, Callable
from urllib.parse import urlparse


# @source: https://stackoverflow.com/questions/31174295
def rsetattr(obj: object, attr: str, val: Any) -> None:
    """setattr() with keylist notation"""
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj: object, attr: str, *args: Any, **kwargs: Any) -> Any:
    """getattr() with keylist notation"""

    def _getattr(obj, attr):
        return getattr(obj, attr, *args, **kwargs)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rhasattr(obj: object, attr: str) -> bool:
    """hasattr() with keylist notation"""
    return rgetattr(obj, attr, None) is not None


def is_valid_url(url: str) -> bool:
    """Returns True iff the given url is valid."""
    parsed_url = urlparse(url)
    return bool(parsed_url.scheme) and bool(parsed_url.netloc)


def caller() -> str:
    """Returns the name of the calling function."""
    return inspect.stack()[2].function


def hasarg(func: Callable, arg: str, arg_type: type) -> bool:
    """Returns True iff the given function has the given argument."""
    argspec = inspect.getfullargspec(func)
    return arg in argspec.args and argspec.annotations.get(arg) == arg_type


# @source: https://stackoverflow.com/questions/3023503/how-can-i-check-if-an-object-is-an-iterator-in-python
def is_iterable(obj: Any, exclude: tuple[type] = (str,)) -> bool:
    """Returns True iff the given object is iterable."""
    return not isinstance(obj, exclude) and isinstance(obj, Iterable)
