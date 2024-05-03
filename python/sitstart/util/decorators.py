import logging
from functools import wraps

from sitstart.logging import get_logger

logger = get_logger(__name__)


def timer(func):
    from time import time

    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        logger.info(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrapper


def memoize(func):
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, tuple(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper


def debug(func):
    level = logger.level

    @wraps(func)
    def wrapper(*args, **kwargs):
        if logger.level > logging.DEBUG:
            logger.setLevel(logging.DEBUG)
        logger.debug(
            f"Function {func.__name__!r} called "
            f"with args {args!r} and kwargs {kwargs!r}"
        )
        result = func(*args, **kwargs)
        logger.debug(f"Function {func.__name__!r} returned {result!r}")
        logger.setLevel(level)

        return result

    return wrapper
