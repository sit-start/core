import logging
import re
from time import sleep

import pytest

from sitstart.util.decorators import debug, memoize, timer


@pytest.mark.slow
def test_timer(caplog):
    caplog.set_level(logging.INFO)
    sleep_time_sec = 1

    @timer
    def my_sleep(time_sec):
        sleep(time_sec)

    my_sleep(sleep_time_sec)

    out = re.search(r"Function \'(.*)\' executed in (\d+\.\d+)s", caplog.text)
    assert out and len(out.groups()) == 2
    assert out.groups()[0] == my_sleep.__name__
    assert float(out.groups()[1]) >= sleep_time_sec


def test_memoize():
    num_invocations = 0

    @memoize
    def func_with_side_effects(x: int):
        nonlocal num_invocations
        num_invocations += 1
        return x * num_invocations

    assert func_with_side_effects(1) == 1
    assert func_with_side_effects(2) == 4
    assert func_with_side_effects(2) == 4


def test_debug(caplog):
    caplog.set_level(logging.DEBUG)

    @debug
    def pow(x, y):
        return x**y

    args = (2,)
    kwargs = dict(y=3)
    result = pow(*args, **kwargs)

    assert f"Function {pow.__name__!r}" in caplog.text
    assert f"args {args!r}" in caplog.text
    assert f"kwargs {kwargs!r}" in caplog.text
    assert f"returned {result!r}" in caplog.text
