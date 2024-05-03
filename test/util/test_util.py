import pytest

from sitstart.util import rgetattr, rhasattr, rsetattr


@pytest.fixture
def test_obj():
    class A:
        a: int = 1

    class B:
        a: A = A()
        b: int = 2

    return B()


def test_rgetattr(test_obj):
    assert rgetattr(test_obj, "a.a") == 1
    assert rgetattr(test_obj, "b") == 2
    assert rgetattr(test_obj, "a.c", None) is None


def test_rsetattr(test_obj):
    rsetattr(test_obj, "a.a", 2)
    assert test_obj.a.a == 2
    rsetattr(test_obj, "b", 3)
    assert test_obj.b == 3
    rsetattr(test_obj, "a.c", 4)
    assert test_obj.a.c == 4


def test_rhasattr(test_obj):
    assert rhasattr(test_obj, "a.a")
    assert rhasattr(test_obj, "b")
    assert not rhasattr(test_obj, "a.c")
