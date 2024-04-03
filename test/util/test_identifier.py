import pytest

from ktd.util.identifier import RUN_ID, StringIdType


def test_is_valid():
    id_type = RUN_ID

    assert id_type.is_valid("R55555")
    assert not id_type.is_valid("R00000")
    assert not id_type.is_valid("R0000")
    assert not id_type.is_valid("P00000")


def test_next_sequential():
    id_type = RUN_ID

    assert id_type.next(existing=["R00000"]) == id_type._id(str(id_type.start))
    assert id_type.next(existing=["R55555"]) == "R55556"

    with pytest.raises(ValueError):
        id_type.next(existing=["R99999"])


def test_next_random():
    id_type = StringIdType(**{**RUN_ID.__dict__, "sequential": False})

    assert id_type.next(seed=42) == "R60227"

    with pytest.raises(RuntimeError):
        id_type.next(exists=lambda x: True, max_attempts=10)
