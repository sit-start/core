import pytest
from omegaconf import DictConfig, ListConfig

from sitstart.util.container import flatten, get, update, walk

TYPE_ARGS = [
    dict(dict_types=[dict], list_types=[list]),
    dict(dict_types=[DictConfig], list_types=[ListConfig]),
]


def test_flatten():
    test_cases = [
        ({}, {}),
        ({"b": {"c": 1}}, {"b.c": 1}),
        ([0, 1, "c"], {"0": 0, "1": 1, "2": "c"}),
        ({"b": {"c": [0, 1]}}, {"b.c.0": 0, "b.c.1": 1}),
    ]

    for type_args in TYPE_ARGS:
        for input, expected_result in test_cases:
            DictType, ListType = type_args["dict_types"][0], type_args["list_types"][0]
            ContainerType = DictType if isinstance(input, dict) else ListType
            input = ContainerType(input)
            expected_result = DictType(expected_result)

            result = flatten(input, **type_args, result_init=lambda: DictType({}))

            assert type_args and input is not None and result == expected_result

    with pytest.raises(ValueError):
        flatten([0, 1], list_types=[])


def test_walk():
    test_cases = [
        ({}, [("", [], [])]),
        ([], [("", [], [])]),
        ({"a": {"b": 1}}, [("", ["a"], []), ("a", [], ["b"])]),
        ([{"a": 1}, 2], [("", ["0"], ["1"]), ("0", [], ["a"])]),
        ([[[1]]], [("", ["0"], []), ("0", ["0"], []), ("0.0", [], ["0"])]),
    ]

    for type_args in TYPE_ARGS:
        for input, expected_result in test_cases:
            DictType, ListType = type_args["dict_types"][0], type_args["list_types"][0]
            InputType = DictType if isinstance(input, dict) else ListType
            result = list(walk(InputType(input), **type_args))

            assert type_args and input is not None and result == expected_result


def test_update():
    test_cases = [
        (({}, "a", 2), {"a": 2}),
        (({"a": 1, "b": {"c": 2}}, "b.d", 3), {"a": 1, "b": {"c": 2, "d": 3}}),
        (({"a": [1, {"b": {"c": 2}}]}, "a.1.b.c", 3), {"a": [1, {"b": {"c": 3}}]}),
    ]

    for type_args in TYPE_ARGS:
        for (input, *args), expected_result in test_cases:
            DictType, ListType = type_args["dict_types"][0], type_args["list_types"][0]
            ContainerType = DictType if isinstance(input, dict) else ListType
            result = ContainerType(input)
            expected_result = ContainerType(expected_result)
            update(result, *args, **type_args)

            assert type_args and input is not None and result == expected_result

    with pytest.raises(KeyError):
        update({"a": 1}, "a.b", 2)

    with pytest.raises(KeyError):
        update({}, "a.b", 1)


def test_get():
    test_cases = [
        (({}, "a", 2), 2),
        (({"a": {"b": 1}}, "a.b"), 1),
        (([0, 1, {"a": 2}], "2.a"), 2),
        (({}, "a.1.b", 0), 0),
        (([0, 1, {"a": 2}], "2.a"), 2),
    ]

    for type_args in TYPE_ARGS:
        for (input, *args), expected_result in test_cases:
            DictType, ListType = type_args["dict_types"][0], type_args["list_types"][0]
            ContainerType = DictType if isinstance(input, dict) else ListType
            input = ContainerType(input)
            result = get(input, *args, **type_args)

            assert type_args and input is not None and result == expected_result

    with pytest.raises(KeyError):
        assert get({"a": 1}, "a.b")
