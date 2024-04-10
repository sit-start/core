import string

import pytest

from ktd.util.string import (
    camel_to_snake,
    int_to_str,
    is_from_alphabet,
    next_str,
    rand_str,
    snake_to_camel,
    str_to_int,
    strip_ansi_codes,
    terminal_hyperlink,
    to_str,
    truncate,
)


def test_strip_ansi_codes():
    assert strip_ansi_codes("") == ""
    assert strip_ansi_codes("\x1b[31mRed Text\x1b[0m") == "Red Text"
    assert (
        strip_ansi_codes("\x1b[33m\x1b[1mBold Yellow Text\x1b[0m") == "Bold Yellow Text"
    )
    assert strip_ansi_codes("\u001b[44mBlue Background\u001b[0m") == "Blue Background"


def test_truncate():
    assert truncate("This is a test", 7) == "This..."
    assert truncate("Short text", 20) == "Short text"
    assert truncate("Long text that needs truncation", 12) == "Long text..."


def test_terminal_hyperlink():
    assert (
        terminal_hyperlink("https://www.google.com", "Google")
        == "\033]8;;https://www.google.com\033\\Google\033]8;;\033\\"
    )


def test_rand_str():
    def from_alphabet(s: str, alphabet: str) -> bool:
        return all(c in alphabet for c in s)

    assert from_alphabet(rand_str(5, string.ascii_lowercase), string.ascii_lowercase)
    with pytest.raises(RuntimeError):
        _ = rand_str(8, "a", test=lambda s: False)
    assert from_alphabet(rand_str(8, "a", test=lambda s: True), "a")


def test_str_to_int():
    assert str_to_int("12345", string.digits) == 12345
    assert str_to_int("FF", "0123456789ABCDEF") == 255
    assert str_to_int("ABC", string.ascii_uppercase) == 28
    with pytest.raises(ValueError):
        _ = str_to_int("", string.ascii_lowercase)


def test_int_to_str():
    assert int_to_str(12345, string.digits) == "12345"
    assert int_to_str(255, "0123456789ABCDEF") == "FF"
    assert int_to_str(28, string.ascii_uppercase, length=3) == "ABC"


def test_is_from_alphabet():
    assert is_from_alphabet("", "")
    assert is_from_alphabet("abc", string.ascii_lowercase)
    assert not is_from_alphabet("123", string.ascii_lowercase)
    assert not is_from_alphabet("XYZ", string.ascii_lowercase)
    assert is_from_alphabet("XYZ", string.ascii_uppercase)


def test_next_str():
    assert next_str("abc", string.ascii_lowercase) == "abd"
    assert next_str("999", string.digits) == "1000"
    assert next_str("0F0F", "0123456789ABCDEF") == "0F10"
    assert next_str("1", "10") == "0"


def test_to_str():
    assert to_str(True) == "True"
    assert to_str(False) == "False"
    assert to_str(123) == "123"
    assert to_str("abc") == "'abc'"
    assert to_str("abc", use_repr=False) == "abc"
    assert to_str(3.14159265) == "3.1415927"
    assert to_str(3.14159265, precision=2) == "3.1"
    assert to_str([1, 2, 3]) == "[1, 2, 3]"
    assert to_str(["a", "b", "c"], list_sep="") == "['a''b''c']"
    assert to_str([1, 2, 3], list_ends=["[(", ")]"]) == "[(1, 2, 3)]"
    assert to_str({"a": 1, "b": 2}) == "{'a': 1, 'b': 2}"
    assert to_str({"a": 1, "b": [1, 2]}, dict_sep=";\n") == "{'a': 1;\n'b': [1, 2]}"
    assert to_str({"a": 1, "b": 2}, dict_ends=["{(", ")}"]) == "{('a': 1, 'b': 2)}"


def test_camel_to_snake():
    assert camel_to_snake("CamelCase") == "camel_case"
    assert camel_to_snake("CamelAI") == "camel_a_i"
    assert camel_to_snake("CamelAI", reversible=False) == "camel_ai"
    assert camel_to_snake("camelCase") == "camel_case"
    assert camel_to_snake("camel") == "camel"


def test_snake_to_camel():
    assert snake_to_camel("snake_case") == "SnakeCase"
    assert snake_to_camel("SNAKE_CASE") == "SnakeCase"
    assert snake_to_camel("snake_case", lowercase=True) == "snakeCase"
    assert snake_to_camel("snake_a_i") == "SnakeAI"
    assert snake_to_camel("snake_ai") == "SnakeAi"
