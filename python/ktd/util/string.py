import random
import re
import string
from typing import Any, Callable


def strip_ansi_codes(s: str) -> str:
    # TODO: add support for RGB color sequence
    # @source: https://stackoverflow.com/questions/15682537/ansi-color-specific-rgb-sequence-bash
    return re.sub(r"\x1b\[([0-9,A-Z]{1,2}(;[0-9]{1,2})?(;[0-9]{3})?)?[m|K]?", "", s)


def truncate(s: str, max_len: int = 50) -> str:
    suffix = "..."
    return s if len(s) <= max_len else s[: max_len - len(suffix)] + suffix


# @source: https://gist.github.com/egmontkob/eb114294efbcd5adb1944c9f3cb5feda
def terminal_hyperlink(url: str, text: str) -> str:
    # Works as expected on iTerm2, but "file://" URLs don't work in VS Code's
    # terminal
    return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"


def rand_str(
    length: int = 6,
    alphabet: str | None = None,
    test: Callable[[str], bool] | None = None,
    max_attempts: int = 100,
    seed: int | None = None,
) -> str:
    random.seed(seed)
    if alphabet is None:
        alphabet = string.digits + string.ascii_uppercase
    for _ in range(max_attempts):
        s = "".join(random.choices(alphabet, k=length))
        if test is None or test(s):
            return s
    raise RuntimeError(
        f"Failed to create a unique string after {max_attempts} attempts. "
        "Consider increasing `max_attempts` or using a larger `length`."
    )


def str_to_int(s: str, alphabet: str) -> int:
    b = len(alphabet)
    n = len(s)
    if n == 0:
        raise ValueError("Input string cannot be empty")
    return sum(alphabet.index(s[i]) * b ** (n - 1 - i) for i in range(n))


def int_to_str(x: int, alphabet: str, length: int | None = None) -> str:
    b = len(alphabet)
    s = ""
    while x > 0:
        s = alphabet[x % b] + s
        x //= b
    if length is not None:
        s = s.rjust(length, alphabet[0])
    return s


def is_from_alphabet(s: str, alphabet: str) -> bool:
    return all(c in alphabet for c in s)


def next_str(s: str, alphabet: str) -> str:
    return int_to_str(str_to_int(s, alphabet) + 1, alphabet, len(s))


def to_str(
    x: Any,
    precision: int = 8,
    list_ends: list[str] = ["[", "]"],
    list_sep: str = ", ",
    dict_ends: list[str] = ["{", "}"],
    dict_sep: str = ", ",
    dict_kv_sep: str = ": ",
    use_repr: bool = True,
) -> str:
    kwargs = locals().copy()
    kwargs.pop("x")

    if not (len(list_ends) == 2 and all(isinstance(x, str) for x in list_ends)):
        raise ValueError("list_ends must be a list of str of len 2")
    if not (len(dict_ends) == 2 and all(isinstance(x, str) for x in dict_ends)):
        raise ValueError("dict_ends must be a list of str of len 2")

    if isinstance(x, bool):
        return "True" if x else "False"
    if isinstance(x, (int, float)):
        return f"{x:.{precision}g}"
    if isinstance(x, list):
        return (
            list_ends[0]
            + f"{list_sep.join(to_str(y, **kwargs) for y in x)}"
            + list_ends[1]
        )
    if isinstance(x, dict):
        # TODO: add support for multiline dict with proper indentation
        return (
            dict_ends[0]
            + dict_sep.join(
                f"{to_str(k, **kwargs)}{dict_kv_sep}{to_str(v, **kwargs)}"
                for k, v in x.items()
            )
            + dict_ends[1]
        )
    return repr(x) if use_repr else str(x)


# https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
def camel_to_snake(s: str, reversible: bool = True) -> str:
    if reversible:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
    s = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", s)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s).lower()


def snake_to_camel(s: str, lowercase: bool = False) -> str:
    result = "".join(word.title() for word in s.split("_"))
    return result[0].lower() + result[1:] if lowercase else result
