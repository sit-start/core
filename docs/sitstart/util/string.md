# string

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [util](./index.md#util) / string

> Auto-generated documentation for [sitstart.util.string](../../../python/sitstart/util/string.py) module.

- [string](#string)
  - [camel_to_snake](#camel_to_snake)
  - [int_to_str](#int_to_str)
  - [is_from_alphabet](#is_from_alphabet)
  - [next_str](#next_str)
  - [rand_str](#rand_str)
  - [snake_to_camel](#snake_to_camel)
  - [str_to_int](#str_to_int)
  - [strip_ansi_codes](#strip_ansi_codes)
  - [terminal_hyperlink](#terminal_hyperlink)
  - [to_str](#to_str)
  - [truncate](#truncate)
  - [verp](#verp)

## camel_to_snake

[Show source in string.py:116](../../../python/sitstart/util/string.py#L116)

#### Signature

```python
def camel_to_snake(s: str, reversible: bool = True) -> str: ...
```



## int_to_str

[Show source in string.py:55](../../../python/sitstart/util/string.py#L55)

#### Signature

```python
def int_to_str(x: int, alphabet: str, length: int | None = None) -> str: ...
```



## is_from_alphabet

[Show source in string.py:66](../../../python/sitstart/util/string.py#L66)

#### Signature

```python
def is_from_alphabet(s: str, alphabet: str) -> bool: ...
```



## next_str

[Show source in string.py:70](../../../python/sitstart/util/string.py#L70)

#### Signature

```python
def next_str(s: str, alphabet: str) -> str: ...
```



## rand_str

[Show source in string.py:27](../../../python/sitstart/util/string.py#L27)

#### Signature

```python
def rand_str(
    length: int = 6,
    alphabet: str | None = None,
    test: Callable[[str], bool] | None = None,
    max_attempts: int = 100,
    seed: int | None = None,
) -> str: ...
```



## snake_to_camel

[Show source in string.py:123](../../../python/sitstart/util/string.py#L123)

#### Signature

```python
def snake_to_camel(s: str, lowercase: bool = False) -> str: ...
```



## str_to_int

[Show source in string.py:47](../../../python/sitstart/util/string.py#L47)

#### Signature

```python
def str_to_int(s: str, alphabet: str) -> int: ...
```



## strip_ansi_codes

[Show source in string.py:9](../../../python/sitstart/util/string.py#L9)

#### Signature

```python
def strip_ansi_codes(s: str) -> str: ...
```



## terminal_hyperlink

[Show source in string.py:21](../../../python/sitstart/util/string.py#L21)

#### Signature

```python
def terminal_hyperlink(url: str, text: str) -> str: ...
```



## to_str

[Show source in string.py:74](../../../python/sitstart/util/string.py#L74)

#### Signature

```python
def to_str(
    x: Any,
    precision: int = 8,
    list_ends: list[str] = ["[", "]"],
    list_sep: str = ", ",
    dict_ends: list[str] = ["{", "}"],
    dict_sep: str = ", ",
    dict_kv_sep: str = ": ",
    use_repr: bool = True,
) -> str: ...
```



## truncate

[Show source in string.py:15](../../../python/sitstart/util/string.py#L15)

#### Signature

```python
def truncate(s: str, max_len: int = 50) -> str: ...
```



## verp

[Show source in string.py:128](../../../python/sitstart/util/string.py#L128)

Interpolate variables in the given string.

Syntax for interpolation targets is similar to that for context
variables in GitHub Actions, where `${{ foo.bar }}` interpolates to
`vars["foo"]["bar"]`.

Keys must be valid Python identifiers.

Values must be strings, ints, or floats. String values cannot be
interpolation targets.

#### Signature

```python
def verp(s: str, vars: dict[str, Any]) -> str: ...
```
