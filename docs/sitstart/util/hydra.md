# hydra

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [util](./index.md#util) / hydra

> Auto-generated documentation for [sitstart.util.hydra](../../../python/sitstart/util/hydra.py) module.

- [hydra](#hydra)
  - [load_config](#load_config)
  - [register_omegaconf_resolvers](#register_omegaconf_resolvers)

## load_config

[Show source in hydra.py:69](../../../python/sitstart/util/hydra.py#L69)

Loads an experiment config. For testing/debugging.

#### Arguments

- `name` - Name of the experiment config in `config_path`.
- `config_path` - Absolute path to the config directory.
- `overrides` - List of Hydra config overrides.
- `version_base` - Hydra version base.

#### Signature

```python
def load_config(
    name: str,
    config_path: str,
    overrides: list[str] | None = None,
    version_base: str = VERSION_BASE,
) -> Container: ...
```

#### See also

- [VERSION_BASE](#version_base)



## register_omegaconf_resolvers

[Show source in hydra.py:36](../../../python/sitstart/util/hydra.py#L36)

Registers OmegaConf resolvers.

Includes all operators in `operator` that don't mutate arguments:
https://docs.python.org/3/library/operator.html#mapping-operators-to-functions

Also includes:
  - logical operators `and` and `or`
  - `min` and `max`
  - ternary operator `if`
  - `list`, `sublist`, & `tensor` for converting `ListConfig` to `list` or `Tensor`
  - `foreach` for applying a resolver to each element of a list or ListConfig
  - `zip` for non-strict zipping
  - `round`, `abs`, and `int`, unary operators for floats and ints

#### Signature

```python
@once
def register_omegaconf_resolvers(): ...
```

#### See also

- [once](./decorators.md#once)
