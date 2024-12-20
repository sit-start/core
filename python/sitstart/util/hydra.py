import operator
from pathlib import Path
from typing import Any
from os.path import isabs, dirname, realpath, relpath

import hydra
from omegaconf import Container, ListConfig, OmegaConf
from torch import tensor

from sitstart.logging import get_logger
from sitstart.util.decorators import once

VERSION_BASE = "1.2"
CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "ml" / "experiments" / "conf")

logger = get_logger(__name__)


def _zip(*iterables: Any) -> ListConfig:
    return ListConfig([ListConfig(x) for x in zip(*iterables)])


def _foreach(resolver_name: str, args: list[Any]) -> Any | None:
    resolver = OmegaConf._get_resolver(resolver_name)

    def _apply(x: Any) -> Any:
        return resolver(config=None, parent=None, node=None, args=x, args_str=None)  # type: ignore

    return ListConfig([_apply(arg) for arg in args])


def _sublist(the_list: ListConfig | list[Any], indices: list[int]) -> ListConfig:
    return ListConfig([the_list[i] for i in indices])


@once
def register_omegaconf_resolvers():
    """Registers OmegaConf resolvers.

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
    """
    for op in (op for op in operator.__all__ if op not in ["setitem", "delitem"]):
        OmegaConf.register_new_resolver(op, getattr(operator, op), replace=True)
    OmegaConf.register_new_resolver("and", lambda x, y: x and y, replace=True)
    OmegaConf.register_new_resolver("or", lambda x, y: x or y, replace=True)
    OmegaConf.register_new_resolver("min", min, replace=True)
    OmegaConf.register_new_resolver("max", max, replace=True)
    OmegaConf.register_new_resolver("if", lambda z, x, y: x if z else y, replace=True)
    OmegaConf.register_new_resolver("foreach", _foreach, replace=True)
    OmegaConf.register_new_resolver("list", lambda t: list(t), replace=True)
    OmegaConf.register_new_resolver("sublist", _sublist, replace=True)
    OmegaConf.register_new_resolver("tensor", lambda t: tensor(list(t)), replace=True)
    OmegaConf.register_new_resolver("zip", _zip, replace=True)
    OmegaConf.register_new_resolver("round", round, replace=True)
    OmegaConf.register_new_resolver("abs", abs, replace=True)
    OmegaConf.register_new_resolver("int", lambda x: int(x), replace=True)


def load_config(
    name: str,
    config_path: str,
    overrides: list[str] | None = None,
    version_base: str = VERSION_BASE,
) -> Container:
    """Loads an experiment config. For testing/debugging.

    Args:
        name: Name of the experiment config in `config_path`.
        config_path: Absolute path to the config directory.
        overrides: List of Hydra config overrides.
        version_base: Hydra version base.
    """
    if isabs(config_path):
        config_path = relpath(config_path, realpath(dirname(__file__)))
    with hydra.initialize(version_base=version_base, config_path=config_path):
        return hydra.compose(config_name=name, overrides=overrides)
