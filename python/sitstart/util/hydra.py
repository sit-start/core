import copy
import functools
import operator
from typing import Any

from hydra.utils import instantiate as _instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from sitstart.logging import get_logger
from sitstart.util.decorators import once
from sitstart.util.container import get, update, walk

logger = get_logger(__name__)


@once
def register_omegaconf_resolvers():
    """Registers OmegaConf resolvers.

    Includes all operators in `operator` that don't mutate arguments:
    https://docs.python.org/3/library/operator.html#mapping-operators-to-functions

    Also includes:
      - logical operators `and` and `or`;
      - `min` and `max` over 2 or more values; and
      - ternary operator `if`.
    """
    for op in (op for op in operator.__all__ if op not in ["setitem", "delitem"]):
        OmegaConf.register_new_resolver(op, getattr(operator, op), replace=True)
    OmegaConf.register_new_resolver("and", lambda x, y: x and y, replace=True)
    OmegaConf.register_new_resolver("or", lambda x, y: x or y, replace=True)
    OmegaConf.register_new_resolver("min", lambda x, y, *z: min(x, y, *z), replace=True)
    OmegaConf.register_new_resolver("max", lambda x, y, *z: max(x, y, *z), replace=True)
    OmegaConf.register_new_resolver("if", lambda z, x, y: x if z else y, replace=True)


_CONTAINER_TYPES: Any = dict(
    dict_types=[dict, DictConfig], list_types=[list, ListConfig]
)


def _get(obj: Any, keylist: str, default: Any = None) -> Any:
    return get(obj, keylist, default=default, **_CONTAINER_TYPES)


def _update(obj: Any, keylist: str, value: Any) -> None:
    update(obj, keylist, value, **_CONTAINER_TYPES)


def instantiate(
    config: Any, _defer_: bool = True, _detach_: bool = True, *args: Any, **kwargs: Any
) -> Any:
    """Config instantiation with additional options.

    Wraps hydra.util.instantate and supports the same arguments, with
    the below additions and modifications.

    Args:
        _defer_: Defer instantiation of targets with `_defer_: True`.
        _detach_: Detach the instantiated object from its parent.
        _convert_: Same as in hydra.utils.instantiate, with the
            additional option of "full", which is equivalent to "all"
            but with full instantiation of partially instantiated
            objects.

    Returns:
        The instantiated object.
    """
    if _defer_ and not _detach_:
        # Hydra's instantiate() sets the output's parent to the input config;
        # when further resolving the output, root config values from
        # the input config are used, not those the instantiated config, which
        # can produce incorrect results.
        logger.warning(
            "Later resolution of deferred nodes can fail if the "
            "instantiated config is not detached. Forcing _detach_ = True"
        )
        _detach_ = True

    result = copy.deepcopy(config)
    allow_objects = result._get_flag("allow_objects")
    struct = OmegaConf.is_struct(config)
    readonly = OmegaConf.is_readonly(config)
    result._set_flag(
        flags=["allow_objects", "struct", "readonly"],
        values=[True, False, False],
    )

    do_full_instantiation = False
    if kwargs.get("_convert_") == "full":
        do_full_instantiation = True
        kwargs["_convert_"] = "all"

    deferred_targets = []
    existing_partial_instances = []
    partial_instances = []

    for key, _, child_keys in walk(config, topdown=False, **_CONTAINER_TYPES):
        obj = _get(result, key)
        for child_key in child_keys if do_full_instantiation else []:
            if isinstance(_get(obj, child_key), functools.partial):
                full_child_key = f"{key}.{child_key}".strip(".")
                existing_partial_instances.append(full_child_key)
                _update(result, full_child_key, None)
        if type(obj) in _CONTAINER_TYPES["dict_types"] and "_target_" in obj:
            defer = _defer_ and _get(obj, "_defer_", False)
            if defer:
                # remove the target, to be restored after instantiation
                _update(result, key, None)
                deferred_targets.append(key)
            elif do_full_instantiation and _get(obj, "_partial_", False):
                partial_instances.append(key)
            # remove "_defer_" since Hydra doesn't recognize it
            _ = obj.pop("_defer_", None)

    result = _instantiate(result, *args, **kwargs)
    if _detach_ and OmegaConf.is_config(result):
        result._set_parent(None)

    # restore deferred targets
    # TODO: confirm this works for nested instantiations
    for key in reversed(deferred_targets):
        deferred = _get(config, key)
        _update(result, key, deferred)

    # fully instantiate
    for key in partial_instances:
        partial = _get(result, key)
        _update(result, key, partial())
    for key in existing_partial_instances:
        partial = _get(config, key)
        _update(result, key, partial())

    # restore flags
    if OmegaConf.is_config(result):
        result._set_flag(
            flags=["allow_objects", "struct", "readonly"],
            values=[allow_objects, struct, readonly],
        )

    return result
