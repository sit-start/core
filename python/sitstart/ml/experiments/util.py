import copy
from pathlib import Path
from typing import Any, Callable, cast

import wandb
import yaml
from hydra.utils import instantiate
from omegaconf import Container, DictConfig, ListConfig, OmegaConf, open_dict
from ray.tune.experiment import Experiment
from ray.tune.search import SearchAlgorithm, Searcher, SearchGenerator, create_searcher
from ray.tune.search.sample import Domain

import sitstart.util.hydra
from sitstart.ml.experiments import CONFIG_ROOT, HYDRA_VERSION_BASE
from sitstart.util.container import get, walk
from sitstart.util.decorators import once
from sitstart.util.hydra import load_config

DEFAULT_PARAM_SPACE_KEY = "param_space"
# https://docs.ray.io/en/latest/tune/api/search_space.html
TUNE_SEARCH_SPACE_API = [
    "ray.tune.uniform",
    "ray.tune.quniform",
    "ray.tune.loguniform",
    "ray.tune.qloguniform",
    "ray.tune.randn",
    "ray.tune.qrandn",
    "ray.tune.randint",
    "ray.tune.qrandint",
    "ray.tune.lograndint",
    "ray.tune.qlograndint",
    "ray.tune.choice",
    "ray.tune.sample_from",
    "ray.tune.grid_search",
]


@once
def register_omegaconf_resolvers():
    """Register OmegaConf resolvers.

    Calls `sitstart.util.hydra.register_omegaconf_resolvers` and
    additional registers resolvers for Ray Tune search space API
    functions under the `rt` namespace.
    """
    sitstart.util.hydra.register_omegaconf_resolvers()

    def get_resolver(op: str) -> Callable:
        def resolver(*args: Any) -> Any:
            return DictConfig({"_target_": op, "_args_": ListConfig(args)})

        return resolver

    for op in TUNE_SEARCH_SPACE_API:
        OmegaConf.register_new_resolver(
            "rt." + op.split(".")[-1], get_resolver(op), replace=True
        )


def load_experiment_config(name: str, overrides: list[str] | None = None) -> Container:
    """Load an experiment config. For testing/debugging."""
    overrides = overrides or []
    config = yaml.safe_load(Path(f"{CONFIG_ROOT}/{name}.yaml").read_text())
    if "name" not in config:
        if not any(override.startswith("name=") for override in overrides):
            overrides += ["name=" + name]

    return load_config(
        name,
        config_path=CONFIG_ROOT,
        overrides=overrides,
        version_base=HYDRA_VERSION_BASE,
    )


def get_experiment_wandb_url(config: Container) -> str | None:
    """Get the Weights & Biases URL for the experiment config."""
    proj_name = OmegaConf.select(config, "name", default=None)
    wandb_enabled = OmegaConf.select(config, "wandb.enabled", default=False)
    wandb_entity = wandb.api.default_entity

    if proj_name and wandb_enabled and wandb_entity:
        return f"https://wandb.ai/{wandb_entity}/{proj_name}"

    return None


def get_search_alg(
    config: Container,
    search_alg_key: str = "tune.search_alg",
    metric_key: str = "eval.select.metric",
    mode_key: str = "eval.select.mode",
) -> SearchAlgorithm:
    """Get the search algorithm from the experiment config."""
    search_alg = OmegaConf.select(config, search_alg_key)
    metric = OmegaConf.select(config, metric_key)
    mode = OmegaConf.select(config, mode_key)

    if not search_alg or not metric or not mode:
        raise ValueError(
            "Search algorithm, metric, and mode must exist at the "
            "provided config nodes."
        )

    searcher_or_search_alg = create_searcher(
        search_alg=search_alg, metric=metric, mode=mode
    )
    if isinstance(searcher_or_search_alg, Searcher):
        return SearchGenerator(searcher_or_search_alg)
    return searcher_or_search_alg


def sample_param_space(
    config: Container, param_space_key: str = DEFAULT_PARAM_SPACE_KEY
) -> None:
    """Temporarily resolves the search space for the given config.

    Uses the first sample from the config's search algorithm.
    """
    if not OmegaConf.select(config, param_space_key):
        raise ValueError(
            "The parameter search space must exist at the provided config node."
        )

    # instantiate any param space target in the search space API
    param_space = copy.deepcopy(OmegaConf.select(config, param_space_key))
    for top, _, _ in walk(OmegaConf.to_container(param_space, resolve=True)):
        val = OmegaConf.select(param_space, top)
        if val and val.get("_target_", None) in TUNE_SEARCH_SPACE_API:
            obj = instantiate(OmegaConf.select(param_space, top), _recursive_=False)
            if isinstance(obj, Domain):
                obj = obj.sample()
            OmegaConf.update(param_space, top, obj, force_add=True, merge=False)

    # a dummy trainable, so we can add an experiment to the search algo
    def trainable(config: dict[str, Any]) -> None:
        pass

    # apply the search algo to sample the parameter space
    search_alg = get_search_alg(config)
    pspace_dict = cast(dict, OmegaConf.to_container(param_space))
    search_alg.add_configurations([Experiment("exp", trainable, config=pspace_dict)])
    sampled_param_space = next(search_alg._iterators[0]).config  # type: ignore
    OmegaConf.update(
        config,
        param_space_key,
        DictConfig(sampled_param_space),
        force_add=True,
        merge=False,
    )


def resolve(
    config: Container,
    exclude: list[str] | None = None,
    sample_params: bool = False,
    param_space_key: str = DEFAULT_PARAM_SPACE_KEY,
) -> None:
    """Resolves an OmegaConf config.

    Optionally excludes the given keylists from resolution.

    Optionally samples the parameter search space; useful for
    testing config resolution without relying on Ray tune to
    resolve the parameter space for a specific trial.
    """
    validate_experiment_config(config, param_space_key=param_space_key)

    config_copy = copy.deepcopy(config)
    with open_dict(config):
        for key in exclude or []:
            OmegaConf.update(config, key, None)

    if sample_params:
        sample_param_space(config, param_space_key)

    OmegaConf.resolve(config)

    with open_dict(config):
        for key in exclude or []:
            unresolved_val = OmegaConf.select(config_copy, key)
            OmegaConf.update(config, key, unresolved_val)


def validate_experiment_config(
    config: Container,
    param_space_key: str = DEFAULT_PARAM_SPACE_KEY,
    trial_key: str = "trial",
) -> None:
    """Validate the experiment config.

    Requires that:
    - all parameter samplers and grid searches are in the parameter
      space node
    - interpolations of nodes in the parameter space and trial nodes
      are only in the trial node
    """

    def in_node(key: str, node_key: str) -> bool:
        return key.startswith(tuple(node_key + x for x in (".", "[]")))

    container = OmegaConf.to_container(config, resolve=False)
    for top, _, obj_keys in walk(container):
        if "_target_" in obj_keys and not in_node(top, param_space_key):
            key = ".".join((top, "_target_"))
            if get(container, key) in TUNE_SEARCH_SPACE_API:
                raise ValueError(
                    "All parameter samplers / grid search must be in the config's "
                    f"parameter space node ({key})."
                )
        for obj_key in obj_keys:
            key = ".".join((top, obj_key))
            val = get(container, key)
            # TODO: update test to catch refs in ray.tune.sample_from
            has_pspace_interp = isinstance(val, str) and f"${{{param_space_key}" in val
            has_trial_interp = isinstance(val, str) and f"${{{trial_key}" in val

            if (has_pspace_interp or has_trial_interp) and not in_node(key, trial_key):
                raise ValueError(
                    "All interpolations of values in the config's parameter space "
                    f"must be in the parameter space node or the trial node ({key})."
                )


def validate_trial_config(config: Container) -> None:
    """Validate the trial node of the given config."""
    train_collate_fn = OmegaConf.select(config, "trial.data.module.collate")
    has_train_metrics = OmegaConf.select(config, "eval.train.metrics")

    if train_collate_fn and has_train_metrics:
        if instantiate(train_collate_fn).train_only:
            raise ValueError(
                "The training collate function cannot be used with training metrics."
            )


def get_param_space_description(
    config: Container, param_space_key: str = DEFAULT_PARAM_SPACE_KEY
) -> str:
    """Returns a one-line description of the config's parameter search space."""
    register_omegaconf_resolvers()

    description = []
    param_space = copy.deepcopy(OmegaConf.select(config, param_space_key))
    param_space_as_container = OmegaConf.to_container(param_space, resolve=True)

    for top, container_keys, obj_keys in walk(param_space_as_container):
        if "_target_" in obj_keys:
            key = ".".join((top, "_target_"))
            if (fn := OmegaConf.select(param_space, key)) in TUNE_SEARCH_SPACE_API:
                field = top.split(".")[-1]
                fn = fn.split(".")[-1]
                values = []
                for container_key in container_keys:
                    key = ".".join((top, container_key))
                    values.append(str(OmegaConf.select(param_space, key)))
                description.append(f"{field}={fn}({','.join(values)})")

    return ",".join(description)


def get_default_storage_path() -> str:
    """Returns the default storage path for experiments."""
    config = load_experiment_config("_defaults_")
    return OmegaConf.select(config, "storage_path")
