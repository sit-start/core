import copy
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, cast

import pytorch_lightning as pl
import wandb
import yaml
from hydra.utils import instantiate
from omegaconf import Container, DictConfig, ListConfig, OmegaConf, open_dict
from ray.tune.experiment import Experiment
from ray.tune.search import SearchAlgorithm, Searcher, SearchGenerator, create_searcher
from ray.tune.search.sample import Domain

import sitstart.util.hydra
from sitstart.logging import get_logger
from sitstart.ml.experiments import CONFIG_ROOT, HYDRA_VERSION_BASE
from sitstart.util.container import walk
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

logger = get_logger(__name__)


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


def load_experiment_config(
    name: str, overrides: list[str] | None = None, config_path: str | None = None
) -> Container:
    """Load an experiment config. For testing/debugging."""
    overrides = overrides or []
    config_path = config_path or CONFIG_ROOT

    base_config = yaml.safe_load(Path(f"{config_path}/{name}.yaml").read_text())
    if "name" not in base_config:
        if not any(override.startswith("name=") for override in overrides):
            overrides += ["name=" + name]

    return load_config(
        name,
        config_path=config_path,
        overrides=overrides,
        version_base=HYDRA_VERSION_BASE,
    )


def apply_experiment_config_overrides(
    config: Container, overrides: list[str]
) -> Container:
    """Apply overrides to the given experiment config."""
    with TemporaryDirectory() as temp_dir:
        config_name = OmegaConf.select(config, "name")
        temp_path = Path(temp_dir) / f"{config_name}.yaml"
        temp_path.write_text(OmegaConf.to_yaml(config))
        return load_config(config_name, overrides=overrides, config_path=temp_dir)


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
    param_space_as_container = OmegaConf.to_container(param_space, resolve=True)

    for top, _, _ in walk(param_space_as_container, topdown=False):
        val = OmegaConf.select(param_space, top)
        if not isinstance(val, DictConfig):
            continue
        if val.get("_target_", None) not in TUNE_SEARCH_SPACE_API:
            continue
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


class TrialResolution(str, Enum):
    RESOLVE = "resolve"
    EXCLUDE = "exclude"
    SAMPLE = "sample"

    def __str__(self) -> str:
        return self.value


def resolve(
    config: Container,
    resolve_trial: str | TrialResolution = "resolve",
    param_space_key: str = DEFAULT_PARAM_SPACE_KEY,
    trial_key: str = "trial",
) -> None:
    """Resolve an OmegaConf config in-place.

    Args:
        config: The config to resolve.
        param_space_key: The key for the parameter search space node.
        trial_key: The key for the trial node.
        resolve_trial: How to handle the trial node's parameters:
            - "resolve": Resolve the trial node's parameters.
            - "exclude": Exclude the trial node's parameters from resolution.
            - "sample": Sample the parameter search space for the trial node.
    """
    register_omegaconf_resolvers()

    resolve_trial = str(resolve_trial)
    validate_experiment_config(
        config, param_space_key=param_space_key, trial_key=trial_key
    )

    if resolve_trial == "sample":
        sample_param_space(config, param_space_key)

    for root_key in config:
        if root_key == trial_key and resolve_trial == "exclude":
            continue
        node = OmegaConf.select(config, root_key)
        if isinstance(node, Container):
            OmegaConf.resolve(node)
        else:
            # ensure we resolve any root nodes that are simple interpolations
            OmegaConf.update(config, root_key, OmegaConf.select(config, root_key))


def validate_experiment_config(
    config: Container,
    param_space_key: str = DEFAULT_PARAM_SPACE_KEY,
    trial_key: str = "trial",
) -> None:
    """Validate the given experiment config.

    Requires the following:
    - The trial key must specify a root config node.
    - No direct or indirect dependencies on the parameter space
      node appear outside the trial node.
    - All parameter samplers and grid searches must be in the
      parameter space node.

    Raises a ValueError if any of the above conditions are not met.
    """
    if any(c in trial_key for c in ".[]"):
        raise ValueError("Trial key must specify a root config node.")
    if not isinstance(config, DictConfig):
        raise ValueError("Config must be a `DictConfig`.")

    config = copy.deepcopy(config)
    with open_dict(config):
        _ = config.pop(param_space_key, None)

    dict_types, list_types = [dict, DictConfig], [list, ListConfig]
    container_types: Any = dict(dict_types=dict_types, list_types=list_types)

    for root_key in [str(k) for k in config]:
        if root_key == trial_key:
            continue
        # check for resolution failures in general, and in particular
        # those due to transitive dependencies on the parameter space node
        try:
            node = OmegaConf.select(config, root_key)
            if not isinstance(node, Container):
                continue
            OmegaConf.resolve(node)
        except Exception as e:
            raise ValueError(
                f"Failed to resolve config node {root_key!r} without the "
                "parameter space node, possibly due to a transitive "
                f"dependency on the parameter space node."
            ) from e

        # check for param samplers / grid searches outside the param space node
        for key, _, obj_keys in walk(node, **container_types):
            if "_target_" in obj_keys:
                if OmegaConf.select(node, key + "._target_") in TUNE_SEARCH_SPACE_API:
                    raise ValueError(
                        "All parameter samplers and grid searches must be "
                        f"in the config's parameter space node ({key})."
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


def get_lightning_modules_from_config(
    config: DictConfig,
    train_loop_config: dict[str, Any] | None = None,
    num_workers: int = 1,
) -> tuple[pl.LightningDataModule, pl.LightningModule]:
    """Get the data and training modules from the given main and train loop configs."""
    register_omegaconf_resolvers()
    config_for_trial = copy.deepcopy(config)
    if train_loop_config:
        config_for_trial.param_space.train_loop_config = train_loop_config
    validate_trial_config(config_for_trial)
    trial_config = instantiate(config_for_trial.trial)

    batch_size = trial_config.data.batch_size // num_workers
    logger.info(
        f"Batch size: {trial_config.data.batch_size} (global), {batch_size} (worker)"
    )
    data_module = trial_config.data.module(batch_size=batch_size)

    training_module = trial_config.training_module(
        loss_fn=trial_config.loss_fn,
        lr_scheduler=trial_config.lr_scheduler,
        test_metrics=instantiate(config_for_trial.eval.test.metrics),
        train_metrics=instantiate(config_for_trial.eval.train.metrics),
        model=trial_config.model.module,
        optimizer=trial_config.optimizer,
    )

    return data_module, training_module


def get_default_storage_path() -> str:
    """Returns the default storage path for experiments."""
    config = load_experiment_config("_defaults_")
    return OmegaConf.select(config, "storage_path")
