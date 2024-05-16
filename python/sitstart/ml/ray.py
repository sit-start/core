import copy
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.train import FailureConfig, RunConfig
from ray.train.torch import TorchConfig, TorchTrainer
from ray.tune import TuneConfig, Tuner
from ray.tune.experiment.trial import Trial

from sitstart.logging import get_logger
from sitstart.ml.train import DataModuleCreator, TrainingModuleCreator, train
from sitstart.ml.training_module import TrainingModule
from sitstart.scm.git.repo_state import RepoState, get_repo
from sitstart.util.container import walk
from sitstart.util.hydra import instantiate, register_omegaconf_resolvers
from sitstart.util.string import to_str

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

register_omegaconf_resolvers()

logger = get_logger(__name__)


def _get_project_name(config: DictConfig) -> str:
    return config.get("name", Path(sys.argv[0]).stem)


def _get_group_name() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _get_trial_dirname(trial: Trial) -> str:
    return trial.trial_id


def _get_trial_name(trial: Trial, incl_params: bool = False) -> str:
    # trial_id is of the form <run_id>_<trial_num>. The unique run_id
    # distinguishes trials from different runs in the same group, and
    # the trial_num, from different trials in the same run. This is
    # particularly important if we're allowing multiple runs to use
    # the same group when those runs may be different.
    if not incl_params:
        return trial.trial_id

    # trial.evaluated_params includes the 'path' of the variable in the
    # config, e.g., train_loop_config/lr=1e-1. We drop the path here
    # unless there's a collision between variable names.
    var_to_params = {}
    for k, v in trial.evaluated_params.items():
        var = k.split("/")[-1]
        var_to_params.setdefault(var, []).append([k, v])
    params = {}
    for k, v in var_to_params.items():
        if len(v) == 1:
            params[k] = v[0][1]
        else:
            for k0, v0 in v:
                params[k0] = v0

    param_str = to_str(
        params, precision=4, list_sep=",", dict_sep=",", dict_kv_sep="=", use_repr=False
    )[1:-1]
    trial_id = trial.trial_id
    return trial_id if not param_str else f"{trial_id}({param_str})"


def _get_repo_state_and_add_to_config(config: DictConfig) -> RepoState | None:
    if not config.save_repo_state:
        return None

    driver = sys.argv[0]
    repo = get_repo(__file__)
    if "PYTEST_CURRENT_TEST" not in os.environ and repo != get_repo(driver):
        raise NotImplementedError(
            f"Driver {driver!r} must be in this repo ({repo.working_dir!r})"
        )

    repo_state = RepoState.from_repo(repo)
    train_loop_config = config.param_space.train_loop_config

    OmegaConf.update(
        train_loop_config, "repo_state_summary", repo_state.summary, force_add=True
    )

    return repo_state


def _get_callbacks(config: DictConfig) -> list:
    callbacks = []
    if config["wandb"]["enabled"]:
        callbacks.append(
            WandbLoggerCallback(
                project=_get_project_name(config), group=_get_group_name()
            )
        )
    return callbacks


def _get_module_creators(
    config: DictConfig,
) -> tuple[TrainingModuleCreator, DataModuleCreator]:
    def get_trial_and_train_loop_configs(train_loop_config: dict[str, Any]):
        # update the config with the trial's train_loop_config, and
        # instantiate/resolve any deferred config nodes
        register_omegaconf_resolvers()
        input_config = copy.deepcopy(config)
        input_config.param_space.train_loop_config = train_loop_config
        trial_config = instantiate(input_config, _defer_=False)

        return trial_config, cast(
            dict[str, Any],
            OmegaConf.to_container(trial_config.param_space.train_loop_config),
        )

    def training_module_creator(train_loop_conf: dict[str, Any]) -> pl.LightningModule:
        trial_conf, train_loop_conf = get_trial_and_train_loop_configs(train_loop_conf)
        return TrainingModule(train_loop_conf, model=trial_conf.model.model())

    def data_module_creator(train_loop_conf: dict[str, Any]) -> pl.LightningDataModule:
        trial_conf, _ = get_trial_and_train_loop_configs(train_loop_conf)
        return trial_conf.data.module()

    return training_module_creator, data_module_creator


def _get_ray_trainer(
    config: DictConfig,
    repo_state: RepoState | None = None,
    tuning: bool = False,
):
    run_config = RunConfig(
        name="_".join([_get_project_name(config), _get_group_name()]),
        checkpoint_config=config.checkpoint(),
        storage_path=config.storage_path,
        callbacks=_get_callbacks(config),
        log_to_file=True,  # NOTE: doesn't work in Jupyter notebook
        failure_config=FailureConfig(max_failures=3),
    )

    def train_loop_per_worker(train_loop_config: dict[str, Any]) -> None:
        train(
            train_loop_config,
            *_get_module_creators(config),
            wandb_enabled=config.wandb.enabled,
            with_ray=True,
        )

    param_space = {} if tuning else instantiate(config.param_space, _convert_="full")

    return TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        run_config=run_config,
        **cast(dict[str, Any], param_space),
        metadata=dict(repo_state=repo_state.__dict__) if repo_state else None,
        torch_config=TorchConfig(backend=config.torch.distributed_backend),
    )


def _validate_config(config: DictConfig) -> None:
    for top, _, obj_keys in walk(OmegaConf.to_container(config, resolve=False)):
        if "_target_" in obj_keys and not top.startswith("param_space."):
            class_name = OmegaConf.select(config, top + "._target_")
            if class_name in TUNE_SEARCH_SPACE_API:
                raise ValueError(
                    "All parameter search spaces must be under "
                    "`param_space` in the config."
                )


def train_with_ray(config: DictConfig) -> None:
    repo_state = _get_repo_state_and_add_to_config(config)
    _validate_config(config)
    config = instantiate(config)

    trainer = _get_ray_trainer(config, repo_state=repo_state)

    result = trainer.fit()
    print(f"Training result: {result}")


def tune_with_ray(config: DictConfig) -> None:
    repo_state = _get_repo_state_and_add_to_config(config)
    _validate_config(config)
    logger.info(f"Tuning with config:\n{OmegaConf.to_yaml(config)}")
    config = instantiate(config)

    trainer = _get_ray_trainer(config, repo_state=repo_state, tuning=True)
    trial_name_args = dict(incl_params=config.tune.long_trial_names)

    tuner = Tuner(
        trainer,
        tune_config=TuneConfig(
            metric=config.eval.metric,
            mode=config.eval.mode,
            num_samples=config.tune.num_samples,
            scheduler=config.tune.scheduler,
            trial_name_creator=lambda trial: _get_trial_name(trial, **trial_name_args),
            trial_dirname_creator=_get_trial_dirname,
        ),
        param_space=instantiate(config.param_space, _convert_="full"),
    )

    results = tuner.fit()

    best_result = results.get_best_result(config.eval.metric, config.eval.mode)
    logger.info(f"Best trial config: {best_result.config}")
    if not best_result.metrics:
        logger.warning("No metrics found in best trial result")
        return
    logger.info(f"Best trial final validation loss: {best_result.metrics['val_loss']}")
    logger.info(
        f"Best trial final validation accuracy: {best_result.metrics['val_acc']}"
    )
