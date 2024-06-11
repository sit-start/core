import copy
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, cast

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.train import FailureConfig, RunConfig, get_context
from ray.train.torch import TorchConfig, TorchTrainer
from ray.tune import TuneConfig, Tuner
from ray.tune.experiment.trial import Trial

from sitstart.logging import get_logger
from sitstart.ml.experiments.util import get_search_alg, resolve
from sitstart.ml.train import train
from sitstart.scm.git.repo_state import RepoState, get_repo
from sitstart.util.hydra import register_omegaconf_resolvers
from sitstart.util.string import to_str

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


def _get_train_loop_per_worker(config: DictConfig) -> Callable[[dict[str, Any]], None]:
    def train_loop_per_worker(train_loop_config: dict[str, Any]) -> None:
        register_omegaconf_resolvers()
        config_for_trial = copy.deepcopy(config)
        config_for_trial.param_space.train_loop_config = train_loop_config
        trial_config = instantiate(config_for_trial.trial)

        training_module = trial_config.training_module(
            loss_fn=trial_config.loss_fn,
            lr_scheduler=trial_config.lr_scheduler,
            metrics=trial_config.metrics,
            model=trial_config.model.module,
            optimizer=trial_config.optimizer,
        )

        batch_size = trial_config.data.batch_size
        worker_batch_size = batch_size
        if train_context := get_context():
            worker_batch_size = batch_size // train_context.get_world_size()
            logger.info(
                f"Batch size: {batch_size} (global), {worker_batch_size} (worker)"
            )
        data_module = trial_config.data.module(batch_size=worker_batch_size)

        train(
            data_module=data_module,
            training_module=training_module,
            float32_matmul_precision=config.float32_matmul_precision,
            logging_interval=config.logging_interval,
            max_num_epochs=config.max_num_epochs,
            project_name=_get_project_name(config),
            seed=config.seed,
            storage_path=config.storage_path,
            use_gpu=config.param_space.scaling_config.use_gpu,
            wandb_enabled=config.wandb.enabled,
            with_ray=True,
        )

    return train_loop_per_worker


def _get_ray_trainer(
    config: DictConfig,
    repo_state: RepoState | None = None,
    tuning: bool = False,
):
    run_config = RunConfig(
        name="_".join([_get_project_name(config), _get_group_name()]),
        checkpoint_config=instantiate(config.checkpoint),
        storage_path=config.storage_path,
        callbacks=_get_callbacks(config),
        log_to_file=True,  # NOTE: doesn't work in Jupyter notebook
        failure_config=FailureConfig(max_failures=3),
    )

    param_space = {} if tuning else instantiate(config.param_space)
    metadata = dict(
        config=OmegaConf.to_container(config, resolve=False),
        repo_state=repo_state.__dict__ if repo_state else None,
    )

    return TorchTrainer(
        train_loop_per_worker=_get_train_loop_per_worker(config),
        run_config=run_config,
        **cast(dict[str, Any], param_space),
        metadata=metadata,
        torch_config=TorchConfig(backend=config.torch.distributed_backend),
    )


def train_with_ray(config: DictConfig) -> None:
    repo_state = _get_repo_state_and_add_to_config(config)
    logger.info(f"Training with config:\n{OmegaConf.to_yaml(config)}")
    resolve(config)

    trainer = _get_ray_trainer(config, repo_state=repo_state)

    result = trainer.fit()
    print(f"Training result: {result}")


def tune_with_ray(config: DictConfig) -> None:
    repo_state = _get_repo_state_and_add_to_config(config)
    logger.info(f"Tuning with config:\n{OmegaConf.to_yaml(config)}")
    # we postpone resolution of the trial node until the Tuner has
    # selected values from the parameter space for the trial.
    resolve(config, exclude=["trial"])

    trainer = _get_ray_trainer(config, repo_state=repo_state, tuning=True)
    trial_name_args = dict(incl_params=config.tune.long_trial_names)

    tuner = Tuner(
        trainer,
        tune_config=TuneConfig(
            num_samples=config.tune.num_samples,
            scheduler=instantiate(config.tune.scheduler),
            trial_name_creator=lambda trial: _get_trial_name(trial, **trial_name_args),
            trial_dirname_creator=_get_trial_dirname,
            search_alg=get_search_alg(config),
        ),
        param_space=instantiate(config.param_space),
    )

    results = tuner.fit()

    best_result = results.get_best_result(
        config.eval.select.metric, config.eval.select.mode
    )
    logger.info(f"Best trial config: {best_result.config}")
    if not best_result.metrics:
        logger.warning("No metrics found in best trial result")
        return
    logger.info(f"Best trial final validation loss: {best_result.metrics['val_loss']}")
    logger.info(
        f"Best trial final validation accuracy: {best_result.metrics['val_acc']}"
    )
