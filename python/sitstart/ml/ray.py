import os
import sys
from typing import Any, Callable, cast

import ray
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.train import FailureConfig, RunConfig, get_context
from ray.train.torch import TorchConfig, TorchTrainer
from ray.tune import TuneConfig, Tuner

from sitstart.logging import get_logger
from sitstart.ml.experiments.name import (
    get_group_name,
    get_project_name,
    get_run_name,
    get_trial_dirname,
    get_trial_name,
)
from sitstart.ml.experiments.restore import (
    get_checkpoint_file_path,
    get_checkpoint_from_config,
    to_local_checkpoint,
)
from sitstart.ml.experiments.util import (
    get_lightning_modules_from_config,
    get_search_alg,
    register_omegaconf_resolvers,
    resolve,
)
from sitstart.ml.train import train
from sitstart.scm.git.repo_state import RepoState, get_repo

register_omegaconf_resolvers()

logger = get_logger(__name__)


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
                project=get_project_name(config), group=get_group_name()
            )
        )
    return callbacks


def _get_train_loop_per_worker(
    config: DictConfig, with_ray: bool = True
) -> Callable[[dict[str, Any]], None]:
    def train_loop_per_worker(train_loop_config: dict[str, Any]) -> None:
        train_context = get_context()
        num_workers = train_context.get_world_size() if train_context else 1
        data_module, training_module = get_lightning_modules_from_config(
            config, train_loop_config, num_workers=num_workers
        )

        ckpt = get_checkpoint_from_config(config)
        if ray_ckpt := ray.train.get_checkpoint():
            if ckpt:
                logger.warning(
                    "Ray is resuming from the session's last checkpoint."
                    "This overrides the user-specified checkpoint."
                )
            ckpt = ray_ckpt
        ckpt_path = (
            get_checkpoint_file_path(to_local_checkpoint(ckpt)) if ckpt else None
        )

        train(
            data_module=data_module,
            training_module=training_module,
            ckpt_path=ckpt_path,
            float32_matmul_precision=config.float32_matmul_precision,
            gradient_clip_algorithm=config.gradient_clip.algorithm,
            gradient_clip_val=config.gradient_clip.value,
            logging_interval=config.logging_interval,
            max_num_epochs=config.max_num_epochs,
            num_sanity_val_steps=config.num_sanity_val_steps,
            project_name=get_project_name(config),
            seed=config.seed,
            storage_path=config.storage_path,
            use_gpu=config.param_space.scaling_config.use_gpu,
            wandb_enabled=config.wandb.enabled,
            with_ray=with_ray,
        )

    return train_loop_per_worker


def _get_ray_trainer(
    config: DictConfig,
    repo_state: RepoState | None = None,
    tuning: bool = False,
):
    run_config = RunConfig(
        name=get_run_name(get_project_name(config)),
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
    resolve(config, resolve_trial="exclude")

    trainer = _get_ray_trainer(config, repo_state=repo_state, tuning=True)
    trial_name_args = dict(incl_params=config.tune.long_trial_names)

    tuner = Tuner(
        trainer,
        tune_config=TuneConfig(
            num_samples=config.tune.num_samples,
            scheduler=instantiate(config.tune.scheduler),
            trial_name_creator=lambda trial: get_trial_name(trial, **trial_name_args),
            trial_dirname_creator=get_trial_dirname,
            search_alg=get_search_alg(config),
        ),
        param_space=instantiate(config.param_space),
    )

    results = tuner.fit()

    metric = config.eval.select.metric
    best_result = results.get_best_result(metric, config.eval.select.mode)

    logger.info(f"Best trial config: {best_result.config}")
    if not best_result.metrics:
        logger.warning("No metrics found in best trial result")
        return
    logger.info(f"Best trial final val_loss: {best_result.metrics['val_loss']}")
    logger.info(f"Best trial final {metric!r}: {best_result.metrics[metric]}")
