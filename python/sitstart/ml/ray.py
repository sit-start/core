import copy
import os
import sys
from pathlib import Path
from typing import Any, Callable, cast

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.train import Checkpoint, FailureConfig, Result, RunConfig, get_context
from ray.train.torch import TorchConfig, TorchTrainer
from ray.tune import TuneConfig, Tuner

from sitstart.aws.util import update_aws_env
from sitstart.logging import get_logger
from sitstart.ml import DEFAULT_CHECKPOINT_ROOT
from sitstart.ml.experiments.name import (
    get_group_name,
    get_project_name,
    get_trial_dirname,
    get_trial_name,
)
from sitstart.ml.experiments.util import (
    get_search_alg,
    register_omegaconf_resolvers,
    resolve,
    validate_trial_config,
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


def _get_ckpt_path(cfg: DictConfig) -> str | None:
    if cfg.restore.checkpoint_path:
        return cfg.restore.checkpoint_path
    if not cfg.restore.run.group or not cfg.restore.run.trial_id:
        return None

    select = cfg.restore.run.get("select", "last")
    if select not in ("best", "last"):
        raise ValueError(
            f"Invalid restore.run.select value; should be 'best' or 'last' ({select})."
        )

    if cfg.storage_path.startswith("s3://"):
        update_aws_env()

    project_path = f"{cfg.storage_path}/{get_project_name(cfg)}"
    run_path = f"{project_path}_{cfg.restore.run.group}/{cfg.restore.run.trial_id}"
    result = Result.from_path(run_path)

    last_ckpt = result.checkpoint
    best_ckpt = result.get_best_checkpoint(cfg.eval.select.metric, cfg.eval.select.mode)
    ckpt = last_ckpt if select == "last" else best_ckpt

    fs_prefix = {"s3": "s3://", "gcs": "gs://"}
    if ckpt:
        fs = ckpt.filesystem
        prefix = fs_prefix.get(fs.type_name, "") if fs else ""
        return prefix + ckpt.path

    return None


def _get_train_loop_per_worker(
    config: DictConfig, with_ray: bool = True
) -> Callable[[dict[str, Any]], None]:
    def train_loop_per_worker(train_loop_config: dict[str, Any]) -> None:
        register_omegaconf_resolvers()
        config_for_trial = copy.deepcopy(config)
        config_for_trial.param_space.train_loop_config = train_loop_config
        validate_trial_config(config_for_trial)
        trial_config = instantiate(config_for_trial.trial)

        training_module = trial_config.training_module(
            loss_fn=trial_config.loss_fn,
            lr_scheduler=trial_config.lr_scheduler,
            test_metrics=instantiate(config_for_trial.eval.test.metrics),
            train_metrics=instantiate(config_for_trial.eval.train.metrics),
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
            ckpt_path=get_local_checkpoint_path(config),
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
        name="_".join([get_project_name(config), get_group_name()]),
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


def get_local_checkpoint_path(config: DictConfig) -> str | None:
    if (ckpt_path := _get_ckpt_path(config)) is None:
        return None
    ckpt = Checkpoint(ckpt_path)
    ckpt_dir = ckpt.to_directory(f"{DEFAULT_CHECKPOINT_ROOT}/ckpt")
    return str(Path(ckpt_dir) / "checkpoint.ckpt")


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
