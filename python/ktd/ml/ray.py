import sys
from pathlib import Path
from datetime import datetime
from typing import Callable

import pytorch_lightning as pl
from ktd.logging import get_logger
from ktd.util.git import get_repo, RepoState
from ktd.util.string import to_str
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.train import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer, TorchConfig
from ray.tune import TuneConfig, Tuner
from ray.tune.experiment.trial import Trial
from ray.tune.schedulers import ASHAScheduler

from .train import DataModuleFactory, TrainingModuleFactory, train

CHECKPOINT_STORAGE_PATH = "s3://ktd-ray/runs"

logger = get_logger(__name__)


def _get_project_name(config: dict) -> str:
    return config.get("project_name", Path(sys.argv[0]).stem)


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


def _get_repo_state_and_add_to_config(config: dict) -> RepoState | None:
    if not config.get("save_repo_state", True):
        return None

    driver = sys.argv[0]
    if (repo := get_repo(__file__)) != get_repo(driver):
        raise NotImplementedError(
            f"Driver {driver!r} must be in this repo ({repo.working_dir!r})"
        )

    repo_state = RepoState.from_repo(repo)
    config["train"]["repo_state"] = repo_state.summary

    return repo_state


def _get_checkpoint_config(config: dict) -> CheckpointConfig:
    return CheckpointConfig(
        num_to_keep=config["checkpoint"]["num_to_keep"],
        checkpoint_score_attribute=config["checkpoint"]["checkpoint_score_attribute"],
        checkpoint_score_order=config["checkpoint"]["checkpoint_score_order"],
    )


def _get_callbacks(config: dict) -> list:
    callbacks = []
    if config["wandb"]["enabled"]:
        callbacks.append(
            WandbLoggerCallback(
                project=_get_project_name(config), group=_get_group_name()
            )
        )
    return callbacks


def _get_scaling_config(config: dict) -> ScalingConfig:
    return ScalingConfig(
        num_workers=config["scale"]["num_workers"],
        use_gpu=config["scale"]["use_gpu"],
        resources_per_worker=config["scale"]["resources_per_worker"],
    )


def _get_ray_trainer(
    config: dict,
    training_module_factory: TrainingModuleFactory,
    data_module_factory: DataModuleFactory,
    repo_state: RepoState | None = None,
    scaling_config: ScalingConfig | None = None,
):
    run_config = RunConfig(
        name="_".join([_get_project_name(config), _get_group_name()]),
        checkpoint_config=_get_checkpoint_config(config),
        storage_path=CHECKPOINT_STORAGE_PATH,
        callbacks=_get_callbacks(config),
        log_to_file=True,  # NOTE: doesn't work in Jupyter notebook
        failure_config=FailureConfig(max_failures=3),
    )

    def train_loop_per_worker(train_loop_config: dict) -> None:
        train(
            train_loop_config,
            training_module_factory,
            data_module_factory,
            wandb_enabled=config["wandb"]["enabled"],
            with_ray=True,
        )

    return TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=config["train"],
        run_config=run_config,
        scaling_config=scaling_config,
        metadata=dict(repo_state=repo_state.__dict__) if repo_state else None,
        torch_config=TorchConfig(backend=config["torch"]["backend"]),
    )


def train_with_ray(
    config: dict,
    training_module_factory: TrainingModuleFactory,
    data_module_factory: DataModuleFactory,
) -> None:
    repo_state = _get_repo_state_and_add_to_config(config)

    trainer = _get_ray_trainer(
        config,
        training_module_factory,
        data_module_factory,
        repo_state=repo_state,
        scaling_config=_get_scaling_config(config),
    )
    result = trainer.fit()
    print(f"Training result: {result}")


def tune_with_ray(
    config: dict,
    training_module_factory: Callable[[dict], pl.LightningModule],
    data_module_factory: Callable[[dict], pl.LightningDataModule],
) -> None:
    repo_state = _get_repo_state_and_add_to_config(config)

    logger.info(f"Tuning with config: {config}")

    trainer = _get_ray_trainer(
        config, training_module_factory, data_module_factory, repo_state=repo_state
    )

    max_num_epochs = config["schedule"]["max_num_epochs"]
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=config["schedule"]["grace_period"],
        reduction_factor=2,
    )

    logger.info(
        f"Using {max_num_epochs=} from the scheduling config in the training loop"
    )
    _ = config["train"]["max_num_epochs"] = max_num_epochs

    tuner = Tuner(
        trainer,
        tune_config=TuneConfig(
            metric="val_loss",  # TODO: val_loss/min or val_acc/max?
            mode="min",
            num_samples=config["tune"]["num_samples"],
            scheduler=scheduler,
            trial_name_creator=lambda trial: _get_trial_name(
                trial, incl_params=config["tune"]["long_trial_names"]
            ),
            trial_dirname_creator=_get_trial_dirname,
        ),
        param_space={
            "scaling_config": _get_scaling_config(config),
            "train_loop_config": config["train"],
        },
    )

    results = tuner.fit()

    best_result = results.get_best_result("val_loss", "min")
    logger.info(f"Best trial config: {best_result.config}")
    if not best_result.metrics:
        logger.warning("No metrics found in best trial result")
        return
    logger.info(f"Best trial final validation loss: {best_result.metrics['val_loss']}")
    logger.info(
        f"Best trial final validation accuracy: {best_result.metrics['val_acc']}"
    )
