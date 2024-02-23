from typing import Callable

import pytorch_lightning as pl
from ktd.logging import get_logger
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.train import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune import TuneConfig, Tuner
from ray.tune.schedulers import ASHAScheduler

from .train import train, TrainingModuleFactory, DataModuleFactory

logger = get_logger(__name__, format="simple")


def _get_ray_trainer(
    config: dict,
    training_module_factory: TrainingModuleFactory,
    data_module_factory: DataModuleFactory,
    num_workers: int = 4,
    use_gpu: bool = True,
):
    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)

    run_config = RunConfig(
        # name=",  # TODO: add useful/descriptive trial name
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val_loss",  # "val_acc"
            checkpoint_score_order="min",  # "max"
        ),
        storage_path="s3://ktd-ray/runs",
        callbacks=(
            [WandbLoggerCallback(project=config["project"])]
            if config.get("log_to_wandb", False)
            else []
        ),
        log_to_file=True,  # NOTE: doesn't work in Jupyter notebook
        failure_config=FailureConfig(max_failures=3),
    )

    def train_loop_per_worker(config: dict) -> None:
        train(config, training_module_factory, data_module_factory, with_ray=True)

    return TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=config,
        run_config=run_config,
        scaling_config=scaling_config,
    )


def train_with_ray(
    config: dict,
    training_module_factory: TrainingModuleFactory,
    data_module_factory: DataModuleFactory,
) -> None:
    trainer = _get_ray_trainer(
        config,
        training_module_factory,
        data_module_factory,
        num_workers=config.get("num_workers", config["num_workers_per_trial"]),
        use_gpu=config["use_gpu"],
    )
    result = trainer.fit()
    print(f"Training result: {result}")


def tune_with_ray(
    config: dict,
    training_module_factory: Callable[[dict], pl.LightningModule],
    data_module_factory: Callable[[dict], pl.LightningDataModule],
) -> None:
    trainer = _get_ray_trainer(
        config,
        training_module_factory,
        data_module_factory,
        num_workers=config["num_workers_per_trial"],
        use_gpu=config["use_gpu"],
    )

    scheduler = ASHAScheduler(
        max_t=config["max_num_epochs"],
        grace_period=config.get("min_num_epochs", 1),
        reduction_factor=2,
    )

    tuner = Tuner(
        trainer,
        tune_config=TuneConfig(
            metric="val_loss",  # TODO: val_loss/min or val_acc/max?
            mode="min",
            num_samples=config["num_hparam_samples"],
            scheduler=scheduler,
        ),
        param_space={
            "train_loop_config": config
        },  # TODO - we prob want to whittle this down to actual hparams we're varying
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
