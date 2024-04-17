import os
from pathlib import Path
from typing import Any, Callable

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import Strategy

from ktd.logging import get_logger

from .callbacks import LoggerCallback

TrainingModuleFactory = Callable[[dict[str, Any]], pl.LightningModule]
DataModuleFactory = Callable[[dict[str, Any]], pl.LightningDataModule]

logger = get_logger(__name__)


def train(
    config,
    training_module_factory: TrainingModuleFactory,
    data_module_factory: DataModuleFactory,
    wandb_enabled: bool = False,
    ckpt_path: str | os.PathLike[str] | None = None,
    **kwargs: Any,
) -> None:
    logger.info(f"Training with config: {config}")

    if config.get("seed", None):
        pl.seed_everything(config["seed"])

    with_ray = kwargs.get("with_ray", False)
    if with_ray:
        import ray
        import ray.train
        from ray.train.lightning import (
            RayDDPStrategy,
            RayLightningEnvironment,
            RayTrainReportCallback,
            prepare_trainer,
        )

    torch.set_float32_matmul_precision(config["float32_matmul_precision"])

    if with_ray and (train_context := ray.train.get_context()):
        config = config.copy()
        batch_size = config["batch_size"]
        config["batch_size"] //= train_context.get_world_size()
        logger.info(
            "Using per-worker and global batch sizes of "
            f"{config['batch_size']}, {batch_size}, resp."
        )

    training_module = training_module_factory(config)
    data_module = data_module_factory(config)

    strategy: str | Strategy = "auto"
    plugins: list[Any] | None = None
    pl_logger = None
    callbacks: list[Callback] = []
    if with_ray:
        strategy = RayDDPStrategy()
        plugins = [RayLightningEnvironment()]
        callbacks.append(RayTrainReportCallback())
    else:
        # TODO: needs testing
        callbacks.append(LoggerCallback(logger, interval=config["log_every_n_steps"]))
        if wandb_enabled:
            pl_logger = WandbLogger(project=config["project"])
            pl_logger.watch(training_module, log="all")

    # TODO: address warning re: missing tensorboard logging directory
    trainer = pl.Trainer(
        devices="auto",
        accelerator="gpu" if config.get("use_gpu") else "cpu",
        strategy=strategy,
        plugins=plugins,
        callbacks=callbacks,
        logger=pl_logger,
        enable_progress_bar=False,
        max_epochs=config["max_num_epochs"],
        log_every_n_steps=config["log_every_n_steps"],
        enable_checkpointing=not with_ray,
    )

    if with_ray:
        trainer = prepare_trainer(trainer)

    if with_ray and (ckpt := ray.train.get_checkpoint()):
        assert (
            ckpt_path is None
        ), "Cannot load both trial- and user-specified checkpoints."
        with ckpt.as_directory() as ckpt_dir:
            ckpt_path = Path(ckpt_dir) / "checkpoint.ckpt"
            trainer.fit(training_module, datamodule=data_module, ckpt_path=ckpt_path)
    else:
        trainer.fit(
            training_module,
            datamodule=data_module,
            ckpt_path=ckpt_path,  # type: ignore
        )


# TODO: flesh out ktd.ml.train.test
def test(
    config,
    training_module_factory: TrainingModuleFactory,
    data_module_factory: DataModuleFactory,
) -> None:
    training_module = training_module_factory(config)
    data_module = data_module_factory(config)

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        enable_progress_bar=False,
        callbacks=[LoggerCallback(logger, interval=config["log_every_n_steps"])],
    )
    output = trainer.test(training_module, datamodule=data_module)
    logger.info(output)
