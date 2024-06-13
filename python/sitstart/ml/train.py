import os
from pathlib import Path
from typing import Any, Callable

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import Strategy

from sitstart.ml import DEFAULT_CHECKPOINT_ROOT
from sitstart.logging import get_logger
from sitstart.ml.callbacks import LoggerCallback

TrainingModuleCreator = Callable[[dict[str, Any]], pl.LightningModule]
DataModuleCreator = Callable[[dict[str, Any]], pl.LightningDataModule]

logger = get_logger(__name__)


def train(
    config: dict[str, Any],
    training_module_creator: TrainingModuleCreator,
    data_module_creator: DataModuleCreator,
    wandb_enabled: bool = False,
    # local ckpt file, or local or remote ckpt dir if with_ray=True
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

    training_module = training_module_creator(config)
    data_module = data_module_creator(config)

    strategy: str | Strategy = "auto"
    plugins: list[Any] | None = None
    pl_logger = None
    callbacks: list[Callback] = []
    if with_ray:
        strategy = RayDDPStrategy()
        plugins = [RayLightningEnvironment()]
        callbacks.append(RayTrainReportCallback())
    else:
        callbacks.append(LoggerCallback(logger, interval=config["logging_interval"]))
        if wandb_enabled:
            pl_logger = WandbLogger(
                project=config["project"], save_dir=config.get("storage_path", None)
            )

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
        log_every_n_steps=config["logging_interval"],
        enable_checkpointing=not with_ray,
        default_root_dir=config.get("storage_path", None) if not with_ray else None,
    )

    if with_ray:
        trainer = prepare_trainer(trainer)

        if ckpt := ray.train.get_checkpoint():
            assert (
                ckpt_path is None
            ), "Cannot load both trial- and user-specified checkpoints."
        elif ckpt_path:
            ckpt = ray.train.Checkpoint(ckpt_path)
        if ckpt:
            ckpt_dir = ckpt.to_directory(f"{DEFAULT_CHECKPOINT_ROOT}/ckpt")
            ckpt_path = Path(ckpt_dir) / "checkpoint.ckpt"

    if ckpt_path:
        logger.info(f"Loading checkpoint from: {ckpt_path}")

    trainer.fit(
        training_module,
        datamodule=data_module,
        ckpt_path=ckpt_path,  # type: ignore
    )


def test(
    config,
    training_module_creator: TrainingModuleCreator,
    data_module_creator: DataModuleCreator,
) -> None:
    training_module = training_module_creator(config)
    data_module = data_module_creator(config)

    trainer = pl.Trainer(
        devices="auto",
        accelerator="gpu" if config.get("use_gpu") else "cpu",
        enable_progress_bar=False,
        default_root_dir=config.get("storage_path", None),
    )
    output = trainer.test(training_module, datamodule=data_module)
    logger.info(output)
