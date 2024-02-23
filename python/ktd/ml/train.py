import os
from pathlib import Path
from typing import Any, Callable

import pytorch_lightning as pl
from ktd.logging import get_logger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import Strategy

from .callbacks import LoggerCallback

TrainingModuleFactory = Callable[[dict[str, Any]], pl.LightningModule]
DataModuleFactory = Callable[[dict[str, Any]], pl.LightningDataModule]


def train(
    config,
    training_module_factory: TrainingModuleFactory,
    data_module_factory: DataModuleFactory,
    ckpt_path: str | os.PathLike[str] | None = None,
    **kwargs: Any,
) -> None:
    with_ray = kwargs.get("with_ray", False)
    if with_ray:
        import ray
        from ray.train.lightning import (
            RayDDPStrategy,
            RayLightningEnvironment,
            RayTrainReportCallback,
            prepare_trainer,
        )

    text_logger = get_logger(__name__, format="simple" if with_ray else "glog")
    text_logger.info(f"{config=}")

    # Set NCCL and Torch distributed logging levels for debugging
    # TODO: remove verbose debugging once we track down the NCCL bug
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    if with_ray and (train_context := ray.train.get_context()):
        config = config.copy()
        batch_size = config["batch_size"]
        config["batch_size"] //= train_context.get_world_size()
        text_logger.info(
            "Using per-worker and global batch sizes of "
            f"{config['batch_size']}, {batch_size}, resp."
        )

    training_module = training_module_factory(config)
    data_module = data_module_factory(config)

    strategy: str | Strategy = "auto"
    plugins: list[Any] | None = None
    logger = None
    callbacks: list[Callback] = []
    if with_ray:
        strategy = RayDDPStrategy()
        plugins = [RayLightningEnvironment()]
        callbacks.append(RayTrainReportCallback())
    else:
        # TODO: needs testing
        callbacks.append(
            LoggerCallback(text_logger, interval=config["log_every_n_steps"])
        )
        if config.get("log_to_wandb", False):
            logger = WandbLogger(project=config["project"])
            logger.watch(training_module, log="all")

    # TODO: address warning re: missing tensorboard logging directory
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        strategy=strategy,
        plugins=plugins,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=False,
        max_epochs=config["max_num_epochs"],
        log_every_n_steps=config["log_every_n_steps"],
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
    logger = get_logger(__name__)

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        enable_progress_bar=False,
        callbacks=[LoggerCallback(logger, interval=config["log_every_n_steps"])],
    )
    output = trainer.test(training_module, datamodule=data_module)
    logger.info(output)