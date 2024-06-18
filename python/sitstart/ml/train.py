import os
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import Strategy

from sitstart.ml import DEFAULT_CHECKPOINT_ROOT
from sitstart.logging import get_logger
from sitstart.ml.callbacks import LoggerCallback

logger = get_logger(__name__)


def train(
    data_module: pl.LightningDataModule,
    training_module: pl.LightningModule,
    *,
    ckpt_path: str | os.PathLike[str] | None = None,
    float32_matmul_precision: str = "default",
    gradient_clip_val: float | None = None,
    gradient_clip_algorithm: str | None = None,
    logging_interval: int = 100,
    max_num_epochs: int = 100,
    project_name: str | None = None,
    seed: int | None = None,
    storage_path: os.PathLike[str] | None = None,
    use_gpu: bool = False,
    wandb_enabled: bool = False,
    with_ray: bool = False,
) -> None:
    """Train a PyTorch Lightning model.

    Args:
        data_module: PyTorch Lightning data module.
        training_module: PyTorch Lightning training module.
        ckpt_path: Path to a checkpoint from which to resume training.
            Must be a local path if _with_ray=False.
        float32_matmul_precision: Precision for matrix multiplication.
        logging_interval: Logging interval in batches.
        max_num_epochs: Maximum number of epochs.
        project_name: Name of the project.
        seed: Random seed.
        storage_path: Path to save results. Must be a local path if
            _with_ray=False.
        use_gpu: Whether to use the GPU.
        wandb_enabled: Whether to enable Weights & Biases logging.
        with_ray: Whether train() is invoked from a Ray training or
            tuning run.
    """
    if seed is not None:
        pl.seed_everything(seed)

    if with_ray:
        import ray
        import ray.train
        from ray.train.lightning import (
            RayDDPStrategy,
            RayLightningEnvironment,
            RayTrainReportCallback,
            prepare_trainer,
        )

    torch.set_float32_matmul_precision(float32_matmul_precision)

    strategy: str | Strategy = "auto"
    plugins: list[Any] | None = None
    pl_logger = None
    callbacks: list[Callback] = []
    if with_ray:
        strategy = RayDDPStrategy()
        plugins = [RayLightningEnvironment()]
        callbacks.append(RayTrainReportCallback())
    else:
        callbacks.append(LoggerCallback(logger))
        if wandb_enabled:
            pl_logger = WandbLogger(
                project=project_name,
                save_dir=str(storage_path) if storage_path else ".",
            )

    # TODO: address warning re: missing tensorboard logging directory
    root_dir = str(storage_path) if storage_path else None
    trainer = pl.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        callbacks=callbacks,
        default_root_dir=root_dir if not with_ray else None,
        devices="auto",
        enable_checkpointing=not with_ray,
        enable_progress_bar=False,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
        logger=pl_logger,
        log_every_n_steps=logging_interval,
        max_epochs=max_num_epochs,
        plugins=plugins,
        strategy=strategy,
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
    data_module: pl.LightningDataModule,
    training_module: pl.LightningModule,
    storage_path: os.PathLike[str] | None = None,
    use_gpu: bool = False,
    with_ray: bool = False,
) -> None:
    root_dir = str(storage_path) if storage_path else None
    trainer = pl.Trainer(
        devices="auto",
        accelerator="gpu" if use_gpu else "cpu",
        enable_progress_bar=False,
        default_root_dir=root_dir if not with_ray else None,
    )
    output = trainer.test(training_module, datamodule=data_module)
    logger.info(output)
