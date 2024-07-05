import os
from pathlib import Path
from typing import Any, Mapping

import pytorch_lightning as pl
import torch
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import Strategy
from pytorch_lightning.utilities.compile import from_compiled
from torch._dynamo import OptimizedModule

from sitstart.logging import get_logger
from sitstart.ml import DEFAULT_CHECKPOINT_ROOT
from sitstart.ml.callbacks import LoggerCallback
from sitstart.ml.data.modules.vision_data_module import VisionDataModule
from sitstart.ml.training_module import TrainingModule

logger = get_logger(__name__)


class Trainer(pl.Trainer):
    def fit(
        self,
        model: pl.LightningModule,
        train_dataloaders: Any = None,
        val_dataloaders: Any = None,
        datamodule: pl.LightningDataModule | None = None,
        ckpt_path: Path | str | None = None,
    ) -> None:
        if isinstance(model, OptimizedModule):
            model = from_compiled(model)
        self._setup_loss_fn(model, datamodule)
        super().fit(
            model=model,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )

    @staticmethod
    def _setup_loss_fn(
        model: pl.LightningModule, data_module: pl.LightningDataModule | None
    ) -> None:
        # update loss function weights if applicable
        is_vision_data_module = isinstance(data_module, VisionDataModule)
        is_training_module = isinstance(model, TrainingModule)
        if not (is_vision_data_module and is_training_module):
            return
        if not hasattr(model.loss_fn, "weight"):
            return

        weight = data_module.criteria_weight
        if weight is None:
            return

        logger.info(f"Updating loss function with weight = {weight}")
        model.loss_fn.weight = weight


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
    num_sanity_val_steps: int | None = None,
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
        ckpt_path: Path to a local checkpoint from which to resume training.
        float32_matmul_precision: Precision for matrix multiplication.
        logging_interval: Logging interval in batches.
        max_num_epochs: Maximum number of epochs.
        num_sanity_val_steps: Number of sanity validation steps. See pl.Trainer.
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

    if use_gpu:
        accelerator = "gpu"
    elif torch.backends.mps.is_available():
        accelerator = "mps"
    else:
        accelerator = "cpu"

    # TODO: address warning re: missing tensorboard logging directory
    root_dir = str(storage_path) if storage_path else None
    trainer = Trainer(
        accelerator=accelerator,
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
        num_sanity_val_steps=num_sanity_val_steps,
        plugins=plugins,
        strategy=strategy,
    )

    if with_ray:
        trainer = prepare_trainer(trainer)
        if ckpt := ray.train.get_checkpoint():
            assert (
                ckpt_path is None
            ), "Cannot load both trial- and user-specified checkpoints."
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
    checkpoint_path: str | os.PathLike[str] | None = None,
    storage_path: str | os.PathLike[str] | None = None,
    accelerator: str | Accelerator = "auto",
) -> list[Mapping[str, float]]:
    """Test a model.

    Args:
        data_module: PyTorch Lightning data module, whose
            `test_dataloader` will be used.
        training_module: PyTorch Lightning training module.
        checkpoint_path: Path to a checkpoint from which model weights
            are loaded.
        storage_path: Path to save results.
        accelerator: Accelerator to use.
    """
    trainer = pl.Trainer(
        devices="auto",
        accelerator=accelerator,
        enable_progress_bar=False,
        default_root_dir=str(storage_path) if storage_path else None,
    )

    if not training_module and not checkpoint_path:
        raise ValueError("Either training_module or checkpoint_path must be provided.")

    logger.info("Testing model" + " with checkpoint." if checkpoint_path else ".")
    return trainer.test(
        training_module,
        datamodule=data_module,
        ckpt_path=str(checkpoint_path) if checkpoint_path else None,
    )
