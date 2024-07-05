import copy
from typing import Any, Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, LRScheduler
from torcheval.metrics import Metric
from torchmetrics import Metric as TorchMetric
from sitstart.logging import get_logger
from sitstart.ml.logging import is_multidim_metric, log_multidim_metric
from sitstart.ml.transforms import BatchTransform, IdentityBatchTransform

logger = get_logger(__name__)


class TrainingModule(pl.LightningModule):
    def __init__(
        self,
        loss_fn: nn.Module,
        lr_scheduler: LRScheduler | Callable[[Optimizer], LRScheduler] | None,
        train_metrics: dict[str, Metric | TorchMetric] | None,
        test_metrics: dict[str, Metric | TorchMetric] | None,
        model: nn.Module,
        optimizer: Optimizer | Callable[[Any], Optimizer],
        train_batch_transform: BatchTransform | None = None,
    ) -> None:
        super().__init__()

        if isinstance(optimizer, Callable):
            optimizer = optimizer(model.parameters())
        if isinstance(lr_scheduler, Callable):
            lr_scheduler = lr_scheduler(optimizer)
        if lr_scheduler is None:
            lr_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=0)

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_batch_transform = train_batch_transform or IdentityBatchTransform()

        self.metrics = {stage: {} for stage in ["train", "val", "test"]}
        for stage in self.metrics:
            stage_metrics = train_metrics if stage == "train" else test_metrics
            for key, metric in (stage_metrics or {}).items():
                self.metrics[stage][key] = copy.deepcopy(metric)
                self.metrics[stage][key].reset()
        self._update_metrics_device()

        if self.metrics["train"] and self.train_batch_transform.train_only:
            logger.warning(
                "Batch transform requires recomputing outputs for training metrics. "
                "This may slow down training. Consider disabling training metrics."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)

    def training_step(self, batch: Any, _: int) -> torch.Tensor:
        inputs, targets = self.train_batch_transform(batch)
        outputs = self.forward(inputs)
        loss = self.loss_fn(outputs, targets)

        if self.metrics["train"] and self.train_batch_transform.train_only:
            inputs, targets = batch
            self.train(False)
            with torch.no_grad():
                outputs = self.forward(inputs)
            self.train(True)

        self._log_step("train", loss, outputs, targets)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._val_test_step("val", batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._val_test_step("test", batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        self._log_end("val")

    def on_train_epoch_end(self) -> None:
        self._log_end("train")

    def on_test_epoch_end(self) -> None:
        self._log_end("test")

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def to(self, *args, **kwargs) -> "TrainingModule":
        super().to(*args, **kwargs)
        self._update_metrics_device()
        return self

    def _log_step(
        self, stage: str, loss: torch.Tensor, output: torch.Tensor, target: torch.Tensor
    ) -> None:
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        for _, metric in self.metrics[stage].items():
            metric.update(output, target)

    def _log_end(self, stage: str) -> None:
        if stage == "train":
            last_lr = self.lr_scheduler.get_last_lr()
            suffix = "" if len(last_lr) == 1 else "_{i}"
            for i, lr in enumerate(last_lr):
                self.log(f"lr{suffix.format(i)}", lr, sync_dist=True)

        for key, metric in self.metrics[stage].items():
            metric_name = f"{stage}_{key}"
            if is_multidim_metric(metric):
                for pl_logger in self.trainer.loggers:
                    log_multidim_metric(pl_logger, metric_name, metric)
            else:
                self.log(metric_name, metric.compute(), sync_dist=True)

            metric.reset()

    def _val_test_step(self, stage: str, batch: Any, _: int) -> None:
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.loss_fn(outputs, targets)

        self._log_step(stage, loss, outputs, targets)

    def _update_metrics_device(self) -> None:
        for stage in self.metrics:
            for metric in self.metrics[stage].values():
                metric.to(self.device)
