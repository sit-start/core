import copy
from typing import Any, Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, LRScheduler
from torcheval.metrics import Metric


class TrainingModule(pl.LightningModule):
    def __init__(
        self,
        loss_fn: nn.Module,
        lr_scheduler: LRScheduler | Callable[[Optimizer], LRScheduler] | None,
        metrics: dict[str, Metric],
        model: nn.Module,
        optimizer: Optimizer | Callable[[Any], Optimizer],
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

        self.metrics = {stage: {} for stage in ["train", "val", "test"]}
        for stage in self.metrics:
            for key, metric in metrics.items():
                self.metrics[stage][key] = copy.deepcopy(metric)
        self._update_metrics_device()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)

    def training_step(self, batch: Any, _: int) -> torch.Tensor:
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.loss_fn(outputs, targets)
        self._log_step("train", loss, outputs, targets)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._val_test_step("val", batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._val_test_step("test", batch, batch_idx)

    def on_train_epoch_start(self) -> None:
        self._log_start("train")

    def on_validation_epoch_start(self) -> None:
        self._log_start("val")

    def on_test_epoch_start(self) -> None:
        self._log_start("test")

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _log_start(self, stage: str):
        for _, metric in self.metrics[stage].items():
            metric.reset()

    def _log_step(
        self, stage: str, loss: torch.Tensor, output: torch.Tensor, target: torch.Tensor
    ) -> None:
        sync_dist = True  # TODO: eval penalty for sync_dist on each step
        self.log(
            f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=sync_dist
        )
        for key, metric in self.metrics[stage].items():
            val = metric.update(output, target).compute()
            self.log(
                f"{stage}_{key}", val, on_step=False, on_epoch=True, sync_dist=sync_dist
            )

    def _val_test_step(self, stage: str, batch: Any, _: int) -> None:
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.loss_fn(outputs, targets)

        self._log_step(stage, loss, outputs, targets)

    def _update_metrics_device(self) -> None:
        for stage in self.metrics:
            for metric in self.metrics[stage].values():
                metric.to(self.device)

    def to(self, *args, **kwargs) -> "TrainingModule":
        super().to(*args, **kwargs)
        self._update_metrics_device()
        return self
