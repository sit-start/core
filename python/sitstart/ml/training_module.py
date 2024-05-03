from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig


class TrainingModule(pl.LightningModule):
    def __init__(
        self,
        config: dict,
        model: nn.Module,
    ) -> None:
        super().__init__()

        self.config = config.copy()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()  # TODO: hardcoded
        self.acc_fn = lambda y_hat, y: (y_hat.argmax(1) == y).mean(dtype=float)

        self.loss = {"val": [], "test": []}
        self.acc = {"val": [], "test": []}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)

    def training_step(self, batch: Any, _: int) -> torch.Tensor:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)

        return loss

    def _val_test_step(
        self, target: str, batch: Any, _: int
    ) -> dict[str, torch.Tensor]:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        accuracy = self.acc_fn(y_hat, y)
        self.loss[target].append(loss)
        self.acc[target].append(accuracy)
        return {f"{target}_loss": loss, f"{target}_acc": accuracy}

    def _on_val_test_epoch_end(self, target: str):
        avg_loss = torch.stack(self.loss[target]).mean()
        avg_acc = torch.stack(self.acc[target]).mean()
        self.log(f"{target}_loss", avg_loss, sync_dist=True)
        self.log(f"{target}_acc", avg_acc, sync_dist=True)
        self.loss[target].clear()
        self.acc[target].clear()

    def validation_step(self, batch: Any, batch_idx: int) -> dict[str, torch.Tensor]:
        return self._val_test_step("val", batch, batch_idx)

    def on_validation_epoch_end(self):
        self._on_val_test_epoch_end("val")

    def test_step(self, batch: Any, batch_idx: int) -> dict[str, torch.Tensor]:
        return self._val_test_step("test", batch, batch_idx)

    def on_test_epoch_end(self):
        self._on_val_test_epoch_end("test")

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        if self.config["optimizer"] == "sgd":
            optimizer = optim.SGD(
                params=self.model.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
                momentum=self.config["momentum"],
                dampening=self.config["dampening"],
            )
        elif self.config["optimizer"] == "adamw":
            optimizer = optim.AdamW(
                params=self.model.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
            )
        else:
            raise RuntimeError(f"Optimizer f{self.config['optimizer']} not recognized")

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config["max_num_epochs"],
            eta_min=self.config["min_lr"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
