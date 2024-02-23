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
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_fn = lambda y_hat, y: (y_hat.argmax(1) == y).mean(dtype=float)

        self.val_loss, self.val_acc = [], []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)

        return loss

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        accuracy = self.acc_fn(y_hat, y)
        self.val_loss.append(loss)
        self.val_acc.append(accuracy)
        return {"val_loss": loss, "val_acc": accuracy}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_loss).mean()
        avg_acc = torch.stack(self.val_acc).mean()
        self.log("val_loss", avg_loss, sync_dist=True)
        self.log("val_acc", avg_acc, sync_dist=True)
        self.val_loss.clear()
        self.val_acc.clear()

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

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
