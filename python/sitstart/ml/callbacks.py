import logging

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback


class LoggerCallback(Callback):
    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        metrics = trainer.callback_metrics
        self._logger.info(
            f"train_loss: {metrics['train_loss']:.3f} [{trainer.current_epoch:>3d}]"
        )
