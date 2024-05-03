import logging
from typing import Any, Mapping

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from torch import Tensor


class LoggerCallback(Callback):
    def __init__(self, logger: logging.Logger, interval: int = 100):
        self._logger = logger
        self._interval = interval

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        metrics = trainer.callback_metrics
        if batch_idx % self._interval == 0:
            self._logger.info(
                f"train_loss: {metrics['train_loss']:.3f} "
                f"[{trainer.current_epoch:>3d} | "
                f"{batch_idx:>4d} / {trainer.num_training_batches:<4d}] "
            )
