from unittest.mock import ANY, call, patch

import pytorch_lightning as pl
import torch


@patch("torcheval.metrics.metric.Metric.reset")
@patch("sitstart.ml.training_module.TrainingModule.log")
def test_training_module(
    log_mock,
    metric_reset_mock,
    training_module: pl.LightningModule,
    data_module: pl.LightningDataModule,
) -> None:
    data_module.setup(stage="fit")
    batch = next(iter(data_module.train_dataloader()))

    optimizer_config = training_module.configure_optimizers()
    assert isinstance(optimizer_config, dict)
    assert isinstance(optimizer_config["optimizer"], torch.optim.SGD)
    assert "lr_scheduler" in optimizer_config

    scheduler_config = optimizer_config["lr_scheduler"]
    assert isinstance(scheduler_config, dict)

    scheduler = scheduler_config["scheduler"]
    assert isinstance(scheduler, torch.optim.lr_scheduler.ConstantLR)

    log_mock.reset_mock()
    loss = training_module.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])
    log_mock.assert_has_calls(
        [call("train_loss", ANY, on_step=False, on_epoch=True, sync_dist=True)]
    )

    training_module.on_train_epoch_end()
    log_mock.assert_has_calls([call("train_acc", ANY, sync_dist=True)])

    training_module.on_validation_epoch_start()
    metric_reset_mock.assert_has_calls([call()])

    training_module.validation_step(batch, 0)
    log_mock.assert_has_calls(
        [call("val_loss", ANY, on_step=False, on_epoch=True, sync_dist=True)]
    )

    training_module.on_validation_epoch_end()
    log_mock.assert_has_calls([call("val_acc", ANY, sync_dist=True)])
