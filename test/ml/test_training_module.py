from typing import Any
from unittest.mock import ANY, call, patch

import torch

from sitstart.ml.train import DataModuleCreator, TrainingModuleCreator


@patch("torcheval.metrics.metric.Metric.reset")
@patch("sitstart.ml.training_module.TrainingModule.log")
def test_training_module(
    log_mock,
    metric_reset_mock,
    train_loop_config: dict[str, Any],
    training_module_creator: TrainingModuleCreator,
    data_module_creator: DataModuleCreator,
) -> None:
    data_module = data_module_creator(train_loop_config)
    data_module.setup(stage="fit")
    batch = next(iter(data_module.train_dataloader()))

    module = training_module_creator(train_loop_config)
    optimizer_config = module.configure_optimizers()
    assert isinstance(optimizer_config, dict)
    assert isinstance(optimizer_config["optimizer"], torch.optim.SGD)
    assert "lr_scheduler" in optimizer_config

    scheduler_config = optimizer_config["lr_scheduler"]
    assert isinstance(scheduler_config, dict)

    scheduler = scheduler_config["scheduler"]
    assert isinstance(scheduler, torch.optim.lr_scheduler.ConstantLR)

    log_mock.reset_mock()
    loss = module.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])
    log_mock.assert_has_calls(
        [call("train_loss", ANY, on_step=False, on_epoch=True, sync_dist=True)]
    )
    log_mock.assert_has_calls(
        [call("train_acc", ANY, on_step=False, on_epoch=True, sync_dist=True)]
    )

    module.on_validation_epoch_start()
    metric_reset_mock.assert_has_calls([call()])

    log_mock.reset_mock()
    module.validation_step(batch, 0)
    log_mock.assert_has_calls(
        [call("val_loss", ANY, on_step=False, on_epoch=True, sync_dist=True)]
    )
    log_mock.assert_has_calls(
        [call("val_acc", ANY, on_step=False, on_epoch=True, sync_dist=True)]
    )
