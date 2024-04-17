from unittest.mock import ANY, call, patch

import pytest
import torch
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig


@patch("ktd.ml.training_module.TrainingModule.log")
def test_training_module(
    log_mock,
    smoketest_training_module_creator,
    smoketest_data_module_creator,
):
    config = {
        "optimizer": "adamw",
        "lr": 0.001,
        "weight_decay": 0.01,
        "momentum": 0.9,
        "dampening": 0,
        "max_num_epochs": 100,
        "min_lr": 0.0001,
    }

    data_module = smoketest_data_module_creator(config)
    data_module.setup()
    batch = next(iter(data_module.train_dataloader()))

    module = smoketest_training_module_creator(config)
    optimizer_config: OptimizerLRSchedulerConfig = module.configure_optimizers()

    assert isinstance(optimizer_config["optimizer"], torch.optim.AdamW)
    assert "lr_scheduler" in optimizer_config
    scheduler_config = optimizer_config["lr_scheduler"]
    assert isinstance(scheduler_config, dict)
    scheduler = scheduler_config["scheduler"]
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    loss = module.training_step(batch, 0)
    assert loss.shape == torch.Size([])

    validation_output = module.validation_step(batch, 0)
    assert "val_loss" in validation_output
    assert "val_acc" in validation_output

    module.on_validation_epoch_end()
    assert module.loss["val"] == []
    assert module.acc["val"] == []
    log_mock.assert_has_calls(
        [call("val_loss", ANY, sync_dist=True), call("val_acc", ANY, sync_dist=True)]
    )

    config["optimizer"] = "sgd"
    module = smoketest_training_module_creator(config)
    optimizer_config = module.configure_optimizers()
    assert isinstance(optimizer_config["optimizer"], torch.optim.SGD)

    config["optimizer"] = "unknown"
    module = smoketest_training_module_creator(config)
    with pytest.raises(RuntimeError):
        module.configure_optimizers()
