from typing import Any
from unittest.mock import ANY, call, patch

import pytest
import torch

from sitstart.ml.train import DataModuleCreator, TrainingModuleCreator


@patch("sitstart.ml.training_module.TrainingModule.log")
def test_training_module(
    log_mock,
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
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    loss = module.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])

    validation_output = module.validation_step(batch, 0)
    assert isinstance(validation_output, dict)
    assert "val_loss" in validation_output
    assert "val_acc" in validation_output

    module.on_validation_epoch_end()
    assert module.loss["val"] == []
    assert module.acc["val"] == []
    log_mock.assert_has_calls(
        [call("val_loss", ANY, sync_dist=True), call("val_acc", ANY, sync_dist=True)]
    )

    train_loop_config["optimizer"] = "adamw"
    module = training_module_creator(train_loop_config)
    optimizer_config = module.configure_optimizers()
    assert isinstance(optimizer_config, dict)
    assert isinstance(optimizer_config["optimizer"], torch.optim.AdamW)

    train_loop_config["optimizer"] = "unknown"
    module = training_module_creator(train_loop_config)
    with pytest.raises(RuntimeError):
        module.configure_optimizers()
