import glob
import os
from typing import Any

import pytest

from sitstart.ml import train
from sitstart.ml.train import DataModuleCreator, TrainingModuleCreator


@pytest.mark.slow
def test_train(
    caplog,
    train_loop_config: dict[str, Any],
    training_module_creator: TrainingModuleCreator,
    data_module_creator: DataModuleCreator,
) -> None:
    config = train_loop_config
    train.train(config, training_module_creator, data_module_creator)
    data_module = data_module_creator(config)
    data_module.setup(stage="fit")
    train_data = getattr(data_module, "train", None)

    assert train_data

    num_batches = (len(train_data) - 1) // config["batch_size"] + 1
    log_root = f"{config['storage_path']}/lightning_logs/version_0"
    logs = [log for log in caplog.records if "train_loss" in log.message]

    assert len(logs) == num_batches * config["max_num_epochs"]
    assert any(all(k in msg for k in config.keys()) for msg in caplog.messages)
    assert os.path.exists(f"{log_root}/hparams.yaml")
    assert glob.glob(f"{log_root}/events.out.tfevents.*")
    assert glob.glob(f"{log_root}/checkpoints/*.ckpt")


@pytest.mark.slow
def test_test(
    caplog,
    train_loop_config: dict[str, Any],
    training_module_creator: TrainingModuleCreator,
    data_module_creator: DataModuleCreator,
) -> None:
    config = train_loop_config
    train.test(config, training_module_creator, data_module_creator)

    log_root = f"{config['storage_path']}/lightning_logs/version_0"
    logs = [log for log in caplog.records if "test_loss" in log.message]

    assert len(logs) == 1
    assert os.path.exists(f"{log_root}/hparams.yaml")
    assert glob.glob(f"{log_root}/events.out.tfevents.*")
