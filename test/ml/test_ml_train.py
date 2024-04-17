import os
import glob
from typing import Any

import pytest

import ktd.ml.train
from ktd.ml.train import DataModuleCreator, TrainingModuleCreator


@pytest.mark.slow
def test_train(
    train_config: dict[str, Any],
    smoketest_training_module_creator: TrainingModuleCreator,
    smoketest_data_module_creator: DataModuleCreator,
    caplog,
):
    ktd.ml.train.train(
        train_config, smoketest_training_module_creator, smoketest_data_module_creator
    )

    data_module = smoketest_data_module_creator(train_config)
    data_module.setup(stage="fit")
    train = getattr(data_module, "train", None)
    assert train
    num_batches = (len(train) - 1) // train_config["batch_size"] + 1
    log_root = f"{train_config['storage_path']}/lightning_logs/version_0"
    logs = [log for log in caplog.records if "train_loss" in log.message]

    assert len(logs) == num_batches * train_config["max_num_epochs"]
    assert any(all(k in msg for k in train_config.keys()) for msg in caplog.messages)
    assert os.path.exists(f"{log_root}/hparams.yaml")
    assert glob.glob(f"{log_root}/events.out.tfevents.*")
    assert glob.glob(f"{log_root}/checkpoints/*.ckpt")


@pytest.mark.slow
def test_test(
    train_config: dict[str, Any],
    smoketest_training_module_creator: TrainingModuleCreator,
    smoketest_data_module_creator: DataModuleCreator,
    caplog,
) -> None:
    ktd.ml.train.test(
        train_config, smoketest_training_module_creator, smoketest_data_module_creator
    )

    log_root = f"{train_config['storage_path']}/lightning_logs/version_0"
    logs = [log for log in caplog.records if "test_loss" in log.message]

    assert len(logs) == 1
    assert os.path.exists(f"{log_root}/hparams.yaml")
    assert glob.glob(f"{log_root}/events.out.tfevents.*")
