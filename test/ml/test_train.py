import glob
import os
import tempfile

import pytest

import sitstart.ml.train
from sitstart.ml.experiments.image_multiclass_smoketest import (
    data_module_creator,
    train_config,
    training_module_creator,
)


@pytest.mark.slow
def test_train(caplog):
    with tempfile.TemporaryDirectory() as storage_path:
        config = train_config(storage_path)
        sitstart.ml.train.train(config, training_module_creator, data_module_creator)

        data_module = data_module_creator(config)
        data_module.setup(stage="fit")
        train = getattr(data_module, "train", None)
        assert train
        num_batches = (len(train) - 1) // config["batch_size"] + 1
        log_root = f"{config['storage_path']}/lightning_logs/version_0"
        logs = [log for log in caplog.records if "train_loss" in log.message]

        assert len(logs) == num_batches * config["max_num_epochs"]
        assert any(all(k in msg for k in config.keys()) for msg in caplog.messages)
        assert os.path.exists(f"{log_root}/hparams.yaml")
        assert glob.glob(f"{log_root}/events.out.tfevents.*")
        assert glob.glob(f"{log_root}/checkpoints/*.ckpt")


@pytest.mark.slow
def test_test(caplog) -> None:
    with tempfile.TemporaryDirectory() as storage_path:
        config = train_config(storage_path)
        sitstart.ml.train.test(config, training_module_creator, data_module_creator)

        log_root = f"{config['storage_path']}/lightning_logs/version_0"
        logs = [log for log in caplog.records if "test_loss" in log.message]

        assert len(logs) == 1
        assert os.path.exists(f"{log_root}/hparams.yaml")
        assert glob.glob(f"{log_root}/events.out.tfevents.*")
