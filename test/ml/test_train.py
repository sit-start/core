import glob
import os

import pytest
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from sitstart.ml import train
from sitstart.ml.experiments.util import register_omegaconf_resolvers

register_omegaconf_resolvers()


@pytest.mark.slow
def test_train(
    caplog,
    config: DictConfig,
    data_module: pl.LightningDataModule,
    training_module: pl.LightningModule,
) -> None:
    OmegaConf.resolve(config)
    train.train(
        data_module,
        training_module,
        ckpt_path=config.restore.checkpoint_path,
        float32_matmul_precision=config.float32_matmul_precision,
        gradient_clip_val=config.gradient_clip.value,
        gradient_clip_algorithm=config.gradient_clip.algorithm,
        logging_interval=config.logging_interval,
        max_num_epochs=config.max_num_epochs,
        project_name=config.name,
        seed=config.seed,
        storage_path=config.storage_path,
        use_gpu=config.param_space.scaling_config.use_gpu,
        wandb_enabled=config.wandb.enabled,
        with_ray=False,
    )

    log_root = f"{config.storage_path}/lightning_logs/version_0"
    logs = [log for log in caplog.records if "train_loss" in log.message]

    assert len(logs) == config.max_num_epochs
    assert os.path.exists(f"{log_root}/hparams.yaml")
    assert glob.glob(f"{log_root}/events.out.tfevents.*")
    assert glob.glob(f"{log_root}/checkpoints/*.ckpt")


@pytest.mark.slow
def test_test(
    caplog,
    config: DictConfig,
    data_module: pl.LightningDataModule,
    training_module: pl.LightningModule,
) -> None:
    OmegaConf.resolve(config)
    train.test(
        data_module,
        training_module,
        storage_path=config.storage_path,
        use_gpu=config.param_space.scaling_config.use_gpu,
    )

    log_root = f"{config.storage_path}/lightning_logs/version_0"
    logs = [log for log in caplog.records if "test_loss" in log.message]

    assert len(logs) == 1
    assert os.path.exists(f"{log_root}/hparams.yaml")
    assert glob.glob(f"{log_root}/events.out.tfevents.*")
