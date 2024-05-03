import logging
from typing import Any

import pytorch_lightning as pl
import ray
import ray.tune
import torch.nn.functional as F
from torch import nn

from sitstart.ml.data.smoke_test import SmokeTest
from sitstart.ml.ray import tune_with_ray
from sitstart.ml.training_module import TrainingModule


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 18, 3)
        self.fc = nn.Linear(18, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        x = self.fc(x)

        return x


def training_module_creator(config: dict[str, Any]) -> pl.LightningModule:
    return TrainingModule(config, Model())


def data_module_creator(config: dict[str, Any]) -> pl.LightningDataModule:
    return SmokeTest(
        batch_size=config.get("batch_size", 10),
        num_train=20,
        num_test=10,
        num_classes=4,
        img_shape=(3, 8, 8),
    )


def train_with_ray_config(storage_path: str | None = None) -> dict[str, Any]:
    return {
        "debug": False,
        "save_repo_state": False,
        "torch": {
            "backend": "gloo",
        },
        "tune": {
            "num_samples": 1,
            "long_trial_names": True,
        },
        "schedule": {
            "max_num_epochs": 2,
            "grace_period": 1,
        },
        "scale": {
            "num_workers": 2,
            "use_gpu": False,
            "trainer_resources": {"CPU": 0},
            "resources_per_worker": {"CPU": 2},
        },
        "checkpoint": {
            "num_to_keep": 2,
            "checkpoint_score_attribute": "val_loss",
            "checkpoint_score_order": "min",
        },
        "train": {
            "seed": 42,
            "weight_decay": 5e-4,
            "lr": 5e-2,
            "min_lr": 0.0,
            "momentum": 0.9,
            "dampening": 0.0,
            "optimizer": "sgd",
            "batch_size": 10,
            "dropout_p": 0.25,
            "float32_matmul_precision": "medium",
            "log_every_n_steps": 1,
            "storage_path": storage_path,
        },
        "wandb": {
            "enabled": False,
        },
    }


def tune_with_ray_config(storage_path: str | None = None) -> dict[str, Any]:
    config = train_with_ray_config(storage_path)
    config["train"]["lr"] = ray.tune.grid_search([1e-3, 1e-2])

    return config


def train_config(local_storage_path: str) -> dict[str, Any]:
    return {
        "seed": 42,
        "weight_decay": 5e-4,
        "lr": 5e-2,
        "min_lr": 0.0,
        "momentum": 0.9,
        "dampening": 0.0,
        "optimizer": "sgd",
        "batch_size": 8,
        "dropout_p": 0.25,
        "float32_matmul_precision": "medium",
        "log_every_n_steps": 1,
        "max_num_epochs": 2,
        "use_gpu": False,
        "storage_path": local_storage_path,
    }


def main():
    config = tune_with_ray_config()
    ray.init(logging_level=logging.INFO)
    tune_with_ray(config, training_module_creator, data_module_creator)


if __name__ == "__main__":
    main()
