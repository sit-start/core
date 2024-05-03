#!/usr/bin/env python
import logging
import os
from typing import Any

import ray
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule

from sitstart.logging import get_logger
from sitstart.ml.data.cifar10 import CIFAR10
from sitstart.ml.data.smoke_test import SmokeTest
from sitstart.ml.ray import tune_with_ray
from sitstart.ml.training_module import TrainingModule

logger = get_logger(format="bare", level=logging.INFO)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, dropout_p, with_batchnorm=True, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes) if with_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes) if with_batchnorm else nn.Identity()

        self.dropout = nn.Dropout2d(p=dropout_p)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                (
                    nn.BatchNorm2d(self.expansion * planes)
                    if with_batchnorm
                    else nn.Identity()
                ),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, dropout_p, with_batchnorm=True):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.dropout_p = dropout_p
        self.with_batchnorm = with_batchnorm

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if with_batchnorm else nn.Identity()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    in_planes=self.in_planes,
                    planes=planes,
                    stride=stride,
                    dropout_p=self.dropout_p,
                    with_batchnorm=self.with_batchnorm,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes: int, dropout_p: float, with_batchnorm: bool = True):
    return ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        dropout_p=dropout_p,
        with_batchnorm=with_batchnorm,
    )


def main():
    config = {
        "debug": False,
        # specifying this shouldn't be necessary, but the wandb project
        # for resnet18_cifar10 seems to be corrupted
        "project_name": "resnet18_cifar10_1",
        "torch": {
            "backend": "nccl",
        },
        "tune": {
            "num_samples": 1,
            "long_trial_names": True,
        },
        "schedule": {
            "max_num_epochs": 75,
            "grace_period": 15,
        },
        "scale": {
            "num_workers": 1,
            "use_gpu": True,
            "resources_per_worker": {"GPU": 1},
        },
        "checkpoint": {
            "num_to_keep": 2,
            "checkpoint_score_attribute": "val_loss",
            "checkpoint_score_order": "min",
        },
        "train": {
            "seed": None,
            "augment_training_data": True,
            "weight_decay": 5e-4,
            "lr": 5e-2,
            "min_lr": 0.0,
            "momentum": 0.9,
            "dampening": 0.0,
            "optimizer": "sgd",
            "batch_size": 128,
            "dropout_p": 0.25,
            "float32_matmul_precision": "medium",
            "log_every_n_steps": 100,
            "smoke_test": False,
            "sync_batchnorm": False,
            "with_batchnorm": True,
        },
        "wandb": {
            "enabled": True,
        },
    }

    def training_module_creator(config: dict[str, Any]) -> LightningModule:
        model = ResNet18(
            num_classes=CIFAR10.NUM_CLASSES,
            dropout_p=config["dropout_p"],
            with_batchnorm=config["with_batchnorm"],
        )
        if config["sync_batchnorm"]:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        return TrainingModule(config, model=model)

    def data_module_creator(config: dict[str, Any]) -> LightningDataModule:
        if config["smoke_test"]:
            return SmokeTest(
                batch_size=config["batch_size"], num_classes=CIFAR10.NUM_CLASSES
            )

        data_dir = os.path.expanduser("~/datasets/cifar10")
        os.makedirs(data_dir, exist_ok=True)

        return CIFAR10(
            data_dir=data_dir,
            batch_size=config["batch_size"],
            augment=config["augment_training_data"],
        )

    ray_logging_level = logging.INFO
    if config["debug"]:
        os.environ["RAY_BACKEND_LOG_LEVEL"] = "debug"
        ray_logging_level = logging.DEBUG

    ray.init(logging_level=ray_logging_level)

    tune_with_ray(config, training_module_creator, data_module_creator)


if __name__ == "__main__":
    main()
