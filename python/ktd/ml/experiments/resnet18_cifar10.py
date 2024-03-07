#!/usr/bin/env python
import logging
import os
import os.path as osp
from typing import Any

import ray
import torch.nn as nn
import torch.nn.functional as F
from ktd.logging import get_logger
from ktd.ml.data.cifar10 import CIFAR10, SmokeTestCIFAR10
from ktd.ml.ray import tune_with_ray
from ktd.ml.training_module import TrainingModule
from pytorch_lightning import LightningDataModule, LightningModule

logger = get_logger(format="bare", level=logging.INFO)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, dropout_p, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

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
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, dropout_p):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.dropout_p = dropout_p

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
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


def ResNet18(num_classes: int, dropout_p: float):
    return ResNet(
        BasicBlock, [2, 2, 2, 2], num_classes=num_classes, dropout_p=dropout_p
    )


def main():
    config = {
        "debug": False,
        "driver_name": osp.basename(__file__).split(".")[0],
        "strict_provenance": False,
        "tune": {"num_samples": 1, "long_trial_names": True},
        "schedule": {
            "max_num_epochs": 15,
            "grace_period": 15,
        },
        "scale": {
            "num_workers": 1,
            "use_gpu": True,
        },
        "train_loop_config": {
            "augment_training_data": False,
            "weight_decay": 5e-4,
            "lr": 1e-2,
            "min_lr": 0.0,
            "momentum": 0.9,
            "dampening": 0.0,
            "optimizer": "sgd",
            "batch_size": 256,
            "dropout_p": 0.0,
            "float32_matmul_precision": "medium",
            "log_every_n_steps": 100,
            "smoke_test": False,
            "sync_batchnorm": False,
        },
        "wandb": {
            "enabled": True,
        },
    }

    def training_module_factory(config: dict[str, Any]) -> LightningModule:
        model = ResNet18(num_classes=CIFAR10.NUM_CLASSES, dropout_p=config["dropout_p"])
        if config["sync_batchnorm"]:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        return TrainingModule(config, model=model)

    def data_module_factory(config: dict[str, Any]) -> LightningDataModule:
        if config["smoke_test"]:
            return SmokeTestCIFAR10(batch_size=config["batch_size"])
        return CIFAR10(
            batch_size=config["batch_size"],
            data_dir="data",
            augment=config["augment_training_data"],
        )

    ray_logging_level = logging.INFO
    if config["debug"]:
        os.environ["RAY_BACKEND_LOG_LEVEL"] = "debug"
        ray_logging_level = logging.DEBUG

    ray.init(logging_level=ray_logging_level)

    tune_with_ray(config, training_module_factory, data_module_factory)


if __name__ == "__main__":
    main()
