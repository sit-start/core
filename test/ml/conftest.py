import tempfile

import pytest
import torch.nn.functional as F
from ray.cluster_utils import Cluster
from torch import nn

from ktd.ml.data.smoke_test import SmokeTest
from ktd.ml.training_module import TrainingModule


@pytest.fixture(scope="module")
def ray_cluster():
    import ray

    # create a local cluster with 1 head node and 2 worker nodes
    cluster = Cluster(
        initialize_head=True,
        connect=True,
        head_node_args={"resources": {"num_cpus": 1}},
        shutdown_at_exit=True,
    )
    cluster.add_node(resources={"num_cpus": 2}, num_cpus=2)
    cluster.add_node(resources={"num_cpus": 2}, num_cpus=2)

    yield cluster

    ray.shutdown()


@pytest.fixture()
def smoketest_training_module_creator(smoketest_model):
    def factory(config):
        return TrainingModule(config, smoketest_model)

    return factory


@pytest.fixture()
def smoketest_model():
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

    return Model()


@pytest.fixture()
def smoketest_data_module_creator():
    def factory(config):
        return SmokeTest(
            batch_size=config.get("batch_size", 10),
            num_train=20,
            num_test=10,
            num_classes=4,
            img_shape=(3, 8, 8),
        )

    return factory


@pytest.fixture()
def config():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield {
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
                # 2 workers, each requesting 2 CPUs, requires we use both worker nodes
                "num_workers": 2,
                "use_gpu": False,
                "resources_per_worker": {"num_cpus": 2},
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
                "storage_path": temp_dir,
            },
            "wandb": {
                "enabled": False,
            },
        }


@pytest.fixture()
def train_config(config):
    with tempfile.TemporaryDirectory() as temp_dir:
        yield {
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
            "storage_path": temp_dir,
        }
