from typing import Any

import pytest
from omegaconf import Container, DictConfig
from ray.cluster_utils import Cluster

from sitstart.ml.experiments.util import load_experiment_config
from sitstart.ml.ray import _get_module_creators
from sitstart.ml.train import DataModuleCreator, TrainingModuleCreator
from sitstart.util.hydra import instantiate

TEST_CONFIG = "test2d"


@pytest.fixture(scope="module")
def ray_cluster():
    import ray

    # create a local cluster with 1 head node and 2 worker nodes
    cluster = Cluster(
        initialize_head=True,
        connect=True,
        head_node_args={"num_cpus": 1},
        shutdown_at_exit=True,
    )
    cluster.add_node(num_cpus=2)
    cluster.add_node(num_cpus=2)

    yield cluster

    ray.shutdown()


@pytest.fixture(scope="function")
def config() -> Container:
    return load_experiment_config(TEST_CONFIG)


@pytest.fixture(scope="function")
def train_loop_config(config: DictConfig) -> dict[str, Any]:
    config = instantiate(config, _defer_=False, _convert_="full")
    return config["param_space"]["train_loop_config"]


@pytest.fixture(scope="function")
def training_module_creator(config: DictConfig) -> TrainingModuleCreator:
    config = instantiate(config)
    return _get_module_creators(config)[0]


@pytest.fixture(scope="function")
def data_module_creator(config: DictConfig) -> DataModuleCreator:
    config = instantiate(config)
    return _get_module_creators(config)[1]
