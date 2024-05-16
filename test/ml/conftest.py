from os.path import dirname, realpath, relpath
from typing import Any

import hydra
import pytest
from omegaconf import DictConfig
from ray.cluster_utils import Cluster

from sitstart.ml.experiments import CONFIG_ROOT, HYDRA_VERSION_BASE
from sitstart.ml.ray import _get_module_creators
from sitstart.ml.train import DataModuleCreator, TrainingModuleCreator
from sitstart.util.hydra import instantiate

CONFIG_PATH = relpath(CONFIG_ROOT, realpath(dirname(__file__)))
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
def config(request: pytest.FixtureRequest) -> DictConfig:
    # replace any values that interpolate values from the hydra config
    # node, since the HydraConfig singleton isn't available
    test_name = request.node.name
    overrides = [f"name={test_name}"]

    with hydra.initialize(version_base=HYDRA_VERSION_BASE, config_path=CONFIG_PATH):
        config = hydra.compose(config_name=TEST_CONFIG, overrides=overrides)

    # sanity check the config
    instantiate(config)

    return config


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
