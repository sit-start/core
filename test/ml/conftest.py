import pytest
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import Container, DictConfig
from ray.cluster_utils import Cluster

from sitstart.ml.experiments.util import load_experiment_config

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
def training_module(config: DictConfig) -> pl.LightningModule:
    trial_config = instantiate(config.trial)
    return trial_config.training_module(
        loss_fn=trial_config.loss_fn,
        lr_scheduler=trial_config.lr_scheduler,
        train_metrics=instantiate(config.eval.train.metrics),
        test_metrics=instantiate(config.eval.test.metrics),
        model=trial_config.model.module,
        optimizer=trial_config.optimizer,
    )


@pytest.fixture(scope="function")
def data_module(config: DictConfig) -> pl.LightningDataModule:
    return instantiate(config.trial.data.module)()
