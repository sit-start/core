import pytest
from omegaconf import DictConfig, OmegaConf
from ray.cluster_utils import Cluster

from sitstart.ml.ray import train_with_ray, tune_with_ray


@pytest.mark.slow
def test_train_with_ray(config: DictConfig, ray_cluster: Cluster):
    train_with_ray(config)


@pytest.mark.slow
def test_tune_with_ray(config: DictConfig, ray_cluster: Cluster):
    num_workers = {"_target_": "ray.tune.grid_search", "values": [1, 2]}
    config.param_space.scaling_config.num_workers = num_workers

    OmegaConf.update(config.param_space.train_loop_config, "lr", force_add=True)
    lr = {"_target_": "ray.tune.grid_search", "values": [1e-3, 5e-2]}
    config.param_space.train_loop_config.lr = lr
    config.trial.optimizer.lr = r"${param_space.train_loop_config.lr}"

    OmegaConf.update(config.param_space.train_loop_config, "batch_size", force_add=True)
    batch_size = {"_target_": "ray.tune.grid_search", "values": [5, 10]}
    config.param_space.train_loop_config.batch_size = batch_size
    config.trial.data.batch_size = r"${param_space.train_loop_config.batch_size}"

    tune_with_ray(config)
