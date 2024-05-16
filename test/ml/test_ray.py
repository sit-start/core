import pytest
from omegaconf import DictConfig, OmegaConf
from ray.cluster_utils import Cluster

from sitstart.ml.ray import train_with_ray, tune_with_ray


@pytest.mark.slow
def test_train_with_ray(ray_cluster: Cluster, config: DictConfig):
    train_with_ray(config)
    # TODO: add checks for checkpoints, logging, # workers used,
    # tensorboard data, repo state


@pytest.mark.slow
def test_tune_with_ray(ray_cluster: Cluster, config: DictConfig):
    config.param_space.scaling_config.num_workers = OmegaConf.create(
        {"_target_": "ray.tune.grid_search", "values": [1, 2]}
    )
    config.param_space.train_loop_config.lr = OmegaConf.create(
        {"_target_": "ray.tune.grid_search", "values": [1e-3, 5e-2]}
    )
    config.param_space.train_loop_config.batch_size = OmegaConf.create(
        {"_target_": "ray.tune.grid_search", "values": [5, 10]}
    )
    tune_with_ray(config)
    # TODO: add checks for checkpoints, logging, # workers used,
    # tensorboard data, repo state, # trials
