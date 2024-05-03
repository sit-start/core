import tempfile

import pytest
from ray.cluster_utils import Cluster

from ktd.ml.experiments.image_multiclass_smoketest import (
    data_module_creator,
    train_with_ray_config,
    tune_with_ray_config,
    training_module_creator,
)
from ktd.ml.ray import train_with_ray, tune_with_ray


@pytest.mark.slow
def test_train_with_ray(ray_cluster: Cluster):
    with tempfile.TemporaryDirectory() as storage_path:
        config = train_with_ray_config(storage_path)
        train_with_ray(config, training_module_creator, data_module_creator)
    # TODO: add checks for checkpoints, logging, # workers used,
    # tensorboard data, repo state


@pytest.mark.slow
def test_tune_with_ray(ray_cluster: Cluster):
    with tempfile.TemporaryDirectory() as storage_path:
        config = tune_with_ray_config(storage_path)
        tune_with_ray(config, training_module_creator, data_module_creator)
    # TODO: add checks for checkpoints, logging, # workers used,
    # tensorboard data, repo state
