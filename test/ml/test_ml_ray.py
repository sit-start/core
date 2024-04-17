from typing import Any

import pytest
from ray.cluster_utils import Cluster

from ktd.ml.ray import train_with_ray
from ktd.ml.train import DataModuleCreator, TrainingModuleCreator


@pytest.mark.slow
def test_train_with_ray(
    ray_cluster: Cluster,
    config: dict[str, Any],
    smoketest_training_module_creator: TrainingModuleCreator,
    smoketest_data_module_creator: DataModuleCreator,
):
    train_with_ray(
        config, smoketest_training_module_creator, smoketest_data_module_creator
    )

    # TODO: add checks for checkpoints, logging, # workers used,
    # tensorboard data, repo state
