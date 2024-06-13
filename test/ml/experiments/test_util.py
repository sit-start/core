from omegaconf import DictConfig
import pytest

from sitstart.ml.experiments.util import validate_experiment_config


def test_validate_experiment_config():
    grid_search = {"_target_": "ray.tune.grid_search", "values": [1, 2]}

    # valid, with all combos
    cfg = {
        "param_space": {"x": grid_search},
        "trial": {"y": r"${param_space.x}", "z": r"${trial.y}"},
    }
    validate_experiment_config(DictConfig(cfg))

    # invalid, grid search in root node
    cfg = {"x": grid_search}
    with pytest.raises(ValueError):
        validate_experiment_config(DictConfig(cfg))

    # invalid, pspace interp in non-trial node
    cfg = {"param_space": {"x": grid_search}, "y": r"${param_space.x}"}
    with pytest.raises(ValueError):
        validate_experiment_config(DictConfig(cfg))

    # invalid, trial interp in non-trial node
    cfg = {"trial": {"x": 42}, "y": r"${trial.x}"}
    with pytest.raises(ValueError):
        validate_experiment_config(DictConfig(cfg))
