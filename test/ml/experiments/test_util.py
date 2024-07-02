from omegaconf import DictConfig
import pytest

from sitstart.ml.experiments.util import validate_experiment_config


def test_validate_experiment_config():
    grid_search = {"_target_": "ray.tune.grid_search", "values": [1, 2]}

    # valid
    cfg = {
        # grid search in pspace
        "param_space": {"a": grid_search},
        # pspace interp in trial
        "trial": {"b": r"${param_space.a}", "c": r"${trial.b}", "d": 22},
        # trial interp outside trial that doesn't depend on pspace
        "e": r"${trial.d}",
    }
    validate_experiment_config(DictConfig(cfg))

    # invalid, grid search in root node
    cfg = {"a": grid_search}
    with pytest.raises(ValueError):
        validate_experiment_config(DictConfig(cfg))

    # invalid, pspace interp in non-trial node
    cfg = {"param_space": {"x": grid_search}, "y": r"${param_space.x}"}
    with pytest.raises(ValueError):
        validate_experiment_config(DictConfig(cfg))

    # invalid, indirect interp of pspace node in non-trial node
    cfg = {"y": r"${trial.y}"}
    with pytest.raises(ValueError):
        validate_experiment_config(DictConfig(cfg))
