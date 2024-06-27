import glob
from pathlib import Path

from sitstart.ml.experiments import CONFIG_ROOT
from sitstart.ml.experiments.util import (
    load_experiment_config,
    register_omegaconf_resolvers,
    resolve,
)

register_omegaconf_resolvers()


def test_experiment_configs() -> None:
    for config_path in glob.glob(f"{CONFIG_ROOT}/*.yaml"):
        config_name = Path(config_path).stem
        if config_name.startswith("_") and config_name.endswith("_"):
            continue
        if not Path(config_path).read_text():
            continue

        print(config_name)
        config = load_experiment_config(Path(config_path).stem)
        resolve(config, sample_params=True)
