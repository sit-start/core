import logging
import os

import hydra
import ray
from omegaconf import DictConfig

from sitstart.logging import get_logger
from sitstart.ml.experiments import HYDRA_VERSION_BASE
from sitstart.ml.ray import tune_with_ray

logger = get_logger(format="bare", level=logging.INFO)


@hydra.main(config_path="conf", version_base=HYDRA_VERSION_BASE)
def main(config: DictConfig) -> None:
    ray_logging_level = logging.INFO
    if config.debug:
        os.environ["RAY_BACKEND_LOG_LEVEL"] = "debug"
        ray_logging_level = logging.DEBUG

    ray.init(logging_level=ray_logging_level)

    tune_with_ray(config)


if __name__ == "__main__":
    main()
