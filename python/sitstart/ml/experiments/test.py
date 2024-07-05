import os
from typing import Any, Literal

import ray.train
from omegaconf import DictConfig

from sitstart.logging import get_logger
from sitstart.ml.experiments import TEST_ROOT
from sitstart.ml.experiments.restore import (
    get_checkpoint,
    get_checkpoint_file_path,
    get_experiment_state,
    to_local_checkpoint,
)
from sitstart.ml.experiments.util import (
    apply_experiment_config_overrides,
    get_lightning_modules_from_config,
)
from sitstart.ml.train import test

logger = get_logger(__name__)


def test_trial(
    project_name: str,
    trial_id: str,
    config_overrides: list[str] | None = None,
    storage_path: str | os.PathLike[str] = TEST_ROOT,
    select: Literal["best", "last"] = "last",
    select_metric: str | None = None,
    select_mode: str | None = None,
) -> Any:
    """Test the trial with the given project name and trial ID.

    Loads the checkpoint for the project + trial and tests with
    `test_checkpoint`. `select_*` arguments are supplied to `get_checkpoint`.

    Defaults to testing the last checkpoint.
    """
    logger.info(f"Testing trial {trial_id!r} in project {project_name!r}.")
    ckpt = get_checkpoint(
        project_name=project_name,
        trial_id=trial_id,
        select=select,
        select_metric=select_metric,
        select_mode=select_mode,
    )
    if not ckpt:
        raise ValueError(
            f"Failed to get checkpoint for trial {trial_id!r} in project {project_name!r}."
        )

    results = test_checkpoint(
        ckpt,
        config_overrides=config_overrides,
        test_storage_path=storage_path,
        trial_id=trial_id,
    )

    return results


def test_checkpoint(
    checkpoint: ray.train.Checkpoint,
    config_overrides: list[str] | None = None,
    test_storage_path: str | os.PathLike[str] = TEST_ROOT,
    trial_id: str = "unspecified",
) -> Any:
    """Test the given checkpoint.

    Args:
        checkpoint: The checkpoint to test. If the checkpoint was
            created at a remote storage path, the remote `Checkpoint` should be
            provided here.
        config_overrides: Additional overrides to apply to the
            experiment config. Useful when minor code changes are incompatible
            with the checkpoint config.
        test_storage_path: The root path for storing results, which are saved
        to `{test_storage_path}/{config.name}/{trial_id}`.
        trial_id: The trial ID, used for storing results.
    """
    logger.info(f"Testing checkpoint {checkpoint!r}.")
    _, config = get_experiment_state(checkpoint)
    if not config:
        raise ValueError("Failed to get config from checkpoint.")
    config = apply_experiment_config_overrides(config, config_overrides or [])
    assert isinstance(config, DictConfig)

    results = test(
        *get_lightning_modules_from_config(config),
        checkpoint_path=get_checkpoint_file_path(to_local_checkpoint(checkpoint)),
        storage_path=f"{test_storage_path}/{config.name}/{trial_id or 'unspecified'}",
    )
    logger.info(f"Test results: {results}")

    return results
