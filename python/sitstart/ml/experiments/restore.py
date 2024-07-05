import json
from pathlib import Path
from typing import Any, Literal

import boto3
import pyarrow.fs
from cloudpathlib import CloudPath, S3Client, S3Path
from omegaconf import DictConfig, OmegaConf
from ray.train import Checkpoint, Result

from sitstart.aws.util import update_aws_env
from sitstart.cloudpathlib.util import get_local_path
from sitstart.logging import get_logger
from sitstart.ml.experiments import RUN_ROOT
from sitstart.ml.experiments.name import get_group_name_from_run_name
from sitstart.ml.experiments.util import get_default_storage_path
from sitstart.scm.git.repo_state import RepoState
from sitstart.util.decorators import memoize

logger = get_logger(__name__)


@memoize
def _get_aws_session(profile: str | None = None) -> boto3.Session:
    return update_aws_env(profile=profile)


def _get_run_group_from_trial(
    storage_path: str, project_name: str, trial_id: str
) -> str | None:
    """Returns the run group for the given project and trial."""
    result = None
    for run_path in get_cloud_path(storage_path).glob(f"{project_name}_*"):
        for _ in run_path.glob(f"{trial_id}*"):
            if result:
                raise ValueError(
                    f"Multiple run groups found for project {project_name} "
                    f"and trial {trial_id}."
                )
            result = get_group_name_from_run_name(project_name, run_path.name)
    return result


def _get_trial_path_from_trial(
    project_name: str,
    storage_path: str,
    trial_id: str,
    run_group: str | None = None,
) -> str | None:
    if not trial_id:
        return None
    run_group = run_group or _get_run_group_from_trial(
        storage_path, project_name, trial_id
    )
    if not run_group:
        return None
    project_path = f"{storage_path}/{project_name}"

    return f"{project_path}_{run_group}/{trial_id}"


def _get_checkpoint_path_from_trial(
    project_name: str,
    storage_path: str,
    trial_id: str,
    run_group: str | None = None,
    select: Literal["best", "last"] = "last",
    select_metric: str | None = None,
    select_mode: str | None = None,
) -> str | None:
    """Returns the checkpoint directory path"""
    if select not in ("best", "last"):
        raise ValueError(f"Invalid value for select: {select}")
    if select == "best" and (select_metric is None or select_mode is None):
        raise ValueError(
            "select_metric and select_mode must be provided when select='best'."
        )

    trial_path = _get_trial_path_from_trial(
        project_name=project_name,
        storage_path=storage_path,
        trial_id=trial_id,
        run_group=run_group,
    )
    if not trial_path:
        return None

    _ = _get_aws_session()
    result = Result.from_path(trial_path)

    ckpt = result.checkpoint
    if select == "best":
        assert select_metric is not None and select_mode is not None
        ckpt = result.get_best_checkpoint(select_metric, select_mode)

    if not ckpt:
        return None

    fs_prefix = {"s3": "s3://", "gcs": "gs://"}
    fs = ckpt.filesystem
    prefix = fs_prefix.get(fs.type_name, "") if fs else ""

    return prefix + ckpt.path


def _to_path_str(path_on_filesystem: str, filesystem: pyarrow.fs.FileSystem) -> str:
    type_name_to_prefix = {"s3": "s3://", "gs": "gs://", "local": ""}
    if filesystem.type_name not in type_name_to_prefix:
        raise ValueError(f"Unsupported filesystem type: {filesystem.type_name}")
    return type_name_to_prefix[filesystem.type_name] + path_on_filesystem


def _get_trial_params_from_trial_path(trial_path: CloudPath | Path) -> dict[str, Any]:
    params_path = get_local_path(trial_path / "params.json")
    return json.loads(params_path.read_text())


def to_local_checkpoint(checkpoint: Checkpoint) -> Checkpoint:
    """Converts a remote checkpoint to a local checkpoint.

    Unlike `ray.train.Checkpoint.to_directory()`, this uses a local
    cache.
    """
    ckpt_path = _to_path_str(checkpoint.path, checkpoint.filesystem)
    cloud_ckpt_path = get_cloud_path(ckpt_path)
    local_ckpt_path = get_local_path(cloud_ckpt_path)

    suffix = f" from remote checkpoint {cloud_ckpt_path}"
    suffix = suffix if cloud_ckpt_path != local_ckpt_path else ""
    logger.info(f"Loading checkpoint at {local_ckpt_path}{suffix}")

    return Checkpoint(local_ckpt_path)


def get_trial_params(
    project_name: str,
    trial_id: str,
    storage_path: str | None = None,
    run_group: str | None = None,
) -> dict[str, Any]:
    logger.info(
        f"Getting trial parameters for trial {trial_id!r} in project {project_name!r}."
    )
    trial_path = _get_trial_path_from_trial(
        project_name=project_name,
        storage_path=storage_path or get_default_storage_path(),
        trial_id=trial_id,
        run_group=run_group,
    )
    if not trial_path:
        raise ValueError(
            f"Params for trial {trial_id!r} not found in project {project_name!r}."
        )

    return _get_trial_params_from_trial_path(get_cloud_path(trial_path))


def get_checkpoint(
    project_name: str,
    trial_id: str,
    storage_path: str | None = None,
    run_group: str | None = None,
    select: Literal["best", "last"] = "last",
    select_metric: str | None = None,
    select_mode: str | None = None,
    to_local: bool = False,
) -> Checkpoint | None:
    """Get the checkpoint for the given trial.

    Remote checkpoints not in the local cache are downloaded.
    `storage_path` defaults to `get_storage_path()`. If `run_group`
    isn't specified, the first run group in the storage path containing
    the given trial ID is used.

    Defaults to the last checkpoint.
    """
    logger.info(
        f"Getting checkpoint for trial {trial_id!r} in project {project_name!r}."
    )
    ckpt_path = _get_checkpoint_path_from_trial(
        project_name=project_name,
        storage_path=storage_path or get_default_storage_path(),
        run_group=run_group,
        trial_id=trial_id,
        select=select,
        select_metric=select_metric,
        select_mode=select_mode,
    )
    if not ckpt_path:
        logger.info(f"Checkpoint not found for trial {trial_id!r}.")
        return None

    ckpt = Checkpoint(ckpt_path)
    return to_local_checkpoint(ckpt) if to_local else ckpt


def get_checkpoint_from_config(
    config: DictConfig, to_local: bool = False
) -> Checkpoint | None:
    """Get the checkpoint for the given config.

    Remote checkpoints not in the local cache are downloaded.
    """
    if not (ckpt_path := config.restore.checkpoint_path):
        ckpt_path = _get_checkpoint_path_from_trial(
            project_name=config.name,
            storage_path=config.storage_path,
            run_group=config.restore.run.group,
            trial_id=config.restore.run.trial_id,
            select=config.restore.run.get("select", "last"),
            select_metric=config.eval.select.metric,
            select_mode=config.eval.select.mode,
        )
    if not ckpt_path:
        logger.info(f"Checkpoint not found for config with name {config.name!r}.")
        return None

    ckpt = Checkpoint(ckpt_path)
    return to_local_checkpoint(ckpt) if to_local else ckpt


def get_checkpoint_file_path(checkpoint: Checkpoint) -> str:
    return f"{checkpoint.path}/checkpoint.ckpt"


def get_cloud_path(
    path: str, local_cache_dir: str | None = RUN_ROOT, aws_profile: str | None = None
) -> CloudPath | Path:
    """Returns a `CloudPath` object for the given path string"""
    if path.startswith("s3://"):
        session = _get_aws_session(aws_profile)
        client = S3Client(local_cache_dir=local_cache_dir, boto3_session=session)
        return S3Path(path, client=client)
    elif path.startswith("gs://"):
        raise NotImplementedError("Google Cloud Storage is not implemented.")
    return Path(path)


def get_experiment_state(
    checkpoint: Checkpoint,
) -> tuple[RepoState | None, DictConfig | None]:
    """Get the repository state and config from the given checkpoint.

    If `checkpoint` was created at a remote storage path, the remote
    `Checkpoint` should be provided here, not, e.g., the output of
    `to_local_checkpoint()`.
    """
    # grab the original checkpoint path before converting to local
    # so we can access the trial params in the root trial directory
    logger.info(f"Loading experiment state from checkpoint {checkpoint}.")
    ckpt_path = _to_path_str(checkpoint.path, checkpoint.filesystem)

    checkpoint = to_local_checkpoint(checkpoint)
    metadata = checkpoint.get_metadata()
    repo_state_dict = metadata.get("repo_state")
    repo_state = RepoState.from_dict(repo_state_dict) if repo_state_dict else None

    config = DictConfig(metadata.get("config"))
    trial_params = _get_trial_params_from_trial_path(get_cloud_path(ckpt_path).parent)
    config.param_space = trial_params
    logger.info(f"Loaded trial parameters:\n{OmegaConf.to_yaml(config.param_space)}")

    return repo_state, config
