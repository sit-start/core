import sys
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig
from ray.tune.experiment.trial import Trial

from sitstart.util.string import to_str


def get_project_name(config: DictConfig) -> str:
    return config.get("name", Path(sys.argv[0]).stem)


def get_group_name() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_trial_dirname(trial: Trial) -> str:
    return trial.trial_id


def get_trial_name(trial: Trial, incl_params: bool = False) -> str:
    # trial_id is of the form <run_id>_<trial_num>. The unique run_id
    # distinguishes trials from different runs in the same group, and
    # the trial_num, from different trials in the same run. This is
    # particularly important if we're allowing multiple runs to use
    # the same group when those runs may be different.
    if not incl_params:
        return trial.trial_id

    # trial.evaluated_params includes the 'path' of the variable in the
    # config, e.g., train_loop_config/lr=1e-1. We drop the path here
    # unless there's a collision between variable names.
    var_to_params = {}
    for k, v in trial.evaluated_params.items():
        var = k.split("/")[-1]
        var_to_params.setdefault(var, []).append([k, v])
    params = {}
    for k, v in var_to_params.items():
        if len(v) == 1:
            params[k] = v[0][1]
        else:
            for k0, v0 in v:
                params[k0] = v0

    param_str = to_str(
        params, precision=4, list_sep=",", dict_sep=",", dict_kv_sep="=", use_repr=False
    )[1:-1]
    trial_id = trial.trial_id
    return trial_id if not param_str else f"{trial_id}({param_str})"
