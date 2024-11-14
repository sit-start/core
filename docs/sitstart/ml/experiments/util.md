# util

[Core Index](../../../README.md#core-index) / [sitstart](../../index.md#sitstart) / [ml](../index.md#ml) / [experiments](./index.md#experiments) / util

> Auto-generated documentation for [sitstart.ml.experiments.util](../../../../python/sitstart/ml/experiments/util.py) module.

#### Attributes

- `TUNE_SEARCH_SPACE_API` - https://docs.ray.io/en/latest/tune/api/search_space.html: ['ray.tune.uniform', 'ray.tune.quniform', 'ray.tune.loguniform', 'ray.tune.qloguniform', 'ray.tune.randn', 'ray.tune.qrandn', 'ray.tune.randint', 'ray.tune.qrandint', 'ray.tune.lograndint', 'ray.tune.qlograndint', 'ray.tune.choice', 'ray.tune.sample_from', 'ray.tune.grid_search']


- [util](#util)
  - [TrialResolution](#trialresolution)
  - [apply_experiment_config_overrides](#apply_experiment_config_overrides)
  - [get_default_storage_path](#get_default_storage_path)
  - [get_experiment_wandb_url](#get_experiment_wandb_url)
  - [get_lightning_modules_from_config](#get_lightning_modules_from_config)
  - [get_param_space_description](#get_param_space_description)
  - [get_search_alg](#get_search_alg)
  - [load_experiment_config](#load_experiment_config)
  - [register_omegaconf_resolvers](#register_omegaconf_resolvers)
  - [resolve](#resolve)
  - [sample_param_space](#sample_param_space)
  - [validate_experiment_config](#validate_experiment_config)
  - [validate_trial_config](#validate_trial_config)

## TrialResolution

[Show source in util.py:179](../../../../python/sitstart/ml/experiments/util.py#L179)

#### Signature

```python
class TrialResolution(str, Enum): ...
```



## apply_experiment_config_overrides

[Show source in util.py:86](../../../../python/sitstart/ml/experiments/util.py#L86)

Apply overrides to the given experiment config.

#### Signature

```python
def apply_experiment_config_overrides(
    config: Container, overrides: list[str]
) -> Container: ...
```



## get_default_storage_path

[Show source in util.py:349](../../../../python/sitstart/ml/experiments/util.py#L349)

Returns the default storage path for experiments.

#### Signature

```python
def get_default_storage_path() -> str: ...
```



## get_experiment_wandb_url

[Show source in util.py:97](../../../../python/sitstart/ml/experiments/util.py#L97)

Get the Weights & Biases URL for the experiment config.

#### Signature

```python
def get_experiment_wandb_url(config: Container) -> str | None: ...
```



## get_lightning_modules_from_config

[Show source in util.py:318](../../../../python/sitstart/ml/experiments/util.py#L318)

Get the data and training modules from the given main and train loop configs.

#### Signature

```python
def get_lightning_modules_from_config(
    config: DictConfig,
    train_loop_config: dict[str, Any] | None = None,
    num_workers: int = 1,
) -> tuple[pl.LightningDataModule, pl.LightningModule]: ...
```



## get_param_space_description

[Show source in util.py:293](../../../../python/sitstart/ml/experiments/util.py#L293)

Returns a one-line description of the config's parameter search space.

#### Signature

```python
def get_param_space_description(
    config: Container, param_space_key: str = DEFAULT_PARAM_SPACE_KEY
) -> str: ...
```

#### See also

- [DEFAULT_PARAM_SPACE_KEY](#default_param_space_key)



## get_search_alg

[Show source in util.py:109](../../../../python/sitstart/ml/experiments/util.py#L109)

Get the search algorithm from the experiment config.

#### Signature

```python
def get_search_alg(
    config: Container,
    search_alg_key: str = "tune.search_alg",
    metric_key: str = "eval.select.metric",
    mode_key: str = "eval.select.mode",
) -> SearchAlgorithm: ...
```



## load_experiment_config

[Show source in util.py:66](../../../../python/sitstart/ml/experiments/util.py#L66)

Load an experiment config. For testing/debugging.

#### Signature

```python
def load_experiment_config(
    name: str, overrides: list[str] | None = None, config_path: str | None = None
) -> Container: ...
```



## register_omegaconf_resolvers

[Show source in util.py:44](../../../../python/sitstart/ml/experiments/util.py#L44)

Register OmegaConf resolvers.

Calls [register_omegaconf_resolvers](../../util/hydra.md#register_omegaconf_resolvers) and
additional registers resolvers for Ray Tune search space API
functions under the `rt` namespace.

#### Signature

```python
@once
def register_omegaconf_resolvers(): ...
```

#### See also

- [once](../../util/decorators.md#once)



## resolve

[Show source in util.py:188](../../../../python/sitstart/ml/experiments/util.py#L188)

Resolve an OmegaConf config in-place.

#### Arguments

- `config` - The config to resolve.
- `param_space_key` - The key for the parameter search space node.
- `trial_key` - The key for the trial node.
- `resolve_trial` - How to handle the trial node's parameters:
    - `-` *"resolve"* - Resolve the trial node's parameters.
    - `-` *"exclude"* - Exclude the trial node's parameters from resolution.
    - `-` *"sample"* - Sample the parameter search space for the trial node.

#### Signature

```python
def resolve(
    config: Container,
    resolve_trial: str | TrialResolution = "resolve",
    param_space_key: str = DEFAULT_PARAM_SPACE_KEY,
    trial_key: str = "trial",
) -> None: ...
```

#### See also

- [DEFAULT_PARAM_SPACE_KEY](#default_param_space_key)



## sample_param_space

[Show source in util.py:134](../../../../python/sitstart/ml/experiments/util.py#L134)

Temporarily resolves the search space for the given config.

Uses the first sample from the config's search algorithm.

#### Signature

```python
def sample_param_space(
    config: Container, param_space_key: str = DEFAULT_PARAM_SPACE_KEY
) -> None: ...
```

#### See also

- [DEFAULT_PARAM_SPACE_KEY](#default_param_space_key)



## validate_experiment_config

[Show source in util.py:226](../../../../python/sitstart/ml/experiments/util.py#L226)

Validate the given experiment config.

Requires the following:
- The trial key must specify a root config node.
- No direct or indirect dependencies on the parameter space
  node appear outside the trial node.
- All parameter samplers and grid searches must be in the
  parameter space node.

Raises a ValueError if any of the above conditions are not met.

#### Signature

```python
def validate_experiment_config(
    config: Container,
    param_space_key: str = DEFAULT_PARAM_SPACE_KEY,
    trial_key: str = "trial",
) -> None: ...
```

#### See also

- [DEFAULT_PARAM_SPACE_KEY](#default_param_space_key)



## validate_trial_config

[Show source in util.py:281](../../../../python/sitstart/ml/experiments/util.py#L281)

Validate the trial node of the given config.

#### Signature

```python
def validate_trial_config(config: Container) -> None: ...
```
