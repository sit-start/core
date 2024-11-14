# util

[Core Index](../../README.md#core-index) / [sitstart](../index.md#sitstart) / [ml](./index.md#ml) / util

> Auto-generated documentation for [sitstart.ml.util](../../../python/sitstart/ml/util.py) module.

- [util](#util)
  - [SubmoduleAction](#submoduleaction)
  - [dedupe](#dedupe)
  - [gamma_correct](#gamma_correct)
  - [get_outputs_and_targets](#get_outputs_and_targets)
  - [get_parameters](#get_parameters)
  - [get_submodules](#get_submodules)
  - [get_transforms](#get_transforms)
  - [init_module](#init_module)
  - [init_parameter](#init_parameter)
  - [init_submodule](#init_submodule)
  - [one_hot](#one_hot)
  - [require_grad](#require_grad)
  - [require_grad_submodule](#require_grad_submodule)
  - [split_dataset](#split_dataset)
  - [update_module](#update_module)
  - [update_submodule](#update_submodule)

## SubmoduleAction

[Show source in util.py:135](../../../python/sitstart/ml/util.py#L135)

#### Signature

```python
class SubmoduleAction(Enum): ...
```



## dedupe

[Show source in util.py:439](../../../python/sitstart/ml/util.py#L439)

Remove duplicates from the input based on the given IDs.

Preserves the order of the input.

#### Signature

```python
def dedupe(input: Sequence[Any], ids: Sequence[Any]) -> Sequence[Any]: ...
```



## gamma_correct

[Show source in util.py:414](../../../python/sitstart/ml/util.py#L414)

Gamma-correct and normalize the input.

The result's L1 norm == `norm`. `norm` defaults to the number of unique
values in the input.

#### Signature

```python
def gamma_correct(
    input: list[Any] | torch.Tensor,
    gamma: float,
    norm: float | Literal["count"] | None = "count",
) -> torch.Tensor: ...
```



## get_outputs_and_targets

[Show source in util.py:485](../../../python/sitstart/ml/util.py#L485)

Compute outputs and targets for the given model and dataloader.

#### Signature

```python
def get_outputs_and_targets(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    max_num_batches: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]: ...
```



## get_parameters

[Show source in util.py:220](../../../python/sitstart/ml/util.py#L220)

Get parameters for the given target.

Accepts shell-style `target` patterns, processed with
`fnmatch.filter`.

#### Signature

```python
def get_parameters(module: nn.Module, target: str) -> dict[str, Parameter]: ...
```



## get_submodules

[Show source in util.py:209](../../../python/sitstart/ml/util.py#L209)

Get sub-modules for the given target.

Accepts shell-style `target` patterns, processed with
`fnmatch.filter`.

#### Signature

```python
def get_submodules(module: nn.Module, target: str) -> dict[str, nn.Module]: ...
```



## get_transforms

[Show source in util.py:304](../../../python/sitstart/ml/util.py#L304)

Get transforms for the given weights name.

See torchvision.models.get_weight for details.

#### Signature

```python
def get_transforms(name: str) -> Callable: ...
```



## init_module

[Show source in util.py:252](../../../python/sitstart/ml/util.py#L252)

Re-initialize the parameters of the given module.

If no initializer is provided, the module must implement
`reset_parameters`.

#### Signature

```python
def init_module(
    module: nn.Module,
    initializer: Callable[[nn.Module], None] | None = None,
    requires_grad: bool = True,
) -> nn.Module: ...
```



## init_parameter

[Show source in util.py:190](../../../python/sitstart/ml/util.py#L190)

Re-initialize the given named target parameter.

Accepts shell-style `target` patterns, processed with
`fnmatch.filter`.

#### Signature

```python
def init_parameter(
    module: nn.Module,
    target: str,
    initializer: ParameterInitializer | DictConfig,
    requires_grad: bool = True,
) -> None: ...
```



## init_submodule

[Show source in util.py:231](../../../python/sitstart/ml/util.py#L231)

Re-initialize parameters of the sub-module for the given target.

Accepts shell-style `target` patterns, processed with
`fnmatch.filter`.

If no initializer is provided, the sub-module must implement
`reset_parameters`.

#### Signature

```python
def init_submodule(
    module: nn.Module,
    target: str,
    initializer: ModuleInitializer | DictConfig | None = None,
    requires_grad: bool = True,
) -> None: ...
```



## one_hot

[Show source in util.py:453](../../../python/sitstart/ml/util.py#L453)

Compute a one-hot encoding of the given target.

Input target shape is [N, 1, *], and output shape, [N, num_classes, *].

#### Arguments

- `target` - Tensor of type long; one-hot encoding is saved into dimension 1
- `num_classes` - Number of classes; defaults to target.max()
- `dtype` - Result dtype; defaults to target.dtype

#### Signature

```python
def one_hot(
    target: torch.Tensor,
    num_classes: int | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor: ...
```



## require_grad

[Show source in util.py:297](../../../python/sitstart/ml/util.py#L297)

Set `requires_grad` for the module's parameters.

#### Signature

```python
def require_grad(module: nn.Module, requires_grad: bool = True) -> nn.Module: ...
```



## require_grad_submodule

[Show source in util.py:279](../../../python/sitstart/ml/util.py#L279)

Set `requires_grad` for all parameters of the `target` sub-module.

`requires_grad` is set to False if `target` is prefixed with '-' and
True otherwise.

Accepts shell-style `target` patterns, processed with
`fnmatch.filter`.

#### Signature

```python
def require_grad_submodule(module: nn.Module, target: str) -> nn.Module: ...
```



## split_dataset

[Show source in util.py:312](../../../python/sitstart/ml/util.py#L312)

Split dataset into training and validation datasets.

The returned pair of Subset instances wrap the original dataset
when train_transform and val_transform are None, or the
original and a duplicate dataset, copied via copy.deepcopy(),
with updated transforms otherwise.

#### Arguments

- `dataset` - Dataset to split; must be an instance of VisionDataset
    if transforms are provided.
- `train_split_size` - Fraction or number of samples to use for
    training. If ids are provided and `dedupe=False`, the split
    is approximate.
- `dataset_size` - Fraction or number of samples to use for both
    training and validation. Defaults to the full dataset size.
- `ids` - List of `len(dataset)` sample IDs. Used for removing all
    but the first occurence of each ID when `dedupe=True`, and
    splitting the dataset by ID when `dedupe=False`. Defaults to
    `range(len(dataset))`.
- `seed` - Seed for random number generation, for shuffling / splitting.
- `train_transform` - Alternative transform to apply to training images.
    If None, dataset.transform is used.
- `val_transform` - Alternative transform to apply to validation images.
    If None, dataset.transform is used.

#### Signature

```python
def split_dataset(
    dataset: Dataset,
    train_split_size: float | int,
    dataset_size: float | int | None = None,
    ids: list[Any] | None = None,
    seed: int | None = None,
    train_transform: Callable | None = None,
    val_transform: Callable | None = None,
    dedupe: bool = False,
) -> tuple[Subset, Subset]: ...
```



## update_module

[Show source in util.py:27](../../../python/sitstart/ml/util.py#L27)

Update sub-modules.

Supports freezing, replacing, appending to, and reinitializing sub-modules.

Useful for minor module changes, transfer learning, and fine-tuning. More
complex module changes should be implemented in a custom module.

Can be instantiated from a DictConfig, e.g., to add dropout to ResNet18,
to train on the 10-class CIFAR-10 dataset:

```yaml
_convert_: none
_recursive_: false
_target_: sitstart.ml.util.update_module
module:
  _target_: torchvision.models.resnet18
  num_classes: 10
append:
  layer*.*.bn2:
      - _target_: torch.nn.Dropout2d
      p: 0.25
```

or apply transfer learning to a pre-trained ResNet34 model,
training only the final fully connected layer:

```yaml
_convert_: none
_recursive_: false
_target_: sitstart.ml.util.update_module
require_grad: ["-"]
module:
  _target_: torchvision.models.resnet34
  weights:
    _target_: torchvision.models.get_weight
    name: ${trial.model.weights}
replace:
    fc:
    _target_: torch.nn.Linear
    in_features: 512
    out_features: ${trial.data.num_classes}
```

Since [update_module](#update_module) accepts Config objects as arguments,
`_recursive_: false` can be used for late instantiation
of arguments.

#### Arguments

- `module` - Module to update.
- [require_grad](#require_grad) - Names of sub-modules to train; sets
    `requires_grad` for all sub-module parameters to False
    if the name is prefixed with "-" and True otherwise.
    Defaults to the root sub-module, [""], which trains all
    sub-modules. Targets are processed in order, so, e.g.,
    to train all but 'layer1', use ["", "-layer1"], or
    only 'head', use ["-", "head"]. See `train_submodule`
    for details.
- `replace` - Names of sub-modules to replace and their
    replacements. Applied after the above step. See
    [update_submodule](#update_submodule) for details.
- `append` - Names of sub-modules to append to, and modules to
    append. Applied after the above steps. See
    [update_submodule](#update_submodule) for details.
- `init` - List of submodule name or (name, initializer) tuples
    for initialization. Applied after the above steps.
    See [init_module](#init_module) for details.
- `param_init` - List of (parameter name, initializer) tuples for
    initialization. Applied after the above steps. See
    [init_parameter](#init_parameter) for details.

#### Signature

```python
def update_module(
    module: nn.Module | ModuleCreator | DictConfig | None = None,
    require_grad: list[str] | ListConfig | None = None,
    replace: dict[str, nn.Module | ModuleCreator] | DictConfig | None = None,
    append: dict[str, nn.Module | ModuleCreator] | DictConfig | None = None,
    init: dict[str, ModuleInitializer | None] | DictConfig | None = None,
    param_init: dict[str, ParameterInitializer] | DictConfig | None = None,
) -> nn.Module: ...
```



## update_submodule

[Show source in util.py:140](../../../python/sitstart/ml/util.py#L140)

Update the sub-module for the given target.

Replaces the sub-module with `value` if `action` is "replace".

Appends `value` to the sub-module if `action` is "append".

Accepts shell-style `target` patterns, processed with
`fnmatch.filter`.

If `target` resolves to more than one sub-module and the given
`value` is of type `nn.Module`, sub-modules after the first
are updated with `copy.deepcopy(value)`.

Compliments `torch.nn.Module.get_submodule`; see its documentation
for details.

#### Signature

```python
def update_submodule(
    action: str | SubmoduleAction,
    module: nn.Module,
    target: str,
    value: nn.Module | ModuleCreator | DictConfig,
) -> None: ...
```
