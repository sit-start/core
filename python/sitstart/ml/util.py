import copy
import fnmatch
from enum import Enum
from typing import Any, Callable, cast

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from torch import nn, randperm
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, Sampler, Subset, WeightedRandomSampler
from torchvision.datasets import VisionDataset
from torchvision.datasets.vision import StandardTransform
from torchvision.models import get_weight

from sitstart.logging import get_logger

ModuleCreator = Callable[[], nn.Module]
ModuleInitializer = Callable[[nn.Module], None]
ParameterInitializer = Callable[[Parameter], None]

logger = get_logger(__name__)


def update_module(
    module: nn.Module | ModuleCreator | DictConfig | None = None,
    require_grad: list[str] | ListConfig | None = None,
    replace: dict[str, nn.Module | ModuleCreator] | DictConfig | None = None,
    append: dict[str, nn.Module | ModuleCreator] | DictConfig | None = None,
    init: dict[str, ModuleInitializer | None] | DictConfig | None = None,
    param_init: dict[str, ParameterInitializer] | DictConfig | None = None,
) -> nn.Module:
    """Update sub-modules.

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

    Since `update_module` accepts Config objects as arguments,
    `_recursive_: false` can be used for late instantiation
    of arguments.

    Args:
        module: Module to update.
        require_grad: Names of sub-modules to train; sets
            `requires_grad` for all sub-module parameters to False
            if the name is prefixed with "-" and True otherwise.
            Defaults to the root sub-module, [""], which trains all
            sub-modules. Targets are processed in order, so, e.g.,
            to train all but 'layer1', use ["", "-layer1"], or
            only 'head', use ["-", "head"]. See `train_submodule`
            for details.
        replace: Names of sub-modules to replace and their
            replacements. Applied after the above step. See
            `update_submodule` for details.
        append: Names of sub-modules to append to, and modules to
            append. Applied after the above steps. See
            `update_submodule` for details.
        init: List of submodule name or (name, initializer) tuples
            for initialization. Applied after the above steps.
            See `init_module` for details.
        param_init: List of (parameter name, initializer) tuples for
            initialization. Applied after the above steps. See
            `init_parameter` for details.
    """
    if isinstance(module, nn.Module):
        module = module
    elif callable(module):
        module = module()
    elif isinstance(module, DictConfig):
        module = instantiate(module)
    else:
        raise ValueError(
            f"Expected nn.Module, Callable, or DictConfig. Got {type(module)})."
        )
    assert isinstance(module, nn.Module)

    for target in [""] if require_grad is None else require_grad:
        require_grad_submodule(module, target)

    for target, value in (replace or {}).items():
        update_submodule(SubmoduleAction.REPLACE, module, str(target), value)

    for target, value in (append or {}).items():
        update_submodule(SubmoduleAction.APPEND, module, str(target), value)

    for target, initializer in (init or {}).items():
        init_submodule(module, str(target), initializer)

    for target, initializer in (param_init or {}).items():
        init_parameter(module, str(target), initializer)

    return module


class SubmoduleAction(Enum):
    REPLACE = "replace"
    APPEND = "append"


def update_submodule(
    action: str | SubmoduleAction,
    module: nn.Module,
    target: str,
    value: nn.Module | ModuleCreator | DictConfig,
) -> None:
    """Update the sub-module for the given target.

    Replaces the sub-module with `value` if `action` is "replace".

    Appends `value` to the sub-module if `action` is "append".

    Accepts shell-style `target` patterns, processed with
    `fnmatch.filter`.

    If `target` resolves to more than one sub-module and the given
    `value` is of type `nn.Module`, sub-modules after the first
    are updated with `copy.deepcopy(value)`.

    Compliments `torch.nn.Module.get_submodule`; see its documentation
    for details.
    """
    action = SubmoduleAction(action) if isinstance(action, str) else action
    submodules = get_submodules(module, target)
    if not submodules:
        raise ValueError(f"No sub-modules found for target {target!r}.")
    needs_copy = False

    for name, submodule in submodules.items():
        keys = name.split(".")
        parent = module.get_submodule(".".join(keys[:-1]))

        if isinstance(value, nn.Module):
            val = copy.deepcopy(value) if needs_copy else value
            needs_copy = True
        elif callable(value):
            val = value()
        elif isinstance(value, DictConfig):
            val = instantiate(value)
        else:
            raise ValueError("Value must be an nn.Module, Callable, or DictConfig.")

        if action == SubmoduleAction.APPEND:
            setattr(parent, keys[-1], nn.Sequential(submodule, val))
        elif action == SubmoduleAction.REPLACE:
            setattr(parent, keys[-1], val)
        else:
            raise ValueError("Action must be 'append' or 'replace'.")


def init_parameter(
    module: nn.Module,
    target: str,
    initializer: ParameterInitializer | DictConfig,
    requires_grad: bool = True,
) -> None:
    """Re-initialize the given named target parameter.

    Accepts shell-style `target` patterns, processed with
    `fnmatch.filter`.
    """
    if isinstance(initializer, DictConfig):
        initializer = cast(ParameterInitializer, instantiate(initializer))

    for param in get_parameters(module, target).values():
        initializer(param)
        param.requires_grad = requires_grad


def get_submodules(module: nn.Module, target: str) -> dict[str, nn.Module]:
    """Get sub-modules for the given target.

    Accepts shell-style `target` patterns, processed with
    `fnmatch.filter`.
    """
    named_modules = {name: module for name, module in module.named_modules()}
    filtered_names = fnmatch.filter(named_modules.keys(), target)
    return {name: named_modules[name] for name in filtered_names}


def get_parameters(module: nn.Module, target: str) -> dict[str, Parameter]:
    """Get parameters for the given target.

    Accepts shell-style `target` patterns, processed with
    `fnmatch.filter`.
    """
    named_params = {name: param for name, param in module.named_parameters()}
    filtered_names = fnmatch.filter(named_params.keys(), target)
    return {name: named_params[name] for name in filtered_names}


def init_submodule(
    module: nn.Module,
    target: str,
    initializer: ModuleInitializer | DictConfig | None = None,
    requires_grad: bool = True,
) -> None:
    """Re-initialize parameters of the sub-module for the given target.

    Accepts shell-style `target` patterns, processed with
    `fnmatch.filter`.

    If no initializer is provided, the sub-module must implement
    `reset_parameters`.
    """
    if isinstance(initializer, DictConfig):
        initializer = cast(ModuleInitializer, instantiate(initializer))

    for submodule in get_submodules(module, target).values():
        init_module(submodule, initializer, requires_grad=requires_grad)


def init_module(
    module: nn.Module,
    initializer: Callable[[nn.Module], None] | None = None,
    requires_grad: bool = True,
) -> nn.Module:
    """Re-initialize the parameters of the given module.

    If no initializer is provided, the module must implement
    `reset_parameters`.
    """

    def default_init(module: nn.Module) -> None:
        if not hasattr(module, "reset_parameters"):
            raise ValueError(
                "Initializer must be provided or module must implement "
                "`reset_parameters`."
            )
        module.reset_parameters()

    initializer = initializer or default_init
    initializer(module)

    require_grad(module, requires_grad)

    return module


def require_grad_submodule(module: nn.Module, target: str) -> nn.Module:
    """Set `requires_grad` for all parameters of the `target` sub-module.

    `requires_grad` is set to False if `target` is prefixed with '-' and
    True otherwise.

    Accepts shell-style `target` patterns, processed with
    `fnmatch.filter`.
    """
    requires_grad = not target.startswith("-")
    target = target if requires_grad else target[1:]

    for submodule in get_submodules(module, target).values():
        require_grad(submodule, requires_grad=requires_grad)

    return module


def require_grad(module: nn.Module, requires_grad: bool = True) -> nn.Module:
    """Set `requires_grad` for the module's parameters."""
    for param in module.parameters():
        param.requires_grad = requires_grad
    return module


def get_transforms(name: str) -> Callable:
    """Get transforms for the given weights name.

    See torchvision.models.get_weight for details.
    """
    return get_weight(name).transforms()


def hash_tensor(x: torch.Tensor) -> int:
    return hash(tuple(x.cpu().reshape(-1).tolist()))


def split_dataset(
    dataset: Dataset,
    train_split_size: float | int,
    dataset_size: float | int | None = None,
    ids: list[Any] | None = None,
    generator: torch.Generator | None = None,
    train_transform: Callable | None = None,
    val_transform: Callable | None = None,
    dedupe: bool = False,
) -> tuple[Subset, Subset]:
    """Split dataset into training and validation datasets.

    The returned pair of Subset instances wrap the original dataset
    when train_transform and val_transform are None, or the
    original and a duplicate dataset, copied via copy.deepcopy(),
    with updated transforms otherwise.

    Args:
        dataset: Dataset to split; must be an instance of VisionDataset
            if transforms are provided.
        train_split_size: Fraction or number of samples to use for
            training. If ids are provided and `dedupe=False`, the split
            is approximate.
        dataset_size: Fraction or number of samples to use for both
            training and validation. Defaults to the full dataset size.
        ids: List of `len(dataset)` sample IDs. Used for removing all
            but the first occurence of each ID when `dedupe=True`, and
            splitting the dataset by ID when `dedupe=False`. Defaults to
            `range(len(dataset))`.
        generator: Random number generator for shuffling IDs.
        train_transform: Alternative transform to apply to training images.
            If None, dataset.transform is used.
        val_transform: Alternative transform to apply to validation images.
            If None, dataset.transform is used.
    """
    if not hasattr(dataset, "__len__"):
        raise ValueError("Dataset must implement __len__")
    if (train_transform or val_transform) and not isinstance(dataset, VisionDataset):
        msg = "Transforms can only be applied to instances of VisionDataset."
        raise ValueError(msg)

    n_input_data = len(dataset)  # type: ignore
    n_data = n_input_data
    if dataset_size is not None:
        logger.info(f"Using {dataset_size} samples from input of size {n_input_data}.")
        n_data = n_input_data * dataset_size if dataset_size <= 1 else dataset_size
    n_data = min(int(n_data), n_input_data)

    if ids and len(ids) != n_input_data:
        raise ValueError("Length of IDs must match length of dataset.")
    ids = ids[:n_data] if ids else list(range(n_data))
    unique_ids = sorted(list(set(ids)))
    n_ids = len(unique_ids)

    id_idx_to_data_idx = {}
    for data_idx, id_ in enumerate(ids):
        id_idx_to_data_idx.setdefault(id_, []).append(data_idx)

    if dedupe:
        n_data = len(unique_ids)
        logger.info(f"After de-duplication, dataset size is {n_data}.")
        for id_idx, data_idx in id_idx_to_data_idx.items():
            id_idx_to_data_idx[id_idx] = data_idx[:1]
        assert len(id_idx_to_data_idx) == n_data

    n_train = n_data * train_split_size if train_split_size <= 1 else train_split_size
    n_train = min(int(n_train), n_data)
    n_val = n_data - n_train

    id_indices = randperm(n_ids, generator=generator).tolist()

    train_indices, val_indices = [], []
    for id_idx in id_indices:
        if len(train_indices) < n_train:
            train_indices += id_idx_to_data_idx[unique_ids[id_idx]]
        elif len(val_indices) < n_val:
            val_indices += id_idx_to_data_idx[unique_ids[id_idx]]
        else:
            break

    if not train_indices or not val_indices:
        return Subset(dataset, train_indices), Subset(dataset, val_indices)

    assert isinstance(dataset, VisionDataset)
    train_transform = train_transform or dataset.transform
    val_transform = val_transform or dataset.transform
    target_transform = dataset.target_transform

    dataset.transforms = StandardTransform(train_transform, target_transform)
    dataset.transform = train_transform
    train = Subset(dataset, train_indices)

    dataset = copy.deepcopy(dataset)
    dataset.transforms = StandardTransform(val_transform, target_transform)
    dataset.transform = val_transform
    val = Subset(dataset, val_indices)

    return train, val


def rebalancing_sampler(
    element_class: list[Any] | torch.Tensor, generator: torch.Generator | None = None
) -> Sampler:
    """Create a WeightedRandomSampler that rebalances the given classes."""
    el_cls = element_class
    el_cls = el_cls.flatten().tolist() if isinstance(el_cls, torch.Tensor) else el_cls

    classes = list(dict.fromkeys(el_cls))
    class_counts = [el_cls.count(c) for c in classes]
    class_inv_freq = [len(el_cls) / count for count in class_counts]
    el_inv_freq = [class_inv_freq[classes.index(c)] for c in el_cls]

    return WeightedRandomSampler(el_inv_freq, len(el_cls), generator=generator)


# adapted from
# @source: https://github.com/PhoenixDL/rising/blob/master/rising/transforms/functional/channel.py
def one_hot(
    target: torch.Tensor,
    num_classes: int | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Compute a one-hot encoding of the given target.

    Input target shape is [N, 1, *], and output shape, [N, num_classes, *].

    Args:
        target: Tensor of type long; one-hot encoding is saved into dimension 1
        num_classes: Number of classes; defaults to target.max()
        dtype: Result dtype; defaults to target.dtype
    """
    if target.dtype != torch.long:
        raise TypeError(
            f"Target tensor needs to be of type torch.long, found {target.dtype}"
        )
    if target.ndim < 2 or target.shape[1] != 1:
        raise ValueError(f"Expected target shape [N, 1, *], found {target.shape}.")

    num_classes = num_classes or int(target.max().item() + 1)
    target_onehot = torch.zeros(
        size=(target.shape[0], num_classes, *target.shape[2:]),
        dtype=dtype or target.dtype,
        device=target.device,
    )

    return target_onehot.scatter_(1, target, 1.0)


def get_outputs_and_targets(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    max_num_batches: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute outputs and targets for the given model and dataloader."""
    logger.info("Computing outputs and targets.")
    max_num_batches = min(max_num_batches or len(dataloader), len(dataloader))
    device = model.device
    if max_num_batches == 0:
        return torch.empty(0).to(device), torch.empty(0).to(device)

    outputs, targets = [], []
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(dataloader):
            if i >= max_num_batches:
                break
            outputs.append(model(input.to(device)))
            targets.append(target.to(device))

    return torch.concat(outputs), torch.concat(targets)
