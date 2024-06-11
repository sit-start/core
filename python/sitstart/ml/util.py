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
    freeze: list[str] | ListConfig | None = None,
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
      freeze: [""]
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
        freeze: Names of sub-modules to freeze. All initialized
            sub-modules will be unfrozen. Defaults to no sub-modules;
            use `freeze=[""]` to freeze the entire module. Prefixing
            the module name with '-' will unfreeze it.
        replace: Names of sub-modules to replace and their
            replacements. See `update_submodule` for details.
        append: Names of sub-modules to append to, and modules to
            append. See `update_submodule` for details.
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

    freeze = freeze or []
    unfreeze = [target[1:] for target in freeze if target.startswith("-")]
    freeze = [target for target in freeze if not target.startswith("-")]
    for target in freeze:
        freeze_module(module.get_submodule(target))
    for target in unfreeze:
        unfreeze_module(module.get_submodule(target))

    for target, value in (append or {}).items():
        update_submodule(SubmoduleAction.APPEND, module, str(target), value)

    for target, value in (replace or {}).items():
        update_submodule(SubmoduleAction.REPLACE, module, str(target), value)

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
    all_targets, _ = zip(*module.named_modules())
    filtered_targets = fnmatch.filter(all_targets, target)
    if not filtered_targets:
        raise ValueError(f"No sub-modules found for target {target!r}.")
    needs_copy = False

    for filtered_target in filtered_targets:
        target_module = module.get_submodule(filtered_target)
        keys = filtered_target.split(".")
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
            setattr(parent, keys[-1], nn.Sequential(target_module, val))
        elif action == SubmoduleAction.REPLACE:
            setattr(parent, keys[-1], val)
        else:
            raise ValueError("Action must be 'append' or 'replace'.")


def init_parameter(
    module: nn.Module,
    target: str,
    initializer: ParameterInitializer | DictConfig,
    unfreeze: bool = True,
) -> None:
    """Re-initialize the given named target parameter.

    Accepts shell-style `target` patterns, processed with
    `fnmatch.filter`.
    """
    if isinstance(initializer, DictConfig):
        initializer = cast(ParameterInitializer, instantiate(initializer))

    all_targets, _ = zip(*module.named_parameters())
    filtered_targets = fnmatch.filter(all_targets, target)

    for filtered_target in filtered_targets:
        param = module.get_parameter(filtered_target)
        initializer(param)
        if unfreeze:
            param.requires_grad = True


def init_submodule(
    module: nn.Module,
    target: str,
    initializer: ModuleInitializer | DictConfig | None = None,
    unfreeze: bool = True,
) -> None:
    """Re-initialize parameters of the sub-module for the given target.

    Accepts shell-style `target` patterns, processed with
    `fnmatch.filter`.

    If no initializer is provided, the sub-module must implement
    `reset_parameters`.
    """
    all_targets, _ = zip(*module.named_modules())
    filtered_targets = fnmatch.filter(all_targets, target)

    if isinstance(initializer, DictConfig):
        initializer = cast(ModuleInitializer, instantiate(initializer))

    for filtered_target in filtered_targets:
        init_module(module.get_submodule(filtered_target), initializer, unfreeze)


def init_module(
    module: nn.Module,
    initializer: Callable[[nn.Module], None] | None = None,
    unfreeze: bool = True,
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

    if unfreeze:
        unfreeze_module(module)

    return module


def freeze_module(module: nn.Module) -> nn.Module:
    """Freeze all parameters of a module."""
    for param in module.parameters():
        param.requires_grad = False
    return module


def unfreeze_module(module: nn.Module) -> nn.Module:
    """Unfreeze all parameters of a module."""
    for param in module.parameters():
        param.requires_grad = True
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
            training. If ids are provided, the split is approximate.
        dataset_size: Fraction or number of samples to use for both
            training and validation. Defaults to the full dataset size.
        ids: List of IDs to be split. If None, dataset is split by index.
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
    """Returns a WeightedRandomSampler that rebalances the given classes."""
    el_cls = element_class
    el_cls = el_cls.flatten().tolist() if isinstance(el_cls, torch.Tensor) else el_cls

    classes = list(dict.fromkeys(el_cls))
    class_counts = [el_cls.count(c) for c in classes]
    class_inv_freq = [len(el_cls) / count for count in class_counts]
    el_inv_freq = [class_inv_freq[classes.index(c)] for c in el_cls]

    return WeightedRandomSampler(el_inv_freq, len(el_cls), generator=generator)
