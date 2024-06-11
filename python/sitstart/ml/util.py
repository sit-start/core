import copy
from typing import Any, Callable

import torch
from torch import randperm
from torch.utils.data import Dataset, Subset
from torchvision.datasets import VisionDataset
from torchvision.datasets.vision import StandardTransform

from sitstart.logging import get_logger


logger = get_logger(__name__)


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
