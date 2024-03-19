import torch
from torch.utils.data import Dataset, random_split


def hash_tensor(x: torch.Tensor) -> int:
    return hash(tuple(x.cpu().reshape(-1).tolist()))


def split_dataset(
    dataset: Dataset, train_split: float, train: bool = True, seed: int | None = 42
) -> Dataset:
    assert hasattr(dataset, "__len__"), "Dataset must implement __len__"

    len_dataset = len(dataset)  # type: ignore[arg-type]
    n_train = int(len_dataset * train_split)
    lengths = [n_train, len_dataset - n_train]
    generator = torch.Generator().manual_seed(seed) if seed else None

    train_dataset, val_dataset = random_split(dataset, lengths, generator=generator)

    return train_dataset if train else val_dataset
