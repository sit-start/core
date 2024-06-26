import os
from typing import Any, Callable, cast

import pytorch_lightning as pl
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision.transforms import Compose

from sitstart.logging import get_logger
from sitstart.ml.data import DEFAULT_DATASET_ROOT
from sitstart.ml.util import split_dataset
from sitstart.util.decorators import memoize
from sitstart.util.general import hasarg
from sitstart.util.torch import generator_from_seed, randint

logger = get_logger(__name__)


class VisionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_class: type,
        batch_size=128,
        data_dir: str | os.PathLike[str] | None = DEFAULT_DATASET_ROOT,
        train_dataset_size: float | int | None = None,
        train_split_size: float | int = 0.8,
        augment: Callable | None = None,
        collate: Callable | None = None,
        transform: Callable | None = None,
        n_workers: int = 8,
        shuffle: bool = True,
        sampler: Sampler | None = None,
        seed: int = 42,
        test_as_val: bool = False,
    ):
        """Data module for sub-classes of VisionDataset.

        Args:
            dataset_class: VisionDataset sub-class.
            batch_size: Batch size.
            data_dir: Root data directory where the dataset will be stored.
            train_dataset_size: Fraction or number of samples from the
                training dataset to use. Defaults to the full training
                dataset.
            train_split_size: Fraction or number of samples from the training
                dataset to use for training. If `test_as_val` is False,
                the remaining samples are used for validation.
            augment: Augmentation transform applied to the training split.
            collate: Collate function for the training split dataloader.
            transform: Base transform applied to the dataset.
            n_workers: Number of workers for data loaders.
            seed: Random seed for splitting and shuffling. Take care when
                using a non-default value to avoid data leakage by, e.g.,
                resuming from checkpoint with a different seed.
            shuffle: Whether to shuffle the data in the training split.
            sampler: Sampler for the training split.
            test_as_val: Whether to use the test set as the validation set.
        """
        super().__init__()
        if collate and collate.requires_shuffle and not shuffle:
            logger.warning("Collate function requires shuffling; enabling shuffle.")
            shuffle = True
        if test_as_val:
            logger.info("Using test set as validation set and ignoring `train_split`.")
            train_split_size = 1.0

        self.dataset_class = dataset_class
        self.batch_size = batch_size
        self.data_dir = str(data_dir) if data_dir else os.getcwd()
        self.train_dataset_size = train_dataset_size
        self._train_split_size = train_split_size
        self.transform = transform
        self.n_workers = n_workers
        self.generator = generator_from_seed(seed)
        self._split_seed = randint(generator=self.generator, dtype=torch.int32).item()
        self._shuffle = shuffle
        self._test_as_val = test_as_val
        self._sampler = sampler
        self.train_transform = Compose((augment, transform)) if augment else transform
        self.train_collate_fn = collate
        self.prepare_data_per_node = True

    @property
    @memoize
    def train_dataset(self) -> datasets.VisionDataset:
        return self._load_dataset("train")

    @property
    def test_dataset(self) -> datasets.VisionDataset | None:
        return self._test_dataset if not self._test_as_val else None

    @property
    def train_split(self) -> Subset:
        return self._train_val_split[0]

    @property
    def val_split(self) -> Subset:
        return self._train_val_split[1]

    @property
    def test_as_val(self) -> bool:
        return self._test_as_val

    @property
    def train_split_size(self) -> float | int:
        return self._train_split_size

    @memoize
    def get_sampler(self) -> Sampler | None:
        return self._sampler

    @property
    def has_sampler(self) -> bool:
        return self._sampler is not None

    def prepare_data(self) -> None:
        if (not hasarg(self.dataset_class, "root", str)) or (
            not hasarg(self.dataset_class, "download", bool)
        ):
            raise ValueError(
                f"Initializer for dataset class {self.dataset_class} does not accept "
                "`root: str` and `download: bool` arguments. Subclass and override "
                "`prepare_data()`."
            )

        self.dataset_class(root=self.data_dir, download=True)

    def setup(self, stage: str | None = None):
        _ = self.train_split, self.val_split, self.test_dataset

    def train_dataloader(self) -> DataLoader:
        logger.info("Creating training data loader.")
        shuffle = self._shuffle
        if self.has_sampler and shuffle:
            logger.info("Using custom sampler and ignoring `shuffle`.")
            shuffle = False

        return DataLoader(
            self.train_split,
            batch_size=self.batch_size,
            collate_fn=self.train_collate_fn,
            num_workers=self.n_workers,
            shuffle=shuffle,
            sampler=self.get_sampler(),
            generator=self.generator,
        )

    def val_dataloader(self) -> DataLoader:
        logger.info("Creating validation data loader.")
        return DataLoader(
            self.val_split, batch_size=self.batch_size, num_workers=self.n_workers
        )

    def test_dataloader(self) -> DataLoader | None:
        logger.info("Creating test data loader.")
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.n_workers
        )

    @property
    def criteria_weight(self) -> torch.Tensor | torch.nn.Module | None:
        """Weight for the loss function, if applicable."""
        return None

    @property
    @memoize
    def _test_dataset(self) -> datasets.VisionDataset | None:
        try:
            return self._load_dataset("test")
        except RuntimeError:
            logger.warning("Test set not found.")
            return None

    @property
    @memoize
    def _train_val_split(self) -> tuple[Subset, Subset]:
        train_split, val_split = self._split_train_val(self.train_dataset)

        if self._test_as_val:
            assert self.train_split_size == 1.0
            if self._test_dataset is None:
                raise ValueError(
                    "Test set is being used for validation but is unavailable."
                )
            assert self._test_dataset is not None
            val_split = Subset(self._test_dataset, range(len(self._test_dataset)))
        logger.info(f"Training with {len(train_split)} samples.")
        suffix = " taken from the test set." if self._test_as_val else "."
        logger.info(f"Validating with {len(val_split)} samples" + suffix)

        return train_split, val_split

    def _split_train_val(
        self, dataset: datasets.VisionDataset, **kwargs
    ) -> tuple[Subset, Subset]:
        return split_dataset(
            dataset,
            dataset_size=self.train_dataset_size,
            train_split_size=self._train_split_size,
            seed=cast(int, self._split_seed),
            train_transform=self.train_transform,
            val_transform=self.transform,
            **kwargs,
        )

    def _load_dataset(self, split: str) -> datasets.VisionDataset:
        train = split == "train"

        kwargs: dict[str, Any] = {
            "root": self.data_dir,
            "transform": self.train_transform if train else self.transform,
        }
        if hasarg(self.dataset_class.__init__, "train", bool):
            kwargs["train"] = train
        elif hasarg(self.dataset_class.__init__, "split", str):
            kwargs["split"] = split

        return self.dataset_class(**kwargs)
