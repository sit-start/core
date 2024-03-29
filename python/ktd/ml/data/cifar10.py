import os

import pytorch_lightning as pl
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

from ktd.logging import get_logger
from ktd.ml.util import split_dataset

logger = get_logger(__name__)


class CIFAR10(pl.LightningDataModule):
    NUM_CLASSES: int = 10

    def __init__(
        self,
        batch_size=128,
        data_dir: str | os.PathLike[str] | None = None,
        train_split: float = 0.8,
        augment: bool = True,
        n_workers: int = 8,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = str(data_dir) if data_dir else os.getcwd()
        self.train_split = train_split
        self.n_workers = n_workers
        self.prepare_data_per_node = True
        self.augment = augment

    def prepare_data(self) -> None:
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None):
        normalization = Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        augmentation = [RandomCrop(32, padding=4), RandomHorizontalFlip()]
        seed = 42

        val_and_test_transform = Compose([ToTensor(), normalization])
        train_transform = Compose(
            (augmentation if self.augment else []) + [ToTensor(), normalization]
        )

        train_dataset = datasets.CIFAR10(
            self.data_dir, train=True, transform=train_transform
        )
        self.train_dataset = split_dataset(
            train_dataset, self.train_split, train=True, seed=seed
        )

        val_dataset = datasets.CIFAR10(
            self.data_dir, train=True, transform=val_and_test_transform
        )
        self.val_dataset = split_dataset(
            val_dataset, self.train_split, train=False, seed=seed
        )

        self.test_dataset = datasets.CIFAR10(
            self.data_dir, train=False, transform=val_and_test_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.n_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.n_workers
        )
