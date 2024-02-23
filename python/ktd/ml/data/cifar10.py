import os
from pathlib import Path

import pytorch_lightning as pl
import torchvision
from filelock import FileLock
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)
from torch.utils.data import DataLoader, random_split, Subset


class CIFAR10(pl.LightningDataModule):
    NUM_CLASSES: int = 10

    def __init__(
        self,
        batch_size=128,
        data_dir: str | os.PathLike[str] | None = None,
        train_split: float = 0.8,
        n_workers: int = 8,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = Path(data_dir) if data_dir else Path.cwd()
        self.train_split = train_split
        self.n_workers = n_workers
        self.prepare_data_per_node = False

    def setup(self, stage: str | None = None):
        normalize = Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        train_transform = Compose(
            [RandomCrop(32, padding=4), RandomHorizontalFlip(), ToTensor(), normalize]
        )
        test_transform = Compose([ToTensor(), normalize])

        with FileLock(self.data_dir / "data.lock"):
            # Load the training dataset twice so that we can specify different
            # transforms for train and val; _should_ be low overhead but DEBUG
            train = torchvision.datasets.CIFAR10(
                root=str(self.data_dir),
                train=True,
                download=True,
                transform=train_transform,
            )
            val = torchvision.datasets.CIFAR10(
                root=str(self.data_dir),
                train=True,
                download=False,
                transform=test_transform,
            )

        n_train = int(len(train) * self.train_split)
        self.train, not_train = random_split(train, [n_train, len(train) - n_train])
        self.val = Subset(val, not_train.indices)

        self.test = torchvision.datasets.CIFAR10(
            root=str(self.data_dir),
            train=False,
            download=True,
            transform=test_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.n_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.n_workers
        )


class SmokeTestCIFAR10(pl.LightningDataModule):
    NUM_CLASSES: int = 10

    def __init__(
        self,
        batch_size: int = 128,
        num_train: int = 1280,
        train_split: float = 0.8,
        num_test: int = 128,
        num_classes: int = 10,
        img_shape: tuple[int, int, int] = (3, 32, 32),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_train = num_train
        self.train_split = train_split
        self.num_test = num_test
        self.prepare_data_per_node = False
        self.num_classes = num_classes
        self.img_shape = img_shape

    def setup(self, stage: str | None = None):
        num_train = int(self.num_train * self.train_split)
        num_val = self.num_train - num_train
        self.train = torchvision.datasets.FakeData(
            num_train,
            self.img_shape,
            num_classes=self.num_classes,
            transform=ToTensor(),
        )
        self.val = torchvision.datasets.FakeData(
            num_val,
            self.img_shape,
            num_classes=self.num_classes,
            transform=ToTensor(),
        )
        self.test = torchvision.datasets.FakeData(
            self.num_test,
            self.img_shape,
            num_classes=self.num_classes,
            transform=ToTensor(),
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
