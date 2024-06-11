from torchvision.datasets import FakeData, VisionDataset
from torchvision.transforms import ToTensor

from sitstart.ml.data.modules.vision_data_module import VisionDataModule

from typing import Callable


class Fake2d(VisionDataModule):
    def __init__(
        self,
        batch_size: int = 10,
        train_split_size: float = 0.8,
        augment: Callable | None = None,
        transform: Callable | None = None,
        num_train: int = 40,
        num_test: int = 10,
        num_classes: int = 10,
        img_shape: tuple[int, int, int] = (3, 32, 32),
    ):
        super().__init__(
            FakeData,
            batch_size=batch_size,
            train_split_size=train_split_size,
            augment=augment,
            transform=transform or ToTensor(),
            n_workers=0,
            shuffle=False,
        )
        self.num_train = num_train
        self.num_test = num_test
        self.num_classes = num_classes
        self.img_shape = img_shape

    def prepare_data(self) -> None:
        pass

    def _load_dataset(self, split: str) -> VisionDataset:
        train = split == "train"
        return FakeData(
            size=self.num_train if train else self.num_test,
            image_size=self.img_shape,
            num_classes=self.num_classes,
            transform=self.train_transform if train else self.transform,
            random_offset=0,
        )
