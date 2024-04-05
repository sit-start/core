import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


class SmokeTest(pl.LightningDataModule):
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
            random_offset=0,
        )
        self.val = torchvision.datasets.FakeData(
            num_val,
            self.img_shape,
            num_classes=self.num_classes,
            transform=ToTensor(),
            random_offset=1,
        )
        self.test = torchvision.datasets.FakeData(
            self.num_test,
            self.img_shape,
            num_classes=self.num_classes,
            transform=ToTensor(),
            random_offset=2,
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
