import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


class Fake2d(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 10,
        num_train: int = 30,
        num_val: int = 10,
        num_test: int = 10,
        num_classes: int = 10,
        img_shape: tuple[int, int, int] = (3, 32, 32),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.prepare_data_per_node = False
        self.num_classes = num_classes
        self.img_shape = img_shape

    def setup(self, stage: str | None = None):
        self.train = torchvision.datasets.FakeData(
            self.num_train,
            self.img_shape,
            num_classes=self.num_classes,
            transform=ToTensor(),
            random_offset=0,
        )
        self.val = torchvision.datasets.FakeData(
            self.num_val,
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
