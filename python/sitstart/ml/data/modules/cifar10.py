from sitstart.ml.data.modules.vision_data_module import VisionDataModule
from torchvision.datasets import CIFAR10 as CIFAR10Dataset


class CIFAR10(VisionDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(CIFAR10Dataset, *args, **kwargs)
