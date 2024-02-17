from abc import ABC, abstractmethod


class DataModule(ABC):

    @abstractmethod
    def train_dataloader(self):
        pass

    @abstractmethod
    def val_dataloader(self):
        pass

    @abstractmethod
    def test_dataloader(self):
        pass

    @abstractmethod
    def prepare_data(self):
        pass
