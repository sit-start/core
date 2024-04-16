import pytest
import torch.nn.functional as F
from torch import nn

from ktd.ml.data.smoke_test import SmokeTest
from ktd.ml.training_module import TrainingModule


@pytest.fixture()
def smoketest_training_module_factory(smoketest_model):
    def factory(config):
        return TrainingModule(config, smoketest_model)

    return factory


@pytest.fixture()
def smoketest_model():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 3)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 18, 3)
            self.fc = nn.Linear(18, 4)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = x.flatten(1)
            x = self.fc(x)

            return x

    return Model()


@pytest.fixture()
def smoketest_data_module_factory():
    def factory(config):
        return SmokeTest(
            batch_size=config.get("batch_size", 10),
            num_train=20,
            num_test=10,
            num_classes=4,
            img_shape=(3, 8, 8),
        )

    return factory
