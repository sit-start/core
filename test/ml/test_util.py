import torch
from torchvision.datasets import VisionDataset

from sitstart.ml.util import hash_tensor, split_dataset


def test_split_dataset():
    n = 30
    train_split_size = 0.7
    generator = torch.Generator().manual_seed(42)

    def to_ids(X):
        return (torch.Tensor(X) % int(n * 0.7)).tolist()

    X = torch.arange(n, dtype=torch.int32)
    Y = 2 * X
    ids = to_ids(X)

    class TestDataset(VisionDataset):
        def __init__(self, X, Y):
            super().__init__(root="")
            self.data = X
            self.targets = Y

        def __getitem__(self, index):
            return self.data[index], self.targets[index]

        def __len__(self):
            return len(self.data)

    dataset = TestDataset(X, Y)
    train, val = split_dataset(dataset, train_split_size, generator=generator)
    X_train, Y_train = zip(*train)
    X_val, Y_val = zip(*val)

    assert sorted(X_train + X_val) == sorted(X.tolist())
    assert sorted(Y_train + Y_val) == sorted(Y.tolist())

    train, val = split_dataset(dataset, train_split_size=12, generator=generator)

    assert len(train) == 12
    assert len(val) == 18

    train, val = split_dataset(
        dataset, train_split_size=5, dataset_size=10, generator=generator
    )

    assert len(train) == 5
    assert len(val) == 5

    train, val = split_dataset(dataset, train_split_size, ids=ids, generator=generator)
    X_train, Y_train = zip(*train)
    X_val, Y_val = zip(*val)

    assert sorted(X_train + X_val) == sorted(X.tolist())
    assert sorted(Y_train + Y_val) == sorted(Y.tolist())
    assert set(to_ids(X_train)).isdisjoint(to_ids(X_val))

    train_transform = lambda x: 2 * x  # noqa
    val_transform = lambda x: 3 * x  # noqa
    train, val = split_dataset(
        dataset,
        train_split_size,
        ids=ids,
        generator=generator,
        train_transform=train_transform,
        val_transform=val_transform,
    )

    assert isinstance(train.dataset, VisionDataset)
    assert train.dataset.transform == train_transform

    assert isinstance(val.dataset, VisionDataset)
    assert val.dataset.transform == val_transform


def test_hash_tensor():
    x = torch.tensor([1, 2, 3])
    assert hash_tensor(x) == hash((1, 2, 3))
