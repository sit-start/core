import torch
from torchvision.datasets import VisionDataset

from sitstart.ml.util import split_dataset, update_module


def test_update_submodule():
    torch.manual_seed(42)

    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_layers = torch.nn.ModuleList(
                [torch.nn.Conv2d(3, 3, 3), torch.nn.Linear(3, 3)]
            )
            self.fc1 = torch.nn.Linear(3, 3)
            self.fc2 = torch.nn.Linear(3, 1)

        def forward(self, x):
            x = self.hidden_layers[0](x).flatten()
            x = self.hidden_layers[1](x)
            x = self.fc1(x)
            return self.fc2(x)

    model = TestModel()
    update_module(model, require_grad=["", "-fc2"])

    assert all(p.requires_grad for p in model.hidden_layers[0].parameters())
    assert all(p.requires_grad for p in model.hidden_layers[1].parameters())
    assert all(p.requires_grad for p in model.fc1.parameters())
    assert all(not p.requires_grad for p in model.fc2.parameters())

    model = TestModel()
    for layer in model.hidden_layers:
        with torch.no_grad():
            for param in layer.parameters():
                param *= 0.0
    update_module(model, init={"hidden_layers.0": None})

    for param in model.hidden_layers[0].parameters():
        assert param.ne(0.0).any()
    for param in model.hidden_layers[1].parameters():
        assert param.eq(0.0).all()

    model = TestModel()
    update_module(
        model,
        replace={
            "hidden_layers.0": torch.nn.Linear(3, 5),
            "fc*": torch.nn.Linear(5, 5),
        },
    )
    assert model.hidden_layers[0].out_features == 5
    assert model.fc1 is not model.fc2
    assert model.fc1.in_features == model.fc1.out_features == 5
    assert model.fc2.in_features == model.fc2.out_features == 5

    model = TestModel()
    update_module(model, replace={"fc*": torch.nn.Linear(3, 5)})


def test_split_dataset():
    n = 30
    train_split_size = 0.7
    seed = 42

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
    train, val = split_dataset(dataset, train_split_size, seed=seed)
    X_train, Y_train = zip(*train)
    X_val, Y_val = zip(*val)

    assert sorted(X_train + X_val) == sorted(X.tolist())
    assert sorted(Y_train + Y_val) == sorted(Y.tolist())

    train, val = split_dataset(dataset, train_split_size=12, seed=seed)

    assert len(train) == 12
    assert len(val) == 18

    train, val = split_dataset(dataset, train_split_size=5, dataset_size=10, seed=seed)

    assert len(train) == 5
    assert len(val) == 5

    train, val = split_dataset(dataset, train_split_size, ids=ids, seed=seed)
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
        seed=seed,
        train_transform=train_transform,
        val_transform=val_transform,
    )

    assert isinstance(train.dataset, VisionDataset)
    assert train.dataset.transform == train_transform

    assert isinstance(val.dataset, VisionDataset)
    assert val.dataset.transform == val_transform
