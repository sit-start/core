import torch
from torch.utils.data import TensorDataset

from ktd.ml.util import hash_tensor, split_dataset


def test_hash_tensor():
    x = torch.tensor([1, 2, 3])
    assert hash_tensor(x) == hash((1, 2, 3))


def test_split_dataset():
    m, d, split, seed = 10, 3, 0.8, 42
    torch.manual_seed(seed)

    X, Y = torch.randn(m, d), torch.randn(m, 1)
    dataset = TensorDataset(X, Y)
    train = split_dataset(dataset, split, train=True, seed=seed)
    test = split_dataset(dataset, split, train=False, seed=seed)

    assert all(x in X and y in Y for x, y in train)
    assert all(x in X and y in Y for x, y in test)
    assert not any(x in train and x in test for x in X)
    assert not any(x in train and x in test for x in X)
