import torch


def hash_tensor(x: torch.Tensor) -> int:
    return hash(tuple(x.cpu().reshape(-1).tolist()))
