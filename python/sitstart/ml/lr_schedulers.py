from functools import partial
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, LRScheduler
from torch.optim.lr_scheduler import (
    SequentialLR as _SequentialLR,
)


class SequentialLR(_SequentialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        schedulers: list[partial[LRScheduler]],
        milestones: list[int],
        last_epoch: int = -1,
    ):
        """`SequentialLR` that accepts a  list of `partial` schedulers."""
        super().__init__(
            optimizer,
            [scheduler(optimizer=optimizer) for scheduler in schedulers],
            milestones=milestones,
            last_epoch=last_epoch,
        )


class LinearWarmup(_SequentialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        scheduler: partial[LRScheduler],
        last_epoch: int = -1,
        warmup_iters: int = 5,
        warmup_factor: float = 1.0 / 3,
    ):
        super().__init__(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer, start_factor=warmup_factor, total_iters=warmup_iters
                ),
                scheduler(optimizer=optimizer),
            ],
            last_epoch=last_epoch,
            milestones=[warmup_iters],
        )


class LinearWarmupCosineAnnealingWarmRestarts(_SequentialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        *args: Any,
        last_epoch: int = -1,
        warmup_iters: int = 5,
        warmup_factor: float = 1.0 / 3,
        **kwargs: Any,
    ):
        super().__init__(
            optimizer,
            last_epoch=last_epoch,
            schedulers=[
                LinearLR(
                    optimizer, start_factor=warmup_factor, total_iters=warmup_iters
                ),
                CosineAnnealingWarmRestarts(optimizer, T_0=T_0, *args, **kwargs),
            ],
            milestones=[warmup_iters],
        )
