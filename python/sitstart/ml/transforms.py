from typing import Any, Callable

import torch
from torch.utils.data import default_collate
from torchvision.transforms.functional import adjust_contrast, convert_image_dtype
from torchvision.transforms.v2 import CutMix, MixUp, RandomChoice

_CUTMIX_AND_MIXUP_TRANSFORM_ARGS = {"requires_shuffle": True, "train_only": True}


class AdjustContrast(torch.nn.Module):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return adjust_contrast(img, self.factor)

    def __repr__(self) -> str:
        return f"AdjustContrast(factor={self.factor})"


class ImageToFloat32(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return convert_image_dtype(img, dtype=torch.float32)


class CutMixUp(torch.nn.Module):
    def __init__(self, p: list[float] | None = None, **kwargs: Any):
        super().__init__()
        cutmix = CutMix(**kwargs)
        mixup = MixUp(**kwargs)
        self._cutmix_and_mixup = RandomChoice(transforms=[cutmix, mixup], p=p)

    def forward(self, *inputs: Any) -> Any:
        return self._cutmix_and_mixup(*inputs)


class BatchTransform(torch.nn.Module):
    def __init__(
        self,
        transform: Callable,
        requires_shuffle: bool = False,
        train_only: bool = False,
    ):
        super().__init__()
        self.transform = transform
        self.requires_shuffle = requires_shuffle
        self.train_only = train_only

    def forward(self, batch: Any) -> Any:
        return self.transform(batch)


class IdentityBatchTransform(BatchTransform):
    def __init__(self):
        super().__init__(lambda x: x)


class CutMixBatchTransform(BatchTransform):
    def __init__(self, **kwargs: Any):
        super().__init__(CutMix(**kwargs), **_CUTMIX_AND_MIXUP_TRANSFORM_ARGS)


class MixUpBatchTransform(BatchTransform):
    def __init__(self, **kwargs: Any):
        super().__init__(MixUp(**kwargs), **_CUTMIX_AND_MIXUP_TRANSFORM_ARGS)


class CutMixUpBatchTransform(BatchTransform):
    def __init__(self, **kwargs: Any):
        super().__init__(CutMixUp(**kwargs), **_CUTMIX_AND_MIXUP_TRANSFORM_ARGS)


class DefaultCollateTransform(BatchTransform):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def forward(self, batch: Any) -> Any:
        return self.transform(*default_collate(batch))


class CutMixCollateTransform(DefaultCollateTransform):
    def __init__(self, **kwargs: Any):
        super().__init__(CutMix(**kwargs), **_CUTMIX_AND_MIXUP_TRANSFORM_ARGS)


class CutMixUpCollateTransform(DefaultCollateTransform):
    def __init__(self, **kwargs: Any):
        super().__init__(CutMixUp(**kwargs), **_CUTMIX_AND_MIXUP_TRANSFORM_ARGS)


class MixUpCollateTransform(DefaultCollateTransform):
    def __init__(self, **kwargs: Any):
        super().__init__(MixUp(**kwargs), **_CUTMIX_AND_MIXUP_TRANSFORM_ARGS)
