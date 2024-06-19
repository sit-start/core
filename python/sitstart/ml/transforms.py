import torch
from torchvision.transforms.functional import adjust_contrast


class AdjustContrast(torch.nn.Module):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return adjust_contrast(img, self.factor)

    def __repr__(self) -> str:
        return f"AdjustContrast(factor={self.factor})"
