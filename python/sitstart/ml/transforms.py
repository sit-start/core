import torch
from torchvision.transforms.functional import adjust_contrast, convert_image_dtype


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
