import torch
from torch.nn import functional as F


def _reduce(input: torch.Tensor, reduction: str) -> torch.Tensor:
    """Reduce the given input. Reduction method is as in torch loss functions."""
    if reduction == "none":
        return input
    if reduction == "mean":
        return input.mean()
    if reduction == "sum":
        return input.sum()
    raise ValueError("Reduction must be one of 'none', 'mean', or 'sum'.")


class FocalLoss(torch.nn.Module):
    """Multi-class focal loss from https://arxiv.org/abs/1708.02002"""

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(input.device) if self.weight is not None else None
        ce_loss = F.cross_entropy(input, target, weight=weight, reduction="none")
        pt = torch.exp(-F.cross_entropy(input, target, reduction="none"))
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        return _reduce(focal_loss, self.reduction)
