import torch
from torcheval.metrics import MulticlassRecall


class AverageMulticlassRecall(MulticlassRecall):
    """Unweighted average multiclass recall.

    Wraps torcheval.metrics.MulticlassRecall which, as of v0.0.7, errors
    when average='macro', num_classes is specified, and the metric is
    updated without a true or false positive in every class.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes=num_classes, average=None)

    def compute(self: "AverageMulticlassRecall") -> torch.Tensor:
        return super().compute().mean()
