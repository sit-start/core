from typing import Any

import torch
from torcheval.metrics import MulticlassRecall
from torchmetrics.classification import (
    MulticlassConfusionMatrix as _MulticlassConfusionMatrix,
)


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


class MulticlassConfusionMatrix(_MulticlassConfusionMatrix):
    def __init__(
        self, num_classes: int, labels: list[str] | None = None, **kwargs: Any
    ) -> None:
        super().__init__(num_classes=num_classes, **kwargs)
        if labels and len(labels) != num_classes:
            raise ValueError("Number of labels must match number of classes")
        self.labels = labels

    def plot(self, labels: list[str] | None = None, **kwargs: Any) -> Any:
        return super().plot(labels=labels or self.labels, **kwargs)
