from typing import cast

import torch
from torch.utils.data import Sampler, Subset, WeightedRandomSampler
from torchvision.datasets.vision import VisionDataset

from sitstart.logging import get_logger
from sitstart.ml.data.datasets.ham10k import HAM10k as HAM10kDataset
from sitstart.ml.data.modules.vision_data_module import VisionDataModule
from sitstart.ml.util import gamma_correct
from sitstart.util.decorators import memoize

logger = get_logger(__name__)


class HAM10k(VisionDataModule):
    def __init__(
        self,
        *args,
        criteria_gamma: float = 0.0,
        dedupe: bool = False,
        rebalance_gamma: float = 0.0,
        **kwargs,
    ):
        super().__init__(HAM10kDataset, *args, **kwargs)
        self._rebalance_gamma = rebalance_gamma
        self._criteria_gamma = criteria_gamma
        self._dedupe = dedupe

    @memoize
    def get_sampler(self) -> Sampler | None:
        if self._rebalance_gamma <= 0.0:
            return super().get_sampler()
        if super().has_sampler:
            raise ValueError("Cannot specify `sampler` when `rebalance_gamma > 0`")

        logger.info(f"Rebalancing with gamma = {self._rebalance_gamma}.")
        train_dataset = cast(HAM10kDataset, self.train_dataset)
        split_indices = torch.tensor(self.train_split.indices)
        train_split_targets = torch.tensor(train_dataset.targets)[split_indices]
        class_weights = self._get_class_weights(self._rebalance_gamma)
        weights = class_weights.gather(0, train_split_targets).tolist()

        return WeightedRandomSampler(weights, len(weights), generator=self.generator)

    @property
    def has_sampler(self) -> bool:
        return self._rebalance_gamma > 0.0 or super().has_sampler

    @VisionDataModule.criteria_weight.getter
    def criteria_weight(self) -> torch.Tensor | torch.nn.Module | None:
        if self._criteria_gamma <= 0.0:
            return None
        return self._get_class_weights(self._criteria_gamma)

    @property
    def class_count(self) -> torch.Tensor:
        # after setup(), can be computed directly as:
        # ```
        #   train_dataset = cast(HAM10kDataset, self.train_dataset)
        #   train_targets = train_dataset.targets
        #   if self._dedupe:
        #       train_targets = dedupe(train_dataset.targets, train_dataset.lesion_ids)
        #   _, class_count = torch.tensor(train_targets).unique(return_counts=True)
        # ```
        if self._dedupe:
            return torch.tensor([727, 5403, 614, 327, 228, 98, 73])
        return torch.tensor([1099, 6705, 1113, 514, 327, 142, 115])

    def _get_class_weights(self, gamma: float) -> torch.Tensor:
        inv_freq = self.class_count.sum() / self.class_count
        inv_freq[self.class_count == 0] = 0.0
        return gamma_correct(inv_freq, gamma=gamma, norm="count")

    def _split_train_val(self, dataset: VisionDataset) -> tuple[Subset, Subset]:
        # There are multiple images of the same lesion in the dataset,
        # so we remove duplicates when `dedupe=True` and  split by lesion ID
        # when `dedupe=False` to avoid data leakage .
        logger.info("Splitting training and validation by lesion ID.")
        return super()._split_train_val(
            dataset, ids=cast(HAM10kDataset, dataset).lesion_ids, dedupe=self._dedupe
        )
