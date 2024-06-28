from typing import cast

from torch.utils.data import Sampler, Subset
from torchvision.datasets.vision import VisionDataset

from sitstart.logging import get_logger
from sitstart.ml.data.datasets.ham10k import HAM10k as HAM10kDataset
from sitstart.ml.data.modules.vision_data_module import VisionDataModule
from sitstart.ml.util import rebalancing_sampler
from sitstart.util.decorators import memoize

logger = get_logger(__name__)


class HAM10k(VisionDataModule):
    def __init__(self, *args, rebalance: bool = False, dedupe: bool = False, **kwargs):
        if rebalance:
            if "sampler" in kwargs:
                raise ValueError("Cannot specify `sampler` when `rebalance=True`")
            if kwargs.setdefault("shuffle", False):
                logger.info("Ignoring `shuffle` and using rebalancing sampler.")
                kwargs["shuffle"] = False
        self._rebalance = rebalance
        self._dedupe = dedupe
        super().__init__(HAM10kDataset, *args, **kwargs)

    @VisionDataModule.sampler.getter
    @memoize
    def sampler(self) -> Sampler | None:
        if self._rebalance:
            logger.info("Creating rebalancing sampler.")
            train_dataset = cast(HAM10kDataset, self.train_dataset)
            train_targets = [train_dataset.targets[i] for i in self.train_split.indices]
            return rebalancing_sampler(train_targets, self.generator)
        return None

    def _split_train_val(self, dataset: VisionDataset) -> tuple[Subset, Subset]:
        # There are multiple images of the same lesion in the dataset,
        # so we remove duplicates when `dedupe=True` and  split by lesion ID
        # when `dedupe=False` to avoid data leakage .
        logger.info("Splitting training and validation by lesion ID.")
        return super()._split_train_val(
            dataset, ids=cast(HAM10kDataset, dataset).lesion_ids, dedupe=self._dedupe
        )
