from typing import cast

from torch.utils.data import Subset
from torchvision.datasets.vision import VisionDataset

from sitstart.logging import get_logger
from sitstart.ml.data.datasets.ham10k import HAM10k as HAM10kDataset
from sitstart.ml.data.modules.vision_data_module import VisionDataModule

logger = get_logger(__name__)


class HAM10k(VisionDataModule):
    def __init__(self, *args, rebalance: bool = False, **kwargs):
        if rebalance:
            if "sampler" in kwargs:
                raise ValueError(
                    "Cannot specify both `sampler` and `class_balance_sampler`"
                )
            if kwargs.setdefault("shuffle", False):
                logger.info("Ignoring `shuffle` and using rebalancing sampler.")
                kwargs["shuffle"] = False
        self._rebalance = rebalance
        super().__init__(HAM10kDataset, *args, **kwargs)

    def _split_train_val(
        self, dataset: VisionDataset, **kwargs
    ) -> tuple[Subset, Subset]:
        # avoid data leakage, where different images of the same lesion appear
        # in train and val
        return super()._split_train_val(
            dataset, ids=cast(HAM10kDataset, dataset).lesion_ids
        )
