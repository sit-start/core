from typing import Any
from pathlib import Path

from .s3 import ImageFolderS3
from torchvision.datasets.imagenet import load_meta_file, verify_str_arg, META_FILE


class ImageNetS3(ImageFolderS3):
    """An S3-backed ImageNet dataset; see torchvision.datasets.ImageNet

    Assumes the dataset has been extracted and parsed by
    torchvision.datasets.ImageNet and uploaded to S3.
    """

    def __init__(
        self,
        s3_root: str,
        local_root: str | None = None,
        split: str = "train",
        **kwargs: Any,
    ) -> None:
        self.root = s3_root
        self.split = verify_str_arg(split, "split", ("train", "val"))

        super().__init__(s3_root=self.split_folder, local_root=local_root, **kwargs)
        self.root = s3_root

        # downloads the meta file if needed
        local_meta_path = Path((self.client.CloudPath(self.root) / META_FILE).fspath)

        wnid_to_classes = load_meta_file(str(local_meta_path.parent), META_FILE)[0]
        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {
            cls: idx for idx, clss in enumerate(self.classes) for cls in clss
        }

    @property
    def split_folder(self) -> str:
        return f"{self.root}/{self.split}"
