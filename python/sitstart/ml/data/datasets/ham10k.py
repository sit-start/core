import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

import pandas as pd
import yaml
from checksumdir import dirhash
from cloudpathlib.s3 import S3Client, S3Path
from omegaconf import DictConfig
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, extract_archive

from sitstart.aws.util import get_aws_session
from sitstart.logging import get_logger
from sitstart.ml.data import DEFAULT_DATASET_ROOT

_MISSING_OR_CORRUPTED_IMAGE_IDS = ["ISIC_0035068"]
_DOWNLOAD_URL = "s3://ktd-datasets/ham10k"
_EXCLUDED_FILES = [".DS_Store"]
_SOURCE_FILES = """
train:
    images:
        md5: 61e9d589c7c6f4a3b35b7dcb54548804
        archives:
            - filename: HAM10000_images_part_1.zip
              subdir: .
              md5: 4639bfa73ab251610530a97c898e6e46
            - filename: HAM10000_images_part_2.zip
              subdir: .
              md5: da43d6cc50f6613013be07e8986b384b
    metadata:
        filename: HAM10000_metadata.csv
        md5: 8f85fb1aa29d80a2797247e434deb79d
test:
    images:
        md5: b9860a7e281314c5cbffe7022672fa9b
        archives:
            - filename: ISIC2018_Task3_Test_Images.zip
              subdir: ISIC2018_Task3_Test_Images
              md5: 0488b6f65849e310857d9207daa55a21
    metadata:
        filename: ISIC2018_Task3_Test_GroundTruth.csv
        md5: 2b5663aacc9644740f27e355cf7f369a
"""

logger = get_logger(__name__)


class HAM10k(VisionDataset):
    """HAM10000 Dataset

    Dermatoscopic images of common pigmented skin lesions from:

    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
    """

    def __init__(
        self,
        root: str = DEFAULT_DATASET_ROOT,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
        aws_profile: str | None = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.split = "train" if train else "test"
        self.source = DictConfig(yaml.safe_load(_SOURCE_FILES))
        self._aws_profile = aws_profile

        if download:
            self.download()

        if not self._check_integrity():
            msg = "Dataset not found or corrupted. Use download=True to download it."
            raise RuntimeError(msg)

        self._load_metadata()

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        image_id, target = self.image_ids[index], self.targets[index]
        image = self._load_image(image_id)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_image(self, image_id: str) -> Any:
        image_path = f"{self.dataset_root}/{self.split}/{image_id}.jpg"
        return Image.open(image_path)

    def _load_metadata(self) -> None:
        data = pd.read_csv(f"{self.dataset_root}/{self.split}.csv")

        for image_id in _MISSING_OR_CORRUPTED_IMAGE_IDS:
            index = data.index[data.image_id == image_id].tolist()
            data.drop(axis=0, index=index, inplace=True)

        class_to_name = [
            ("bkl", "benign keratosis"),
            ("nv", "melanocytic nevus"),
            ("mel", "melanoma"),
            ("bcc", "basal cell carcinoma"),
            ("akiec", "actinic keratosis"),
            ("vasc", "vascular lesion"),
            ("df", "dermatofibroma"),
        ]
        self.classes = [c for c, _ in class_to_name]
        self.class_names = [n for _, n in class_to_name]
        class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        # note that lesion IDs can only be compared within a split
        self.lesion_ids = data["lesion_id"].tolist()

        self.image_ids, self.targets = [], []
        for _, x in data.iterrows():
            self.image_ids.append(x.image_id)
            self.targets.append(class_to_idx[x.dx])

    def _check_integrity(self) -> bool:
        for name, split in self.source.items():
            images_path = f"{self.dataset_root}/{name}"
            images_md5_path = Path(f"{self.dataset_root}/{name}.md5")
            metadata_path = f"{self.dataset_root}/{name}.csv"

            if not Path(images_path).exists():
                return False
            if not Path(images_md5_path).exists():
                logger.info(f"Computing md5 hash for {name} images.")
                images_md5 = dirhash(images_path, "md5", excluded_files=_EXCLUDED_FILES)
                Path(images_md5_path).write_text(images_md5)
            else:
                images_md5 = Path(images_md5_path).read_text()
            if images_md5 != split.images.md5:
                return False
            if not check_integrity(metadata_path, split.metadata.md5):
                return False
        return True

    @property
    def dataset_root(self) -> str:
        return f"{self.root}/ham10k"

    def download(self) -> None:
        if self._check_integrity():
            logger.info("Files downloaded and verified.")
            return

        logger.info(f"Downloading and extracting dataset to {self.dataset_root}.")

        os.makedirs(self.dataset_root, exist_ok=True)
        downloads_path = f"{self.dataset_root}/downloads"

        session = get_aws_session(profile=self._aws_profile)
        s3_client = S3Client(local_cache_dir=downloads_path, boto3_session=session)

        for name, split in self.source.items():
            logger.info(f"Downloading and extracting {name} data.")
            metadata_path = f"{self.dataset_root}/{name}.csv"
            image_path = f"{self.dataset_root}/{name}"
            shutil.rmtree(image_path, ignore_errors=True)
            os.makedirs(image_path)

            for archive in split.images.archives:
                logger.info(f"Downloading {archive.filename}.")
                remote_archive_path = f"{_DOWNLOAD_URL}/{archive.filename}"
                archive_path = S3Path(remote_archive_path, client=s3_client).fspath
                if not check_integrity(archive_path, archive.md5):
                    raise RuntimeError(f"Downloaded file {archive_path} is corrupted.")

                logger.info(f"Extracting {archive.filename}.")
                with TemporaryDirectory(prefix=f"{downloads_path}/") as extract_path:
                    extract_archive(archive_path, extract_path)
                    extracted_image_path = f"{extract_path}/{archive.subdir}"
                    for file in os.listdir(extracted_image_path):
                        shutil.move(f"{extracted_image_path}/{file}", image_path)

            logger.info(f"Downloading metadata {split.metadata.filename}.")
            remote_metadata_path = f"{_DOWNLOAD_URL}/{split.metadata.filename}"
            metadata_path = S3Path(remote_metadata_path, client=s3_client).fspath
            new_metadata_path = f"{self.dataset_root}/{name}.csv"
            shutil.copyfile(metadata_path, new_metadata_path)

        logger.info(f"Downloaded and extracted dataset to {self.dataset_root}.")
