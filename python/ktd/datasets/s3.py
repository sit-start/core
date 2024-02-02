from pathlib import PosixPath
from typing import Any, Callable

import boto3
from cloudpathlib import S3Path, S3Client
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import (
    IMG_EXTENSIONS,
    default_loader,
    has_file_allowed_extension,
)
from ktd.utilities.decorators import timer
from ktd.logging import get_logger

logger = get_logger(__name__)


class DatasetFolderS3(VisionDataset):
    """
    An S3-backed dataset; see torchvision.datasets.DatasetFolder
    
    Parameters
    ----------
    ...
        See torchvision.datasets.DatasetFolder.__init__
    force_use_cached : bool
        If a sample has already been downloaded, use the cached version without
        checking for modifications on S3.
    """

    def __init__(
        self,
        s3_root: str,
        loader: Callable[[str], Any],
        local_root: str | None = None,
        extensions: tuple[str, ...] | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        is_valid_file: Callable[[str], bool] | None = None,
        force_use_cached: bool = False,
    ) -> None:
        super().__init__(
            s3_root, transform=transform, target_transform=target_transform
        )
        self.client = S3Client(local_cache_dir=local_root)

        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self._remote_to_local_path = {} if force_use_cached else None

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: dict[str, int],
        extensions: tuple[str, ...] | None = None,
        is_valid_file: Callable[[str], bool] | None = None,
    ) -> list[tuple[str, int]]:
        if class_to_idx is None:
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(
            directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file
        )

    def find_classes(self, url: str) -> tuple[list[str], dict[str, int]]:
        return find_classes(url)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        path, target = self.samples[index]

        # accessing `CloudPath.fspath` downloads the file if there is no local
        # copy or if the local copy is outdated
        if self._remote_to_local_path is not None:
            if path in self._remote_to_local_path:
                local_path = self._remote_to_local_path[path]
            else:
                local_path = self.client.CloudPath(path).fspath
                self._remote_to_local_path[path] = local_path
        else:
            local_path = self.client.CloudPath(path).fspath

        sample = self.loader(local_path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


def find_classes(s3_path: str) -> tuple[list[str], dict[str, int]]:
    # modified from torchvision.datasts.folder.find_classes
    bucket, prefix = s3_path.split("://")[1].split("/", 1)
    prefix = prefix.rstrip("/") + "/"
    s3 = boto3.client("s3")

    classes = []
    for result in s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix, Delimiter="/"
    ):
        classes += [
            PosixPath(entry["Prefix"]).name for entry in result["CommonPrefixes"]
        ]
    classes.sort()

    if not classes:
        raise FileNotFoundError(
            f"Couldn't find any class folder in s3://{bucket}/{prefix}."
        )
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


@timer
def make_dataset(
    s3_path: str,
    class_to_idx: dict[str, int] | None = None,
    extensions: str | tuple[str, ...] | None = None,
    is_valid_file: Callable[[str], bool] | None = None,
) -> list[tuple[str, int]]:
    # adapted from torchvision.datasts.folder.make_dataset
    if class_to_idx is None:
        _, class_to_idx = find_classes(s3_path)
    elif not class_to_idx:
        raise ValueError(
            "'class_to_index' must have at least one entry to collect any samples."
        )

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time"
        )

    if extensions is not None:

        def _is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)

        is_valid_file = _is_valid_file

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_url = S3Path(s3_path) / target_class
        # skip checking `target_url.is_dir()` for expediency, and since the
        # most common case uses `find_classes()`, which only returns valid
        # directories
        for root, _, fnames in sorted(target_url.walk(follow_symlinks=True)):
            for fname in sorted(fnames):
                # don't use os.path.join() here, since that will call
                # __fspath__() on the S3Path, which downloads the file
                path = f"{root}/{fname}"
                if is_valid_file(path):  # type: ignore
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = (
            "Found no valid file for the classes "
            f"{', '.join(sorted(empty_classes))}. "
        )
        if extensions is not None:
            msg += "Supported extensions are: "
            f"{extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


class ImageFolderS3(DatasetFolderS3):
    """An S3-backed image dataset; see torchvision.datasets.ImageFolder"""

    def __init__(
        self,
        s3_root: str,
        local_root: str | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Callable[[str], bool] | None = None,
        force_use_cached: bool = False,
    ):
        # adapted from torchvision.datasets.ImageFolder
        super().__init__(
            s3_root=s3_root,
            local_root=local_root,
            loader=loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            force_use_cached=force_use_cached,
        )
        self.imgs = self.samples
