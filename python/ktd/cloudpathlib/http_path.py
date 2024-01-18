import os
from typing import TYPE_CHECKING

from cloudpathlib.cloudpath import CloudPath, register_path_class

if TYPE_CHECKING:
    from ktd.cloudpathlib.http_client import HttpClient


@register_path_class("http")
class HttpPath(CloudPath):
    cloud_prefix: str = "http://" 
    client: "HttpClient"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def drive(self) -> str:
        return ""

    def is_dir(self) -> bool:
        return False

    def is_file(self) -> bool:
        return self.exists()

    def mkdir(self, parents=False, exist_ok=False):
        raise RuntimeError("`touch` not supported for http paths")

    def touch(self, exist_ok: bool = True):
        raise RuntimeError("`touch` not supported for http paths")

    def stat(self) -> os.stat_result:
        meta = self.client._get_metadata(self)
        # TODO exceptions
        mtime = (
            int(meta["last_modified"].timestamp())
            if meta["last_modified"] is not None
            else None
        )
        return os.stat_result(
            (
                None,  # type: ignore mode
                None,  # ino
                self.cloud_prefix,  # dev,
                None,  # nlink,
                None,  # uid,
                None,  # gid,
                meta["size"],  # size,
                None,  # atime,
                mtime,  # mtime,
                None,  # ctime,
            )
        )

    @property
    def etag(self):
        return self.client._get_metadata(self).get("etag")

@register_path_class("https")
class HttpsPath(HttpPath):
    cloud_prefix: str = "https://"
    client: "HttpClient"