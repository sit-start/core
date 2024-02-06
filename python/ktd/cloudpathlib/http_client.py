import mimetypes
import os
import urllib.error
import urllib.request
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Iterable

import requests
from cloudpathlib.client import Client, register_client_class
from cloudpathlib.enums import FileCacheMode
from dateutil.parser import parse as parsedate
from ktd.cloudpathlib.http_path import HttpPath
from ktd.logging import get_logger

logger = get_logger(__name__)


@register_client_class("http")  # TODO: more testing for http
@register_client_class("https")
class HttpClient(Client):
    def __init__(
        self,
        file_cache_mode: str | FileCacheMode | None = None,
        local_cache_dir: str | os.PathLike | None = None,
        content_type_method: Callable | None = mimetypes.guess_type,
    ):
        super().__init__(
            local_cache_dir=local_cache_dir,
            content_type_method=content_type_method,
            file_cache_mode=file_cache_mode,
        )
        # TODO: consider using requests_cache for more sophisicated
        # caching / decisions around missing last-modified
        self._session = requests.Session()

    def _get_metadata(self, cloud_path: HttpPath) -> dict[str, Any]:
        with self._session.head(cloud_path.as_uri(), allow_redirects=True) as response:
            headers = response.headers
            if "Last-Modified" in headers:
                last_modified = parsedate(headers["Last-Modified"])
            else:
                logger.warning(
                    f"No Last-Modified header for {cloud_path}. "
                    "Using current time. File will not be cached."
                )
                last_modified = datetime.now()
            return {
                "last_modified": last_modified,
                "size": headers.get("Content-Length", None),
                "etag": headers.get("ETag", None),
                "content_type": headers["Content-Type"],
            }

    def _exists(self, cloud_path: HttpPath) -> bool:
        try:
            with self._session.head(cloud_path.as_uri(), allow_redirects=True):
                return True
        except urllib.error.HTTPError:
            return False

    def _download_file(self, cloud_path: HttpPath, local_path: str | PathLike) -> Path:
        local_path = Path(local_path)
        chunk_size = 1024
        with open(local_path, "wb", buffering=chunk_size) as f:
            with self._session.get(
                cloud_path.as_uri(), stream=True, allow_redirects=True
            ) as response:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        f.flush()
        return local_path

    def _list_dir(
        self, cloud_path: HttpPath, recursive: bool
    ) -> Iterable[tuple[HttpPath, bool]]:
        raise RuntimeError("Not supported for http paths")

    def _move_file(
        self, src: HttpPath, dst: HttpPath, remove_src: bool = True
    ) -> HttpPath:
        raise RuntimeError("Not supported for http paths")

    def _remove(self, path: HttpPath, missing_ok: bool = True) -> None:
        raise RuntimeError("Not supported for http paths")

    def _upload_file(
        self, local_path: str | os.PathLike, cloud_path: HttpPath
    ) -> HttpPath:
        raise RuntimeError("Not supported for http paths")
