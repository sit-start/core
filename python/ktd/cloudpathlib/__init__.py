"""Wrapper around cloudpathlib so that http(s) clients/paths are registered"""

from cloudpathlib import CloudPath

from .http_client import HttpClient
from .http_path import HttpPath, HttpsPath

__all__ = [
    "CloudPath",
    "HttpClient",
    "HttpPath",
    "HttpsPath",
]
