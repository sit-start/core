from pathlib import Path

from sitstart.cloudpathlib import CloudPath


def get_local_path(path: CloudPath | Path) -> Path:
    """Returns the local path for the given CloudPath.

    CloudPath.fspath caches files; this caches directories recursively
    as well.
    """
    if isinstance(path, Path):
        return path

    if path.is_file():
        return Path(path.fspath)

    for file_or_dir in path.glob("**/*"):
        if file_or_dir.is_file():
            _ = file_or_dir.fspath

    return Path(path.fspath)
