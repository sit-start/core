import glob
import json
import os
import shlex
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Generator

from checksumdir import dirhash

from ktd.aws.util import sso_login
from ktd.logging import get_logger
from ktd.util import is_valid_url
from ktd.util.run import run

SYSTEM_FILE_ROOT = f"{os.environ['DEV']}/core/system/files"
SYSTEM_ARCHIVE_URL = "s3://sitstart/system/files/{}.tar.gz"
SYSTEM_CONFIG_PATH = f"{os.environ['DEV']}/core/system/config.json"
SYSTEM_ATTRIBUTES = {
    "arch": os.uname().machine,
    "os": os.uname().sysname,
    "hostname": os.uname().nodename,
}


logger = get_logger(__name__)


def _run(cmd: str, **kwargs):
    run(shlex.split(cmd), **kwargs)


def _get_path_and_constraints(path: str) -> tuple[str, dict[str, str]]:
    path_constraints_sep = "##"
    if path_constraints_sep not in path:
        return path, {}
    path, suffix = path.rsplit(path_constraints_sep, 1)
    constraints = {}
    for constraint in suffix.split(","):
        key, value = constraint.split(".", 1)
        constraints[key] = value
    return path, constraints


def _get_variant_path(path: str, constraints: dict[str, str]) -> str:
    if not constraints:
        return path
    return f"{path}##{','.join(f'{k}.{v}' for k, v in constraints.items())}"


def _is_specific_system_variant(
    constraints: dict[str, str], system_attributes: dict[str, str] | None = None
) -> bool:
    system_attributes = system_attributes or SYSTEM_ATTRIBUTES
    if not constraints:
        return False
    for key, value in constraints.items():
        if system_attributes.get(key) != value:
            return False
    return True


def _copy_filtered_files(
    src_dir: str, dest_dir: str, cmd: str, as_root: bool = False
) -> None:
    sudo = "sudo " if as_root else ""
    for src_file, dest_file in _filtered_system_files(root_dir=src_dir):
        logger.debug(f"{src_dir}/{src_file} -> {dest_dir}/{dest_file}")
        _run(f"{sudo}mkdir -p {dest_dir}/{os.path.dirname(dest_file)}")
        _run(f"{sudo}{cmd} {src_dir}/{src_file} {dest_dir}/{dest_file}")


def _system_files(
    root_dir: str = SYSTEM_FILE_ROOT,
) -> Generator[str, None, None]:
    for file in glob.glob("**", root_dir=root_dir, recursive=True):
        if os.path.isfile(f"{root_dir}/{file}"):
            yield file


# Implements a subset of yadm-style alternative file notation.
# @source: https://yadm.io/docs/alternates
def _filtered_system_files(
    root_dir: str = SYSTEM_FILE_ROOT,
    system_attributes: dict[str, str] | None = None,
) -> Generator[tuple[str, str], None, None]:
    system_attributes = system_attributes or SYSTEM_ATTRIBUTES

    # get all files and their variants
    file_variants, files_with_defaults = {}, []
    for path_with_constraints in _system_files(root_dir=root_dir):
        path, constraints = _get_path_and_constraints(path_with_constraints)
        if not constraints:
            files_with_defaults.append(path)
        file_variants.setdefault(path, []).append(constraints)

    # find paths with matching variants; if no specific variant is
    # found, use the default (with no suffix) if it exists
    for path, variants in file_variants.items():
        matching_variants = [
            v for v in variants if _is_specific_system_variant(v, system_attributes)
        ]
        if len(matching_variants) > 1:
            raise RuntimeError(f"Multiple variants match {path}: {matching_variants}")
        if matching_variants:
            yield _get_variant_path(path, matching_variants[0]), path
            continue
        if path in files_with_defaults:
            yield path, path


def get_system_config() -> dict:
    if os.path.exists(SYSTEM_CONFIG_PATH):
        return json.loads(Path(SYSTEM_CONFIG_PATH).read_text())
    return {}


def hash_system_files() -> str:
    return dirhash(SYSTEM_FILE_ROOT, "md5")


def system_file_archive_url() -> str:
    return SYSTEM_ARCHIVE_URL.format(hash_system_files())


def push_system_files() -> None:
    """Push repo system files to S3."""
    sso_login()

    dest = system_file_archive_url()
    files = " ".join(_system_files())

    logger.info("Pushing system files to S3.")
    with NamedTemporaryFile(suffix=".tar.gz") as temp_file:
        temp_file.close()
        _run(f"tar -czf {temp_file.name} -C {SYSTEM_FILE_ROOT} {files}")
        _run(f"aws s3 --quiet cp {temp_file.name} {dest}")
    logger.info(f"System files pushed to {dest}.")

    config = get_system_config()
    config["comment"] = "This file is auto-generated."
    config["archive_url"] = dest

    Path(SYSTEM_CONFIG_PATH).write_text(json.dumps(config, indent=2) + "\n")
    logger.info(f"System file config updated at {SYSTEM_CONFIG_PATH}.")


def deploy_system_files_from_filesystem(
    dest_dir: str, src_dir: str = SYSTEM_FILE_ROOT, as_root: bool = False
) -> None:
    """Deploy system files to the local filesystem as symlinks. For development."""
    _copy_filtered_files(
        src_dir=src_dir, dest_dir=dest_dir, cmd="ln -sf", as_root=as_root
    )


def deploy_system_files(dest_dir: str, as_root: bool = False) -> None:
    """Deploy system files from the S3 archive to the local filesystem.

    For final deployment, use `dest_dir = "/", as_root = True`.
    """
    sso_login()

    file_config = json.loads(Path(SYSTEM_CONFIG_PATH).read_text())
    remote_archive_url = file_config["archive_url"]

    with TemporaryDirectory() as temp_dir:
        archive_url = f"{temp_dir}/{os.path.basename(remote_archive_url)}"
        _run(f"aws s3 cp {remote_archive_url} {archive_url}")

        files_dir = f"{temp_dir}/files"
        os.makedirs(files_dir)
        _run(f"tar -xzf {archive_url} -C {files_dir}")
        _copy_filtered_files(
            src_dir=files_dir, dest_dir=dest_dir, cmd="cp", as_root=as_root
        )


def deploy_dotfiles(username: str) -> None:
    repo_url = f"git@github.com:{username}/dotfiles.git"
    if not is_valid_url(f"ssh://{repo_url}"):
        raise ValueError(f"Invalid repo URL: {repo_url}.")
    _run(f"bash -l -c 'deploy_yadm_repo {shlex.quote(repo_url)}'")
