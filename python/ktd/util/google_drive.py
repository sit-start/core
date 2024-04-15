from collections import deque
from typing import Any

from ktd.logging import get_logger

logger = get_logger(__name__)

File = dict[str, Any]

FOLDER_TYPE = "application/vnd.google-apps.folder"
PAGE_SIZE = 50


def _resolve_file_fields(
    file_fields: list[str] | None, required: list[str] | None = None
) -> list[str]:
    if file_fields is None:
        return ["*"]
    return list(set(file_fields + ["id"] + (required if required else [])))


def get_path(
    service: Any, file_id: str, file_fields: list[str] | None = None
) -> list[dict[str, str]]:
    file_fields = _resolve_file_fields(file_fields, required=["name", "parents"])
    field_str = ", ".join(file_fields)
    path = deque()
    while file_id:
        file = service.files().get(fileId=file_id, fields=field_str).execute()
        path.appendleft(dict(name=file.get("name"), id=file.get("id")))
        file_id = parents[0] if (parents := file.get("parents")) and parents[0] else ""
    return list(path)


def get_file(service: Any, file_id: str, fields: list[str] | None = None) -> File:
    file_fields = _resolve_file_fields(file_fields=fields, required=["id", "name"])
    return service.files().get(fileId=file_id, fields=", ".join(file_fields)).execute()


def get_file_from_path(
    service: Any, path: str, fields: list[str] | None = None
) -> File:
    """Get a File dictionary with the given fields for the given path.

    `path` must be an absolute path and start with '/' or 'My Drive'.
    """
    if path.startswith("My Drive"):
        path = path[8:]
    if not path.startswith("/"):
        raise ValueError("Path must be absolute.")
    path = path.strip("/")
    if not path:
        raise ValueError("Path must be non-empty.")
    file = get_file(service, "root", fields=["id", "name"])
    for el in path.split("/"):
        query = (
            f" '{file['id']}' in parents and trashed=false "
            f"and mimeType='{FOLDER_TYPE}' and name='{el}'"
        )
        results = (
            service.files()
            .list(q=query, fields="files(id, name)", pageSize=2)
            .execute()
        )
        if not results.get("files"):
            raise FileNotFoundError(f"Folder '{el}' not found in '{file['name']}'")
        if len(results.get("files", [])) > 1:
            raise ValueError(
                f"Multiple folders with name '{el}' found in '{file['name']}'"
            )

        file = results["files"][0]
    return get_file(service, file["id"], fields=fields)


def get_folder_contents(
    service: Any,
    folder: File | str,
    file_fields: list[str] | None = None,
    folders_only: bool = False,
) -> tuple[File, list[File], list[File]]:
    """Returns a 3-tuple of folder, folders and files in the folder.

    `folder` can be a File dictionary or the file id.
    """
    file_fields = _resolve_file_fields(file_fields, required=["id", "name", "mimeType"])
    field_str = f"nextPageToken, files({', '.join(file_fields)})"
    if isinstance(folder, str):
        folder = get_file(service, folder, fields=file_fields)
    query = f" '{folder['id']}' in parents and trashed=false "
    if folders_only:
        query += f" and mimeType='{FOLDER_TYPE}'"
    files, folders = [], []
    next_page_token = None
    kwargs: dict[str, Any] = dict(q=query, fields=field_str, pageSize=PAGE_SIZE)

    while True:
        kwargs["pageToken"] = next_page_token
        results = service.files().list(**kwargs).execute()
        next_page_token = results.get("nextPageToken")
        for item in results.get("files", []):
            out_list = folders if item.get("mimeType") == FOLDER_TYPE else files
            out_list.append(item)
        if not next_page_token:
            break

    return folder, folders, files


def walk(
    service: Any,
    folder_id: str,
    fields: list[str] | None = None,
    topdown: bool = True,
    folders_only: bool = False,
    folder_ids_to_ignore: list[str] | None = None,
):
    folder_ids_to_ignore = folder_ids_to_ignore or []
    kwargs: Any = dict(
        fields=fields,
        topdown=topdown,
        folders_only=folders_only,
        folder_ids_to_ignore=folder_ids_to_ignore,
    )
    while folder_id and folder_id not in folder_ids_to_ignore:
        top, folders, files = get_folder_contents(
            service, folder_id, file_fields=fields, folders_only=folders_only
        )
        folder_id = ""
        if topdown:
            yield top, folders, files
            for folder in folders:
                yield from walk(service, folder["id"], **kwargs)
        else:
            for folder in folders:
                yield from walk(service, folder["id"], **kwargs)
            yield top, folders, files


def get_folder_paths(
    service: Any,
    root_folder_id: str | None = None,
    folder_ids_to_ignore: list[str] | None = None,
) -> dict[str, str]:
    if root_folder_id is None or root_folder_id == "root":
        root_folder_id = get_file(service, "root", fields=["id"])["id"]
    assert root_folder_id is not None
    root_path = get_path(service, root_folder_id)
    paths = {root_folder_id: "/".join(el["name"] for el in root_path)}

    for root, folders, _ in walk(
        service,
        root_folder_id,
        fields=["name", "id"],
        folders_only=True,
        folder_ids_to_ignore=folder_ids_to_ignore,
    ):
        for folder in folders:
            paths[folder["id"]] = f"{paths[root['id']]}/{folder['name']}"
            logger.debug(f"{paths[folder['id']]} ({folder['id']})")

    return paths
