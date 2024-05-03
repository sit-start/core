from unittest.mock import MagicMock, patch

import pytest

from sitstart.util.google_drive import (
    FOLDER_TYPE,
    PAGE_SIZE,
    _resolve_file_fields,
    get_file,
    get_file_from_path,
    get_folder_contents,
    get_folder_paths,
    get_path,
    walk,
)


def test__resolve_file_fields():
    assert _resolve_file_fields(None) == ["*"]
    assert _resolve_file_fields(None, required=["a"]) == ["*"]
    assert set(_resolve_file_fields(["a", "b"])) == set(["a", "b", "id"])
    assert set(_resolve_file_fields(["a"], ["b", "c"])) == set(["id", "a", "b", "c"])


def test_get_path():
    service_mock = MagicMock()
    service_mock.files().get().execute.side_effect = [
        {"parents": ["parent1_id"], "name": "file", "id": "file_id"},
        {"parents": None, "name": "parent1", "id": "parent1_id"},
    ]

    assert get_path(service_mock, "file_id") == [
        {"name": "parent1", "id": "parent1_id"},
        {"name": "file", "id": "file_id"},
    ]


def test_get_file():
    files = [
        {"name": "file1", "id": "id1"},
        {"name": "file2", "id": "id2", "field1": "value1"},
    ]
    service_mock = MagicMock()
    service_mock.files().get().execute.side_effect = files

    assert get_file(service_mock, "id1") == files[0]
    assert get_file(service_mock, "id2", fields=["field1"]) == files[1]


def test_get_file_from_path():
    service_mock = MagicMock()
    service_mock.files().list().execute.return_value = {
        "files": [{"name": "file", "id": "file_id"}]
    }

    execute_side_effect = [
        {"name": "My Drive", "id": "root_id"},
        {"name": "file", "id": "file_id"},
    ]

    service_mock.files().get().execute.side_effect = execute_side_effect
    assert get_file_from_path(service_mock, "My Drive/file") == {
        "name": "file",
        "id": "file_id",
    }

    service_mock.files().get().execute.side_effect = execute_side_effect
    assert get_file_from_path(service_mock, "/file") == {
        "name": "file",
        "id": "file_id",
    }

    service_mock.files().get().execute.side_effect = execute_side_effect
    with pytest.raises(ValueError):
        get_file_from_path(service_mock, "file")

    service_mock.files().list().execute.return_value = {"files": []}
    with pytest.raises(FileNotFoundError):
        get_file_from_path(service_mock, "/nonexistent_file")

    service_mock.files().list().execute.return_value = {
        "files": [{"name": "file", "id": "file_id"}] * 2
    }
    with pytest.raises(ValueError):
        get_file_from_path(service_mock, "/file")

    with pytest.raises(ValueError):
        get_file_from_path(service_mock, "/")


def test_get_folder_contents():
    root_folder = {"name": "folder", "id": "folder_id", "mimeType": FOLDER_TYPE}
    sub_folder = {"name": "sub_folder", "id": "sub_folder_id", "mimeType": FOLDER_TYPE}
    file_type = "application/vnd.google-apps.file"

    service_mock = MagicMock()
    service_mock.files().get().execute.return_value = root_folder
    execute_results = [
        {
            "files": [
                {"name": f"file{i:02d}", "id": f"file{i:02d}_id", "mimeType": file_type}
                for i in range(PAGE_SIZE)
            ],
            "nextPageToken": "token",
        },
        {
            "files": [
                {"name": "file", "id": "file_id", "mimeType": file_type},
                sub_folder,
            ],
            "nextPageToken": None,
        },
    ]

    service_mock.files().list().execute.side_effect = execute_results
    folder, folders, files = get_folder_contents(service_mock, "folder_id")
    assert folder == root_folder
    assert folders == [sub_folder]
    assert len(files) == PAGE_SIZE + 1

    service_mock.files().list().execute.side_effect = execute_results
    _, _, _ = get_folder_contents(service_mock, "folder_id", folders_only=True)
    query = service_mock.files().list.call_args[1].get("q")
    assert query and f"mimeType='{FOLDER_TYPE}'" in query


@patch("sitstart.util.google_drive.get_folder_contents")
def test_walk(get_folder_contents_mock):
    service_mock = MagicMock()

    file_type = "application/vnd.google-apps.file"
    folders = [
        {"name": f"folder{i:02d}", "id": f"folder{i:02d}_id", "mimeType": FOLDER_TYPE}
        for i in range(4)
    ]
    files = [
        {"name": f"file{i:02d}", "id": f"file{i:02d}_id", "mimeType": file_type}
        for i in range(4)
    ]

    # returned results correspond to contents created as follows:
    # > mkdir -p folder00/{folder01/folder03,folder02}
    # > touch {file00,folder00/{folder01/folder03/file03,folder02/{file01,file02}}}
    get_folder_contents_return_value = [
        (folders[0], folders[1:3], files[:1]),
        (folders[1], folders[3:4], []),
        (folders[3], [], files[3:4]),
        (folders[2], [], files[1:3]),
    ]

    expected_results_topdown = get_folder_contents_return_value
    get_folder_contents_mock.side_effect = get_folder_contents_return_value
    results = list(walk(service_mock, folders[0]["id"]))
    assert results == expected_results_topdown

    expected_results_bottomup = [
        (folders[3], [], files[3:4]),
        (folders[1], folders[3:4], []),
        (folders[2], [], files[1:3]),
        (folders[0], folders[1:3], files[:1]),
    ]
    get_folder_contents_mock.side_effect = get_folder_contents_return_value
    results = list(walk(service_mock, folders[0]["id"], topdown=False))
    assert results == expected_results_bottomup


@patch("sitstart.util.google_drive.walk")
@patch("sitstart.util.google_drive.get_path")
@patch("sitstart.util.google_drive.get_file")
def test_get_folder_paths(get_file_mock, get_path_mock, walk_mock):
    service_mock = MagicMock()
    folders = [
        {"name": f"folder{i:02d}", "id": f"folder{i:02d}_id", "mimeType": FOLDER_TYPE}
        for i in range(4)
    ]
    root_folder = {"name": "My Drive", "id": "root_id", "mimeType": FOLDER_TYPE}
    get_file_mock.return_value = root_folder
    get_path_mock.return_value = [root_folder]

    # returned results correspond to contents created as follows:
    # > mkdir -p folder00/{folder01/folder03,folder02}
    walk_mock.return_value = [
        (root_folder, folders[:1], []),
        (folders[0], folders[1:3], []),
        (folders[1], folders[3:4], []),
        (folders[3], [], []),
        (folders[2], [], []),
    ]

    expected_result = {
        root_folder["id"]: "My Drive",
        folders[0]["id"]: "My Drive/folder00",
        folders[1]["id"]: "My Drive/folder00/folder01",
        folders[2]["id"]: "My Drive/folder00/folder02",
        folders[3]["id"]: "My Drive/folder00/folder01/folder03",
    }
    result = get_folder_paths(service_mock)
    assert result == expected_result
