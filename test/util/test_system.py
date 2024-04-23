from unittest.mock import patch, call

import pytest

from ktd.util.system import (
    _copy_filtered_files,
    _filtered_system_files,
    _get_path_and_constraints,
    _get_variant_path,
    _is_specific_system_variant,
    _system_files,
    deploy_system_files,
    deploy_system_files_from_filesystem,
    push_system_files,
    deploy_dotfiles,
    SYSTEM_ARCHIVE_URL,
    SYSTEM_FILE_ROOT,
)


def test__get_path_and_constraints():
    assert _get_path_and_constraints("path##key.value") == (
        "path",
        {"key": "value"},
    )
    assert _get_path_and_constraints("path##key1.value1,key2.value2") == (
        "path",
        {"key1": "value1", "key2": "value2"},
    )
    assert _get_path_and_constraints("path") == ("path", {})


def test__get_variant_path():
    assert _get_variant_path("path", {"key": "value"}) == "path##key.value"
    assert _get_variant_path("path", {}) == "path"


def test__is_specific_system_variant():
    attr = {"key": "value"}
    assert _is_specific_system_variant({"key": "value"}, system_attributes=attr)
    assert not _is_specific_system_variant({}, system_attributes=attr)
    assert not _is_specific_system_variant({"key": "value1"}, system_attributes=attr)


@patch("ktd.util.system._run")
@patch("ktd.util.system._filtered_system_files")
def test__copy_filtered_files(mock__filtered_system_files, mock__run):
    mock__filtered_system_files.return_value = [("a/b##os.Linux", "a/b"), ("c", "c")]
    _copy_filtered_files("src", "dest", "cmd")
    mock__run.assert_has_calls(
        [
            call("mkdir -p dest/a"),
            call("cmd src/a/b##os.Linux dest/a/b"),
            call("mkdir -p dest/"),
            call("cmd src/c dest/c"),
        ]
    )


@patch("glob.glob")
@patch("os.path.isfile")
def test__system_files(mock_isfile, mock_glob):
    mock_glob.return_value = ["file1", "dir2/file2"]
    mock_isfile.return_value = True
    assert list(_system_files("root_dir")) == ["file1", "dir2/file2"]


@patch("ktd.util.system._system_files")
def test__filtered_system_files(mock__system_files):
    system_attributes = {"arch": "x86_64", "os": "Linux"}

    mock__system_files.return_value = ["a##arch.arm64", "a##arch.x86_64", "a", "b"]
    result = list(_filtered_system_files(system_attributes=system_attributes))
    assert result == [("a##arch.x86_64", "a"), ("b", "b")]

    mock__system_files.return_value = ["a##arch.x86_64", "a##os.Linux"]
    with pytest.raises(RuntimeError):
        list(_filtered_system_files(system_attributes=system_attributes))

    mock__system_files.return_value = ["a##arch.arm64", "a"]
    result = list(_filtered_system_files(system_attributes=system_attributes))
    assert result == [("a", "a")]


@patch("ktd.util.system._run")
@patch("ktd.util.system._system_files")
@patch("ktd.util.system.Path.read_text")
@patch("ktd.util.system.Path.write_text")
@patch("ktd.util.system.NamedTemporaryFile")
@patch("ktd.util.system.system_file_archive_url")
@patch("ktd.util.system.sso_login")
def test_push_system_files(
    mock_sso_login,
    mock_system_file_archive_url,
    mock_temp_file,
    mock_write_text,
    mock_read_text,
    mock__system_files,
    mock__run,
):
    new_archive_url = "s3://bucket/with/new_archive.tar.gz"
    mock_system_file_archive_url.return_value = new_archive_url
    mock_temp_file.return_value.__enter__.return_value.name = "temp_files.tar.gz"
    mock_read_text.return_value = (
        '{"archive_url": "s3://bucket/with/existing_archive.tar.gz"}'
    )
    mock__system_files.return_value = ["file1", "file2"]

    push_system_files()

    mock_sso_login.assert_called_once()
    mock__run.assert_has_calls(
        [
            call(f"tar -czf temp_files.tar.gz -C {SYSTEM_FILE_ROOT} file1 file2"),
            call(f"aws s3 --quiet cp temp_files.tar.gz {new_archive_url}"),
        ]
    )
    mock_write_text.assert_called_once()


@patch("ktd.util.system._copy_filtered_files")
def test_deploy_system_files_from_filesystem(mock__copy_filtered_files):
    deploy_system_files_from_filesystem(src_dir="a", dest_dir="b")
    mock__copy_filtered_files.assert_called_with(
        src_dir="a", dest_dir="b", cmd="ln -sf", as_root=False
    )


@patch("ktd.util.system._run")
@patch("ktd.util.system._copy_filtered_files")
@patch("os.makedirs")
@patch("ktd.util.system.Path.read_text")
@patch("ktd.util.system.TemporaryDirectory")
@patch("ktd.util.system.sso_login")
def test_deploy_system_files(
    mock_sso_login,
    mock_temp_dir,
    mock_read_text,
    mock_makedirs,
    mock__copy_filtered_files,
    mock__run,
):
    mock_temp_dir.return_value.__enter__.return_value = "temp_dir"
    mock_read_text.return_value = (
        f'{{"archive_url": "{SYSTEM_ARCHIVE_URL.format("1234")}"}}'
    )
    deploy_system_files(dest_dir="/", as_root=True)

    mock_makedirs.assert_called_with("temp_dir/files")
    mock__run.assert_has_calls(
        [
            call(
                "aws s3 cp s3://sitstart/system/files/1234.tar.gz temp_dir/1234.tar.gz"
            ),
            call("tar -xzf temp_dir/1234.tar.gz -C temp_dir/files"),
        ]
    )
    mock__copy_filtered_files.assert_called_with(
        src_dir="temp_dir/files", dest_dir="/", cmd="cp", as_root=True
    )


@patch("ktd.util.system._run")
def test_deploy_dotfiles(mock__run):
    username = "some_user"
    deploy_dotfiles(username)
    mock__run.assert_called_with(
        f"bash -l -c 'deploy_yadm_repo git@github.com:{username}/dotfiles.git'"
    )
