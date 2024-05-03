from unittest import mock

from sitstart.util.vscode import DEFAULT_FOLDER, DEFAULT_TARGET, open_vscode_over_ssh


@mock.patch("sitstart.util.vscode.run")
def test_open_vscode_over_ssh(mock_run):
    hostname = "example.com"
    path = "/path/to/workspace"
    target = "file"
    expected_uri = f"vscode-remote://ssh-remote+{hostname}{path}"

    open_vscode_over_ssh(hostname, target, path)
    mock_run.assert_called_once()
    assert f"--{target}-uri" in mock_run.call_args[0][0]
    assert expected_uri in mock_run.call_args[0][0]

    mock_run.reset_mock()
    expected_uri = f"vscode-remote://ssh-remote+{hostname}{DEFAULT_FOLDER}"

    open_vscode_over_ssh(hostname)
    mock_run.assert_called_once()
    assert f"--{DEFAULT_TARGET}-uri" in mock_run.call_args[0][0]
    assert expected_uri in mock_run.call_args[0][0]
