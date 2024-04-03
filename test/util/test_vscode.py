from unittest import mock

from ktd.util.vscode import DEFAULT_WORKSPACE, VSCodeTarget, open_vscode_over_ssh


@mock.patch("ktd.util.vscode.run")
def test_open_vscode_over_ssh(mock_run):
    hostname = "example.com"
    path = "/path/to/workspace"
    target = VSCodeTarget.FOLDER
    expected_uri = f"vscode-remote://ssh-remote+{hostname}{path}"

    open_vscode_over_ssh(hostname, target, path)

    mock_run.assert_called_once_with(["code", "--", "--folder-uri", expected_uri])


@mock.patch("ktd.util.vscode.run")
def test_open_vscode_over_ssh_with_default_workspace(mock_run):
    hostname = "example.com"
    path = DEFAULT_WORKSPACE
    expected_uri = f"vscode-remote://ssh-remote+{hostname}{path}"

    open_vscode_over_ssh(hostname)

    mock_run.assert_called_once_with(["code", "--", "--file-uri", expected_uri])
