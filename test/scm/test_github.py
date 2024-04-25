from types import SimpleNamespace
from unittest import mock

from callee.strings import Regex

from ktd.scm.github import create_private_fork, get_ssh_url, get_user, _run


@mock.patch("ktd.scm.github.run")
def test__run(run_mock):
    cmd = "command"
    _run(cmd)
    run_mock.assert_called_once_with(["command"], output="capture")


@mock.patch("ktd.scm.github._run")
def test_get_user(_run_mock):
    _run_mock.return_value = SimpleNamespace(stdout='{"login": "user"}')
    assert get_user() == "user"


def test_get_ssh_url():
    assert get_ssh_url("user", "repo") == "git@github.com:user/repo.git"


@mock.patch("ktd.scm.github._run")
def test_create_private_fork(_run_mock):
    repo = "repo"
    account = "org"
    fork_name = "fork"
    fork = get_ssh_url(account, fork_name)
    repo_url = f"git@github.com:{account}/{repo}.git"

    create_private_fork(repo_url, fork_name, clone=True, org=account)
    _run_mock.assert_any_call(Regex(f"git clone.*{repo}.*"), cwd=mock.ANY)
    _run_mock.assert_any_call(f"gh repo create {account}/{fork_name} --private")
    _run_mock.assert_any_call(Regex(f"git clone.*{fork}.*"))

    _run_mock.reset_mock()
    create_private_fork(repo_url, fork_name, clone=False, org=account)
    assert mock.call(Regex(f"git clone.*{fork}.*")) not in _run_mock.mock_calls
