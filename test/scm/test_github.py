import os
from types import SimpleNamespace
from unittest import mock

from ktd.scm.github import create_private_fork, get_ssh_url, get_user, _run


@mock.patch("ktd.scm.github.run")
def test__run(run_mock):
    cmd = "ls -la"
    _run(cmd)
    run_mock.assert_called_once_with(["ls", "-la"], output="capture")


@mock.patch("ktd.scm.github._run")
def test_get_user(_run_mock):
    _run_mock.return_value = SimpleNamespace(stdout='{"login": "user"}')

    assert get_user() == "user"
    _run_mock.assert_called_once_with("gh api user")


def test_get_ssh_url():
    assert get_ssh_url("user", "repo") == "git@github.com:user/repo"


@mock.patch("ktd.scm.github._run")
def test_create_private_fork(_run_mock):
    repo = "repo"
    account = "org"
    fork_name = "fork"
    fork = get_ssh_url(account, fork_name)
    repo_url = f"git@github.com:{account}/{repo}"
    fork_dir = f"{os.getcwd()}/{fork_name}"

    create_private_fork(repo_url, fork_name, clone=True, org=account)

    _run_mock.assert_has_calls(
        [
            mock.call(f"git clone --bare {repo_url}", cwd=mock.ANY),
            mock.call(f"gh repo create {account}/{fork_name} --private"),
            mock.call(f"git push --mirror {fork}", cwd=mock.ANY),
            mock.call(f"git clone {fork}"),
            mock.call(f"git remote add upstream {repo_url}", cwd=fork_dir),
            mock.call("git remote set-url --push upstream DISABLE", cwd=fork_dir),
        ]
    )
