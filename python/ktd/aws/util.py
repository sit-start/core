import subprocess


def sso_login(profile_name=None) -> None:
    cmd = ["aws", "sso", "login"] + (
        ["--profile", profile_name] if profile_name is not None else []
    )
    subprocess.run(cmd)
