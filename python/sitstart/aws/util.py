import logging
import os
import subprocess
from typing import Optional

import boto3
import botocore
import botocore.exceptions

from sitstart.logging import get_logger

logger = get_logger(__name__)


def is_logged_in(session: Optional[boto3.Session] = None) -> bool:
    """Check if the given session is logged in"""
    sts = session.client("sts") if session else boto3.client("sts")

    credentials_logger = logging.getLogger("botocore.credentials")
    credentials_log_level = credentials_logger.level
    credentials_logger.setLevel(logging.ERROR)

    try:
        sts.get_caller_identity()  # type: ignore
        result = True
    except botocore.exceptions.SSOError as e:
        logger.warning(e)
        result = False

    credentials_logger.setLevel(credentials_log_level)

    return result


def sso_login(profile_name=None) -> None:
    """Login to AWS via SSO if not already logged in"""
    if os.getenv("GITHUB_ACTIONS", False):
        # rely on OIDC and the sitstart-github-action role for tests
        logger.warning("Skipping AWS SSO login in GitHub Actions environment.")
        return
    if is_logged_in(boto3.Session(profile_name=profile_name)):
        logger.debug("AWS SSO session has valid credentials; skipping login")
        return
    logger.info("Logging in to AWS via SSO.")
    cmd = ["aws", "sso", "login"] + (
        ["--profile", profile_name] if profile_name is not None else []
    )
    subprocess.run(cmd)


def get_aws_session(profile: str | None = None) -> boto3.Session:
    sso_login(profile_name=profile)
    return boto3.Session(profile_name=profile)
