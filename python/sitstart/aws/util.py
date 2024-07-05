import logging
import os
import subprocess
from typing import Optional

import boto3
import botocore
import botocore.exceptions

from sitstart.logging import get_logger

logger = get_logger(__name__)

logging.getLogger("botocore.credentials").setLevel(logging.WARNING)


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


def update_aws_env(
    session: boto3.Session | None = None, profile: str | None = None
) -> boto3.Session:
    """Update AWS environment variables.

    Useful for components like pyarrow that rely on environment
    variables for AWS credentials.
    """
    if session and profile:
        raise ValueError("Only one of session or profile should be provided.")

    if not session or profile is not None:
        _ = os.environ.pop("AWS_ACCESS_KEY_ID", None)
        _ = os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
        _ = os.environ.pop("AWS_SESSION_TOKEN", None)
        _ = os.environ.pop("AWS_DEFAULT_REGION", None)
        session = get_aws_session(profile=profile)
    credentials = session.get_credentials()

    if credentials.access_key:
        os.environ["AWS_ACCESS_KEY_ID"] = credentials.access_key
    if credentials.secret_key:
        os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
    if credentials.token:
        os.environ["AWS_SESSION_TOKEN"] = credentials.token
    if session.region_name:
        os.environ["AWS_DEFAULT_REGION"] = session.region_name

    return session
