import logging
import subprocess
from typing import Optional

import boto3
import botocore
import botocore.exceptions
from ktd.logging import get_logger

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
    if is_logged_in(boto3.Session(profile_name=profile_name)):
        logger.info("AWS SSO session has valid credentials; skipping login")
        return
    cmd = ["aws", "sso", "login"] + (
        ["--profile", profile_name] if profile_name is not None else []
    )
    subprocess.run(cmd)
