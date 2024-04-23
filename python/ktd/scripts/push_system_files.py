#!/usr/bin/env python3
import sys

from ktd.logging import get_logger
from ktd.util.system import (
    push_system_files,
    get_system_config,
    system_file_archive_url,
)

# NOTE: third-party dependencies, direct and indirect, need to be added
# to `additional_dependencies` in the push-system-files hook in
# .pre-commit-config.yaml.

logger = get_logger(__name__)

# If the system config contains the expected archive URL, which is based
# on a hash of the current system files, we assume that the archive is
# up-to-date and do nothing.
config = get_system_config()
if config["archive_url"] == system_file_archive_url():
    logger.info("System config is up-to-date.")
    sys.exit(0)

# Otherwise, push a new system file archive and update the system config.
try:
    logger.info("System config is out-of-date. Pushing system files.")
    push_system_files()
except Exception as e:
    logger.error(f"Failed to push system files: {e}.")
    sys.exit(1)
