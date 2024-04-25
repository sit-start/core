from enum import Enum

from ktd.logging import get_logger
from ktd.util.run import run

logger = get_logger(__name__)

DEFAULT_FOLDER = "/home/ec2-user/dev/core"
DEFAULT_TARGET = "folder"


class VSCodeTarget(Enum):
    FILE = "file"
    FOLDER = "folder"


def open_vscode_over_ssh(
    hostname: str,
    target: str | VSCodeTarget = DEFAULT_TARGET,
    path: str = DEFAULT_FOLDER,
) -> None:
    target = target if isinstance(target, VSCodeTarget) else VSCodeTarget(target)
    logger.info(f"Opening {target.value} {path!r} in VS Code on {hostname}")
    run(
        [
            "code",
            "--",
            f"--{target.value}-uri",
            f"vscode-remote://ssh-remote+{hostname}{path}",
        ]
    )
