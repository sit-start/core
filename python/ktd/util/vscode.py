from enum import Enum
import os

from ktd.logging import get_logger
from ktd.util.run import run

logger = get_logger(__name__)

DEFAULT_WORKSPACE = "/home/ec2-user/dev/dev.code-workspace"


class VSCodeTarget(Enum):
    FILE = "file"
    FOLDER = "folder"


def open_vscode_over_ssh(
    hostname: str,
    target: str | VSCodeTarget = VSCodeTarget.FILE,
    path: str = DEFAULT_WORKSPACE,
) -> None:
    target = target if isinstance(target, VSCodeTarget) else VSCodeTarget(target)
    logger.info(f"Opening {os.path.basename(path)} in VS Code on {hostname}")
    run(
        [
            "code",
            "--",
            f"--{target.value}-uri",
            f"vscode-remote://ssh-remote+{hostname}{path}",
        ]
    )
