import subprocess
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

from ktd.logging import get_logger
from ktd.util.string import strip_ansi_codes

logger = get_logger(__name__)


class Output(Enum):
    STD = "std"
    CAPTURE = "capture"
    FILE = "file"


def run(
    cmd: list[str], output: str | Output = Output.STD, check: bool = True, **kwargs: Any
) -> subprocess.CompletedProcess[bytes]:
    output = output if isinstance(output, Output) else Output(output)

    kwargs["check"] = check
    kwargs["capture_output"] = output == Output.CAPTURE
    kwargs.pop("stdout", None)
    kwargs.pop("stderr", None)

    logger.debug(f"Running command: {' '.join(cmd)}")

    if output == Output.STD:
        return subprocess.run(cmd, **kwargs)

    if output == Output.CAPTURE:
        try:
            return subprocess.run(cmd, **kwargs)
        except subprocess.CalledProcessError as e:
            logger.error(
                "{exception}\nstdout:\n{stdout}\nstderr:\n{stderr}".format(
                    exception=e,
                    stdout=e.stdout.decode(sys.stdout.encoding),
                    stderr=e.stderr.decode(sys.stderr.encoding),
                )
            )
            raise e

    else:  # output == Output.FILE
        log_path = Path(tempfile.mkdtemp(prefix="/tmp/")) / f"{cmd[0]}.log"
        logger.info(f"Writing {cmd[0]!r} command output to {log_path}")

        with log_path.open("w") as f:
            out = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, **kwargs)
        log_path.write_text(strip_ansi_codes(log_path.read_text()))
        return out
