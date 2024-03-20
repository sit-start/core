import subprocess
import sys
import tempfile
from enum import Enum
from pathlib import Path

from ktd.logging import get_logger
from ktd.util.string import strip_ansi_codes

logger = get_logger(__name__)


class Output(Enum):
    STD = "std"
    QUIET = "quiet"
    FILE = "file"


def run(cmd: list[str], output: str | Output = Output.STD, check: bool = True) -> None:
    output = output if isinstance(output, Output) else Output(output)

    logger.debug(f"Running command: {' '.join(cmd)}")

    if output == Output.STD:
        subprocess.run(cmd, check=check)
        return

    if output == Output.QUIET:
        try:
            subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check
            )
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
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=check)
        log_path.write_text(strip_ansi_codes(log_path.read_text()))
