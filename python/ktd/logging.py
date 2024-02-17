import logging
from enum import Enum
from typing import Optional


class Format(Enum):
    GLOG = "glog"
    SIMPLE = "simple"
    BARE = "bare"


FORMATS = {
    Format.GLOG: "%(levelname).1s%(asctime)s.%(msecs)d %(process)d %(filename)s:%(lineno)d] %(message)s",
    Format.SIMPLE: "%(levelname).1s %(filename)s:%(lineno)d] %(message)s",
    Format.BARE: "%(message)s",
}
DATE_FORMAT = "%m%d %H:%M:%S"


def get_logger(
    name: Optional[str] = None,
    format: str | Format = Format.GLOG,
    level: int = logging.INFO,
) -> logging.Logger:
    """Returns a default-configured logger with glog-style formatting"""
    # Note that this does update the root logging config; that can be
    # reset with, e.g., logging.basicConfig(force=True)
    if isinstance(format, str):
        try:
            format_str = FORMATS[Format(format)]
        except ValueError:
            format_str = format
    else:
        format_str = FORMATS[format]
    logging.basicConfig(format=format_str, force=True, datefmt=DATE_FORMAT, level=level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
