import logging
from enum import Enum
from typing import Optional


class Format(Enum):
    GLOG = "glog"
    SIMPLE = "simple"
    BARE = "bare"


MSG_FORMATS = {
    Format.GLOG: "%(levelname).1s%(asctime)s %(process)d %(filename)s:%(lineno)d] %(message)s",
    Format.SIMPLE: "%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s",
    Format.BARE: "%(message)s",
}
DATE_FORMATS = {
    Format.SIMPLE: "%H:%M:%S",
}


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
            msg_fmt = MSG_FORMATS[Format(format)]
            date_fmt = DATE_FORMATS.get(Format(format), None)
        except ValueError:
            msg_fmt = format
            date_fmt = None
    else:
        msg_fmt = MSG_FORMATS[format]
        date_fmt = DATE_FORMATS.get(format, None)
    logging.basicConfig(format=msg_fmt, force=True, datefmt=date_fmt, level=level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
