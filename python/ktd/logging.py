import logging
from typing import Optional

FORMAT = "%(levelname).1s%(asctime)s.%(msecs)d %(process)d %(filename)s:%(lineno)d] %(message)s"
DATE_FORMAT = "%m%d %H:%M:%S"


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Returns a default-configured logger with glog-style formatting"""
    # Note that this does update the root logging config; that can be
    # reset with, e.g., logging.basicConfig(force=True)
    logging.basicConfig(format=FORMAT, force=True, datefmt=DATE_FORMAT, level=level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
