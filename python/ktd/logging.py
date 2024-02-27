import logging
from enum import Enum


class Format(Enum):
    GLOG = "glog"
    SIMPLE = "simple"
    BARE = "bare"


MSG_FORMATS = {
    Format.GLOG: "%(levelname).1s%(asctime)s %(process)d %(filename)s:%(lineno)d] "
    "%(message)s",
    Format.SIMPLE: "%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s",
    Format.BARE: "%(message)s",
}
DATE_FORMATS = {
    Format.SIMPLE: "%H:%M:%S",
}


def get_logger(
    name: str | None = None,
    format: str | Format | None = Format.GLOG,
    level: int = logging.INFO,
    force: bool = False,
) -> logging.Logger:
    kwargs = {
        "level": level,
        "force": force,
    }

    if format:
        if isinstance(format, str):
            try:
                kwargs["format"] = MSG_FORMATS[Format(format)]
                kwargs["datefmt"] = DATE_FORMATS.get(Format(format), None)
            except ValueError:
                kwargs["format"] = format
                kwargs["datefmt"] = None
        else:
            kwargs["format"] = MSG_FORMATS[format]
            kwargs["datefmt"] = DATE_FORMATS.get(format, None)

    logging.basicConfig(**kwargs)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger
