import pytest
import logging
from sitstart.logging import get_logger, Format


def test_get_logger(capsys):
    logger = get_logger()
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.INFO

    assert get_logger(name="TestLogger").name == "TestLogger"
    assert get_logger(level=logging.DEBUG).level == logging.DEBUG

    message = "Testing 1-2-3..."
    get_logger(format=Format.BARE, force=True).info(message)
    get_logger(format="simple", force=True).info(message)
    get_logger(format="glog", force=True).warning(message)

    # use capsys after `force=True``, since that resets the root logger
    [bare_log, simple_log, glog_log] = capsys.readouterr().err.strip().splitlines()
    assert bare_log == message
    assert len(simple_log) > len(bare_log) and message in simple_log
    assert len(glog_log) > len(simple_log) and message in glog_log

    with pytest.raises(ValueError):
        logger = get_logger(format="invalid_format", force=True)
