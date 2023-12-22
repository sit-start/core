import logging

logging.basicConfig()


def get_logger(name: str) -> logging.Logger:
    """Returns a default-configured logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
