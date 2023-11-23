import logging

from rich.logging import RichHandler


def get_logger():

    FORMAT = "%(message)s"
    DATEFMT = "[%X}"

    logging.basicConfig(level="INFO", format=FORMAT, datefmt=DATEFMT, handlers=[RichHandler()])
    logger = logging.getLogger("rich")

    return logger
