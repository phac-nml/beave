"""Generic initializer for the logger utilites.

Python loggers are federates meaning the follow a tree structre, in order
to make logging simple, while supporting a file log to the output directory
we use a simple filter instead of a complicated configuration setup for the program.

It is worth noting that filters are much slower so if we add log messages in a hot
loop program performance will suffer.

The file handling logger is added at run time in-order to place the final log
file in the output directory, rather than in the working directory. This is why
a filter is used instead of a global configuration.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGING_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

SHARED_STREAM_HANDLER: logging.StreamHandler = logging.StreamHandler(sys.stderr)

logging.basicConfig(
    datefmt="%Y-%m-%d %H:%M:%S",
    format=LOGGING_FORMAT,
    handlers=[SHARED_STREAM_HANDLER],
    level=logging.DEBUG,
)


class DebugFilter(logging.Filter):
    """Do not show debug filters."""

    def filter(self, record) -> bool:
        """Do not show debug level messages."""
        return record.levelno >= logging.INFO


def init_logger(module_name: str) -> logging.Logger:
    """Initialize logger for a provided module."""
    logger: logging.Logger = logging.getLogger(module_name)
    return logger


def add_file_logger(output_directory: Path) -> None:
    """Attach a file handle to the root logger."""
    root_logger: logging.Logger = logging.getLogger()
    output_log: Path = output_directory / "beave.log"
    handler: RotatingFileHandler = RotatingFileHandler(str(output_log), backupCount=10, delay=True)
    if output_log.is_file():
        handler.doRollover()  # make new log file for each new run
    handler.setFormatter(logging.Formatter(LOGGING_FORMAT))
    root_logger.addHandler(handler)
