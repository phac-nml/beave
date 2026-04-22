"""Generic initializer for the logger utilites."""

import logging
import sys


def init_logger(module_name: str) -> logging.Logger:
    """Initialize logger for a provided module."""
    logger = logging.getLogger(module_name)
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logger
