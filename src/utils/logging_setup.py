"""
Logging configuration.

Provides a consistent logging setup across all modules.
Uses stdlib logging — no external dependency needed.
File named logging_setup.py to avoid shadowing the stdlib logging module.
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "world_models",
    level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """Create and configure a logger.

    Args:
        name: Logger name (appears in log messages).
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to also write logs to a file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    log = setup_logger()
    log.info("Logger initialized successfully.")
    log.debug("This won't show at INFO level.")
    log.warning("This is a warning test.")
