"""
Logging utility for MedAgent.
Provides unified console and file logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from src.config.paths import LOGS_DIR


def get_logger(
    name: str = "MedAgent",
    log_file: Optional[str] = "medagent.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Creates and configures a standardized logger.

    Args:
        name: Logger name (module or component name)
        log_file: Optional log filename within LOGS_DIR
        level: Logging level (default INFO)

    Returns:
        Configured logging.Logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger was already created
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_path = LOGS_DIR / log_file
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
