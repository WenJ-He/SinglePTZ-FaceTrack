"""Logging module: console + daily-rotating file."""

import logging
import os
from logging.handlers import TimedRotatingFileHandler


def setup_logger(name: str = "app", level: str = "INFO",
                 log_file: str = "logs/app.log") -> logging.Logger:
    """Configure root logger with console + daily-rotating file output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (daily rotation, keep 30 days)
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = TimedRotatingFileHandler(
            log_file, when="midnight", interval=1, backupCount=30,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
