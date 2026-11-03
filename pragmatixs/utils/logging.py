import os
import sys

from loguru import logger

from pragmatixs.utils.fs import get_root_dir


def setup_logging(log_dir: str | None = None) -> None:
    if log_dir is None:
        root_dir = get_root_dir()
        log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    os.environ["LOG_DIR"] = log_dir


def setup_logger(log_dir: str = None, filename: str = None, level: str = "DEBUG"):
    if os.getenv("LOG_DIR", None) is None:
        setup_logging(log_dir=log_dir)
    log_dir = os.getenv("LOG_DIR")

    logfile = os.path.join(log_dir, filename)
    if os.path.exists(logfile):
        os.remove(logfile)

    logger.remove()
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:MMM-DD HH:mm:ss}</green> | <level>{level}</level> |"
            " <cyan>{file.name}</cyan>:<cyan>{line}</cyan> | {message}"
        ),
        level=level,
        colorize=True,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    logger.add(
        logfile,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file.name}:{line} | {message}",
        level=level,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    return logger
