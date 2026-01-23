# src/utils/logging_utils.py
"""
Tiny helper that gives you a ready-to-use logger which writes both to
stdout *and* to a .log file.
"""
from __future__ import annotations
import logging, sys
from pathlib import Path


def make_logger(log_path: str | Path,
                level: int = logging.INFO,
                name: str = "res") -> logging.Logger:
    """
    Parameters
    ----------
    log_path : where the *.log* file will be saved (parent dirs auto-created)
    level    : logging level (DEBUG, INFO, …)
    name     : name of the logger (module-global singleton pattern)
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:                     # already initialised
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    # file
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)

    # console
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(level)
    logger.addHandler(sh)

    logger.debug(f"Logger initialised → {log_path.absolute()}")
    return logger
