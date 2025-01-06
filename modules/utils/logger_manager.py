# modules/utils/logger_manager.py
# -*- coding: utf-8 -*-
"""
Logger Manager.

Sets up a logger based on a specified verbosity level.

@author: Tony-Luna
"""

import sys
import logging

def get_logger(verbosity: int) -> logging.Logger:
    """
    Sets up a logger for console output.

    Args:
        verbosity (int): 0=WARNING, 1=INFO, 2=DEBUG

    Returns:
        logging.Logger: Configured logger.
    """
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logger = logging.getLogger("SoccerVideoAnalysis")
    logger.setLevel(level)

    # Avoid duplicate handlers in case multiple calls
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
