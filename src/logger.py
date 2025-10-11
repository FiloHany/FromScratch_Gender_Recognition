import logging
import os
from logging.handlers import RotatingFileHandler
from .config import Config

def setup_logger(name: str = 'gender_detection') -> logging.Logger:
    """
    Set up a logger with console and file handlers following best practices.
    Uses RotatingFileHandler for log rotation to handle scalability.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()  # Avoid duplicate handlers

    logger.setLevel(getattr(logging, Config.LOG_LEVEL))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler with rotation
    if not os.path.exists(os.path.dirname(Config.LOG_FILE)):
        os.makedirs(os.path.dirname(Config.LOG_FILE))
    file_handler = RotatingFileHandler(
        Config.LOG_FILE, maxBytes=10*1024*1024, backupCount=5  # 10MB per file, 5 backups
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger
