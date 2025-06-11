import os
import logging
from logging.handlers import RotatingFileHandler

LOG_DIR = 'logs'
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5

def setup_logger(name, log_file, level=logging.INFO):
    """Set up a production-ready logger with rotating files and console output."""
    
    # Ensure log directory exists
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers to the same logger
    if not logger.hasHandlers():
        # Log format (includes filename and line number)
        log_format = '%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s'
        formatter = logging.Formatter(log_format)

        # Create rotating file handler (10 MB max, keep 5 backups)
        file_handler = RotatingFileHandler(
            os.path.join(LOG_DIR, log_file), 
            maxBytes=MAX_LOG_SIZE, 
            backupCount=BACKUP_COUNT
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)

        # Create console handler for real-time logging
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)

        # Attach handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
