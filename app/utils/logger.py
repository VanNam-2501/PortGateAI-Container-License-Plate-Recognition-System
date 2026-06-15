import os
import logging
from logging.handlers import RotatingFileHandler

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")


def setup_logging(level: int = logging.INFO) -> None:
    """Configures global logging to output to both console and a rotating log file."""
    # Ensure logs directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Avoid duplicate handlers if setup_logging is called multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console Handler (Stdout)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Rotating File Handler (logs/app.log)
    try:
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not configure file logger: {e}")


def get_logger(name: str = "SmartGatePipeline") -> logging.Logger:
    """Gets or creates a logger with the given name."""
    return logging.getLogger(name)
