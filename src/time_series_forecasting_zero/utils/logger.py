"""Logging configuration for the forecasting framework."""

import os
import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: str = "./outputs/logs/app.log",
    console_output: bool = True,
    file_output: bool = True
) -> None:
    """
    Configure the logger with console and file handlers.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        console_output: Whether to log to console
        file_output: Whether to log to file
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler
    if console_output:
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True
        )
    
    # Add file handler
    if file_output:
        # Create log directory if it doesn't exist
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            enqueue=True
        )
    
    logger.info(f"Logger initialized with level={log_level}")
    if file_output:
        logger.info(f"Log file: {log_file}")


# Initialize logger with default settings when module is imported
if not os.getenv("LOGGER_INITIALIZED"):
    setup_logger()
    os.environ["LOGGER_INITIALIZED"] = "true"
