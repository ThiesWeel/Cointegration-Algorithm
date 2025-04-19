import logging
import os

def get_logger(name: str, log_path: str = None, to_terminal: bool = False):
    """
    Create and configure a logger.

    Args:
        name (str): Name of the logger.
        log_path (str): Path to the log file (optional).
        to_terminal (bool): Whether to log to the terminal.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Avoid duplicate logs in the root logger

    # Clear existing handlers to prevent duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler (if log_path is provided)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(file_handler)

    # Terminal (stream) handler
    if to_terminal:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

    return logger