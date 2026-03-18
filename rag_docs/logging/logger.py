import logging
from pathlib import Path

from rag_docs.config.settings import LOGS_PATH


def get_logger(name: str) -> logging.Logger:
  

    LOGS_PATH.mkdir(parents=True, exist_ok=True)

    log_file = LOGS_PATH / "rag_docs.log"

   
    # Example output:
    # 2024-01-15 10:30:45 [INFO] ingestion - ✅ Loaded paper.pdf
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(module)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture ALL levels

    # Only show INFO and above in terminal (not DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Save EVERYTHING (DEBUG and above) to file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # If logger already has handlers (called twice), don't add more
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger