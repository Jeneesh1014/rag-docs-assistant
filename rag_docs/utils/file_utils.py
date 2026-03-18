# rag_docs/utils/file_utils.py
# small helper functions used in multiple places across the project
# rather than copy-pasting the same logic everywhere, we import from here

import shutil
from pathlib import Path

from rag_docs.logging.logger import get_logger

logger = get_logger(__name__)


def check_folder_exists(folder_path: Path, folder_name: str = "Folder") -> bool:
    # returns True if folder exists, False if not
    if not folder_path.exists():
        logger.error(f"{folder_name} not found: {folder_path}")
        logger.error("Make sure you're running from the project root folder")
        return False
    return True


def delete_folder(folder_path: Path) -> bool:
    # deletes a folder and everything inside it
    # returns True if it worked, False if something went wrong
    try:
        shutil.rmtree(folder_path)
        logger.info(f"Deleted folder: {folder_path}")
        return True
    except Exception as e:
        logger.error(f"Couldn't delete {folder_path}: {e}")
        return False


def folder_is_empty(folder_path: Path) -> bool:
    # treat a missing folder the same as an empty one
    if not folder_path.exists():
        return True
    return not any(folder_path.iterdir())


def ask_user_yes_no(question: str) -> bool:
    # asks the user a yes/no question in the terminal and waits for input
    response = input(f"\n{question} (yes/no): ")
    return response.strip().lower() == "yes"