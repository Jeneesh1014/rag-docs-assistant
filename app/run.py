"""
Run FastAPI (port 8000) and Gradio (port 7860) in one process.
Waits until /health responds before opening the UI.
"""

import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import uvicorn
from dotenv import load_dotenv

from rag_docs.logging.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


def _serve_api():
    from app.api import app

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


def _wait_for_health(timeout_s: float = 120.0) -> bool:
    url = "http://127.0.0.1:8000/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.4)
    return False


def main():
    thread = threading.Thread(target=_serve_api, daemon=True)
    thread.start()

    if not _wait_for_health():
        logger.error("API did not become healthy in time; exiting")
        sys.exit(1)

    from app.ui import demo

    logger.info("FastAPI: http://127.0.0.1:8000  (docs at /docs)")
    logger.info("Gradio: http://127.0.0.1:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
