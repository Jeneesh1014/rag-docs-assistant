#!/bin/bash
set -e

if [[ -f venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source venv/bin/activate
fi

echo "Starting API (8000) and Gradio (7860) via app/run.py"
exec python app/run.py
