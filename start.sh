#!/bin/bash

source venv/bin/activate

echo "Starting FastAPI on port 8000..."
uvicorn api:app --port 8000 &
API_PID=$!

echo "Waiting for API to be ready..."
sleep 10

echo "Starting Gradio UI on port 7860..."
python app.py &
GRADIO_PID=$!

echo ""
echo "Both servers are running."
echo "  Gradio UI  → http://localhost:7860"
echo "  FastAPI    → http://localhost:8000"
echo "  API docs   → http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both."

trap "kill $API_PID $GRADIO_PID 2>/dev/null; exit" INT

wait