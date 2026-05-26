#!/usr/bin/env bash
# RAG System Boot Script (Linux/Mac)
set -euo pipefail

echo "========================================"
echo "  RAG System Boot Script"
echo "========================================"

# Check free RAM
FREE_MB=$(free -m 2>/dev/null | awk '/^Mem:/{print $7}' || echo "0")
FREE_GB=$(echo "scale=1; $FREE_MB / 1024" | bc 2>/dev/null || echo "unknown")
echo "Free RAM: ${FREE_GB} GB"

if [ "$FREE_MB" -lt 2048 ] 2>/dev/null; then
    echo "ERROR: Less than 2 GB free RAM. Aborting."
    exit 1
fi

# Set environment
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM"

# Activate venv
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_ACTIVATE="$PROJECT_DIR/.venv/bin/activate"

if [ -f "$VENV_ACTIVATE" ]; then
    echo "Activating virtual environment..."
    source "$VENV_ACTIVATE"
fi

# Launch server
echo "Starting RAG server..."
cd "$PROJECT_DIR"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
