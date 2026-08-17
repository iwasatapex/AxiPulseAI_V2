#!/usr/bin/env bash
# Launch the AxiPulseAI V2 Streamlit GUI.
# Run from the V2 project root.
set -euo pipefail
cd "$(dirname "$0")/.."

# Prefer the reproducible project venv (`.venv`, then legacy `venv`),
# falling back to the system interpreter only as a last resort.
PYTHON=.venv/bin/python
if [ ! -x "$PYTHON" ]; then
  PYTHON=venv/bin/python
fi
if [ ! -x "$PYTHON" ]; then
  PYTHON=python3
fi

echo "Starting AxiPulseAI V2 GUI at http://localhost:8501"
exec "$PYTHON" -m streamlit run gui/app.py --server.headless true "$@"
