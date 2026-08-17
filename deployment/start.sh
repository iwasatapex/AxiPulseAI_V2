#!/bin/bash

set -e

echo "Starting AxiPulseAI"

uvicorn api.main:app \
--host 0.0.0.0 \
--port 8000
