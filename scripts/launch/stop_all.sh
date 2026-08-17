#!/bin/bash

echo "🛑 Stopping AxiPulseAI services..."

# Kill processes
pkill -f "uvicorn api.main:app" 2>/dev/null
pkill -f "streamlit run" 2>/dev/null

# Kill by port
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:8501 | xargs kill -9 2>/dev/null

echo "✅ All services stopped"
