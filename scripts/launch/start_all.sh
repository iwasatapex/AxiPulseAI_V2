#!/bin/bash

echo "🚀 Starting AxiPulseAI Complete System..."

# Start API in background
echo "📡 Starting API..."
./start_api.sh > logs/api.log 2>&1 &
API_PID=$!
echo "✅ API started (PID: $API_PID)"

# Wait for API
sleep 3

# Start Dashboard
echo "📊 Starting Dashboard..."
./start_dashboard.sh > logs/dashboard.log 2>&1 &
DASH_PID=$!
echo "✅ Dashboard started (PID: $DASH_PID)"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ AxiPulseAI System Running!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 API:        http://localhost:8000"
echo "📚 API Docs:   http://localhost:8000/docs"
echo "📊 Dashboard:  http://localhost:8501"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for Ctrl+C
wait
