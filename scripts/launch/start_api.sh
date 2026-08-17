#!/bin/bash

echo "🚀 Starting AxiPulseAI API..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found${NC}"
    exit 1
fi

# Install dependencies if needed
echo -e "${YELLOW}📦 Checking dependencies...${NC}"
pip install -q fastapi uvicorn[standard] pydantic pandas numpy scikit-learn joblib psutil 2>/dev/null

# Create directories
mkdir -p logs data models

# Check models
if [ ! -f "models/operation_health_model.pkl" ]; then
    echo -e "${YELLOW}⚠️  Health model not found. Using mock predictions.${NC}"
fi

if [ ! -f "models/nps_predictor_model.pkl" ]; then
    echo -e "${YELLOW}⚠️  NPS model not found. Using mock predictions.${NC}"
fi

# Start API
echo -e "${GREEN}✅ Starting API on http://localhost:8000${NC}"
echo -e "${GREEN}📚 Docs: http://localhost:8000/docs${NC}"
echo -e "${GREEN}📊 Dashboard: http://localhost:8501${NC}"

uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload --log-level info
