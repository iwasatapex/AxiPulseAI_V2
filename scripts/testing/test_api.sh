#!/bin/bash

echo "🧪 Testing AxiPulseAI API..."

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Test 1: Root
echo -n "Testing root endpoint... "
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/)
if [ "$response" = "200" ]; then
    echo -e "${GREEN}✅ Passed${NC}"
else
    echo -e "${RED}❌ Failed (HTTP $response)${NC}"
fi

# Test 2: Health
echo -n "Testing health predictor... "
response=$(curl -s -X POST http://localhost:8000/api/v1/health/predict \
    -H "Content-Type: application/json" \
    -d '{"target_quality":85,"target_competency":80,"target_attendance":92,"target_release_rate":75,"target_transfer_rate":8,"actual_quality":82,"actual_competency":78,"actual_attendance":90,"actual_release_rate":72,"actual_transfer_rate":9,"total_calls_received":1850,"operational_intelligence_factor":0.85,"business_intelligence_factor":0.72,"member_intelligence_factor":0.68}' \
    -o /dev/null -w "%{http_code}")
if [ "$response" = "200" ]; then
    echo -e "${GREEN}✅ Passed${NC}"
else
    echo -e "${RED}❌ Failed (HTTP $response)${NC}"
fi

# Test 3: NPS
echo -n "Testing NPS predictor... "
response=$(curl -s -X POST http://localhost:8000/api/v1/nps/predict \
    -H "Content-Type: application/json" \
    -d '{"operational_health":76.4,"business_intelligence_factor":0.72,"member_intelligence_factor":0.68,"target_release_rate":75,"actual_release_rate":72,"release_gap":3,"release_delta":1.5,"total_calls_received":1850}' \
    -o /dev/null -w "%{http_code}")
if [ "$response" = "200" ]; then
    echo -e "${GREEN}✅ Passed${NC}"
else
    echo -e "${RED}❌ Failed (HTTP $response)${NC}"
fi

# Test 4: System
echo -n "Testing system status... "
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/v1/system/status)
if [ "$response" = "200" ]; then
    echo -e "${GREEN}✅ Passed${NC}"
else
    echo -e "${RED}❌ Failed (HTTP $response)${NC}"
fi

echo ""
echo "✅ All tests complete!"
