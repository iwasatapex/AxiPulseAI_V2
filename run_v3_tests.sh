#!/bin/bash
set -euo pipefail

echo "========================================="
echo "RUNNING ADIE V3 TESTS"
echo "========================================="

# Run only V3 tests
echo "1. Running V3 engine tests..."
python3 -m pytest tests/adie_v3/ -v --tb=short

echo ""
echo "2. Testing V3 import..."
python3 -c "from core.adie.v3 import DecisionEngine; print('✅ V3 import PASS')"

echo ""
echo "3. Testing V3 API routes..."
python3 -c "from api.routes import adie_v3_routes; print('✅ V3 routes import PASS')"

echo ""
echo "4. Testing V2 compatibility..."
python3 -c "from core.decision_intelligence import ADIE; adie = ADIE(); print('✅ V2 import PASS')"

echo ""
echo "5. Testing app import..."
python3 -c "from api.main import app; print('✅ App import PASS')"

echo ""
echo "========================================="
echo "ALL V3 TESTS PASSED"
echo "========================================="
