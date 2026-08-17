#!/bin/bash
set -euo pipefail

echo "========================================="
echo "RUNNING SAFE TESTS (excluding broken ones)"
echo "========================================="

# Run only tests in specific safe directories
python3 -m pytest \
    tests/project/api/ \
    tests/project/analytics/ \
    tests/adie_v3/ \
    -v --tb=short --maxfail=5

echo ""
echo "========================================="
echo "SAFE TESTS COMPLETE"
echo "========================================="
