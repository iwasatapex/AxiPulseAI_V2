#!/bin/bash

set -e

URL=${1:-http://localhost:8000/health}

STATUS=$(curl -s -o /dev/null -w "%{http_code}" $URL)


if [ "$STATUS" = "200" ]; then
    echo "HEALTH PASS"
    exit 0
else
    echo "HEALTH FAIL: $STATUS"
    exit 1
fi
