"""Operational Prometheus metrics endpoint.

This endpoint is intentionally public and read-only, matching the health
endpoint architecture (see ``/health``). It exposes Prometheus metrics for
monitoring/observability and must be reachable without authentication.
Protected business endpoints retain their own authentication.
"""
from fastapi import APIRouter
from fastapi.responses import Response

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter()


@router.get("")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
