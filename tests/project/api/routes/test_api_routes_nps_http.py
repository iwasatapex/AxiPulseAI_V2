"""HTTP-level tests for POST /api/v1/nps/predict using the real NPS engine.

These run only in the intended API environment (FastAPI TestClient needs
``httpx``; the API database layer needs ``sqlalchemy``). They are skipped
automatically elsewhere, which also proves the NPS prediction path itself
never requires SQLAlchemy.
"""

import pytest

pytest.importorskip("httpx")
pytest.importorskip("sqlalchemy")

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.routes.nps_routes as nps_routes
from api.services.nps_service import NPSService

API_KEY = "dev-key-change-me"

VALID = {
    "operational_health": 95.0,
    "target_quality": 87.0,
    "actual_quality": 87.0,
    "target_competency": 93.0,
    "actual_competency": 93.0,
    "target_attendance": 90.0,
    "actual_attendance": 90.0,
    "target_release_rate": 60.0,
    "actual_release_rate": 60.0,
    "target_transfer_rate": 9.0,
    "actual_transfer_rate": 9.0,
    "total_calls_received": 2000,
}


def _headers():
    return {"X-API-Key": API_KEY}


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(nps_routes, "_service", None)
    app = FastAPI()
    app.include_router(nps_routes.router)
    return TestClient(app)


def test_http_predict_valid_returns_real_output(client):
    resp = client.post("/predict", json=VALID, headers=_headers())
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "success"
    data = body["data"]
    assert -100.0 <= data["nps"] <= 100.0
    assert "score_counts" in data
    assert "bayesian_score_distribution" in data
    assert "distribution" not in data
    assert "ensemble_details" not in data


def test_http_predict_invalid_returns_422(client):
    resp = client.post("/predict", json={}, headers=_headers())
    assert resp.status_code == 422

    resp = client.post(
        "/predict",
        json={**VALID, "operational_health": 999.0},
        headers=_headers(),
    )
    assert resp.status_code == 422


def test_http_predict_requires_api_key(client):
    resp = client.post("/predict", json=VALID)
    assert resp.status_code == 401


def test_http_predict_model_unavailable_returns_503(client, monkeypatch):
    monkeypatch.setattr(
        nps_routes,
        "_service",
        NPSService(model_path="models/__missing_nps_model__.pkl"),
    )
    resp = client.post("/predict", json=VALID, headers=_headers())
    assert resp.status_code == 503
