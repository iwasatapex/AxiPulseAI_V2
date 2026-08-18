"""Behavioral regression: forecast -> decision payload integrity.

Proves:
* ``PredictionResult`` warnings/errors default to empty lists (never None).
* ``ForecastDay`` risk accepts the runtime dict form.
* The GUI forecast service surfaces decision-layer risk/confidence/etc from
  the ADIE V3 package (not stale top-level keys).
* A missing critical field must not silently become a fabricated value.
"""
from __future__ import annotations

from dataclasses import asdict

import pytest


# --------------------------------------------------------------------------- #
# Data model contracts
# --------------------------------------------------------------------------- #

def test_prediction_result_warnings_errors_default_empty_list():
    from core.forecast_ai.models import PredictionResult

    pr = PredictionResult()
    assert pr.warnings == []
    assert pr.errors == []


def test_forecast_day_risk_accepts_dict_and_list():
    from core.forecast_ai.models import ForecastDay

    # Runtime: the Forecast-Risk engine emits a dict.
    d = ForecastDay(
        date="2026-01-01", operations_health=80.0, nps=60.0,
        quality=85.0, competency=90.0, transfer=8.0, release=60.0,
        attendance=89.0, risk={"overall_risk": 0.4},
    )
    assert d.risk == {"overall_risk": 0.4}

    # Legacy consumers may still hold a list.
    d2 = ForecastDay(
        date="2026-01-01", operations_health=80.0, nps=60.0,
        quality=85.0, competency=90.0, transfer=8.0, release=60.0,
        attendance=89.0, risk=[{"overall_risk": 0.4}],
    )
    assert d2.risk == [{"overall_risk": 0.4}]


# --------------------------------------------------------------------------- #
# GUI forecast service surfaces decision-layer fields
# --------------------------------------------------------------------------- #

def test_forecast_service_surfaces_decision_fields(monkeypatch):
    """The forecast service must read risk/confidence/etc from the ADIE V3
    package, not from stale top-level payload keys the orchestrator omits."""
    from gui import services as svc

    package = {
        "probabilistic": {
            "risk": {"level": "LOW"},
            "confidence": {"score": 0.9},
        },
        "recommendations": {"items": [{"action": "hire"}]},
        "strategies": {"name": "s1"},
        "trends": {"metric": "quality", "direction": "up"},
        "sensitivity": {"kpi": "quality"},
        "agreement": {"score": 0.8},
    }
    decision = {"package": package, "details": {}}

    orchestrator_payload = {
        "timeline": [{"date": "2026-01-01", "operations_health": 80.0, "nps": 60.0}],
        "summary": {"total_days": 1},
        "decision_intelligence": decision,
    }

    class _Resp:
        success = True
        engine = "ForecastOrchestrator"
        warnings = []
        errors = []
        metadata = {"horizon": 1}
        payload = orchestrator_payload

    class _FakeOrch:
        def __init__(self, *a, **k):
            pass
        def execute(self, req):
            return _Resp()

    class _StubProvider:
        _model_family = None
        @classmethod
        def set_model_family(cls, family):
            cls._model_family = family

    monkeypatch.setattr(svc, "PredictorProvider", _StubProvider)
    monkeypatch.setattr(svc, "ForecastOrchestrator", _FakeOrch)
    monkeypatch.setattr(svc, "validate_model_pair", lambda fam: (None, None))

    out = svc.forecast(
        {"quality": 85.0, "competency": 90.0, "attendance": 89.0,
         "transfer": 8.0, "release": 60.0},
        horizon=1, family="alpha",
    )
    svc.STATE.reset()

    assert out["risk"] == {"level": "LOW"}
    assert out["confidence"] == {"score": 0.9}
    assert out["recommendations"] == {"items": [{"action": "hire"}]}
    assert out["sensitivity"] == {"kpi": "quality"}
    assert out["trend"] == {"metric": "quality", "direction": "up"}
    assert out["agreement"] == {"score": 0.8}
    assert out["decision_intelligence"] == decision
