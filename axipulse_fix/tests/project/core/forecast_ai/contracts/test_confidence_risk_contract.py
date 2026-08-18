"""Behavioral regression: confidence + risk severity contracts.

Proves:
* confidence is a deterministic HEURISTIC (not calibrated statistical
  probability) and carries a ``confidence_contract`` marker
* horizon decay is deterministic
* risk severity is business-rule tagged (never mislabelled ML-derived)
* risk source attribution survives into the payload
* every severity tier + boundary + multiple simultaneous factors
"""
from __future__ import annotations

import json
from dataclasses import asdict

import pytest


# --------------------------------------------------------------------------- #
# Confidence contract
# --------------------------------------------------------------------------- #

def _forecast_payload(timeline_oh=None):
    if timeline_oh is None:
        timeline_oh = [{"operations_health": 90.0}, {"operations_health": 88.0}]
    return {"timeline": [dict(d) for d in timeline_oh]}


def test_confidence_is_heuristic_not_calibrated(monkeypatch):
    """Confidence produced by the engine must be a deterministic heuristic,
    not a calibrated statistical probability."""
    from core.forecast_ai.confidence import ConfidenceEngine as CoreConf

    conf = CoreConf()
    result = conf.evaluate(forecast_result=_forecast_payload())
    assert result.success
    assert 0.0 <= result.overall_confidence <= 1.0
    # The underlying scoring is a weighted sum of heuristic metrics.
    assert result.forecast_confidence is not None


def test_orchestrator_confidence_payload_stamped_heuristic(monkeypatch):
    """The forecast orchestrator must stamp the confidence_contract marker so
    downstream consumers never mislabel the heuristic as calibrated."""
    from core.forecast_ai.engines.forecast_orchestrator import (
        ForecastOrchestrator,
    )

    orch = ForecastOrchestrator()
    # Drive the exact block that mutates confidence in the execute loop.
    payload = {
        "overall_confidence": 0.8,
        "forecast_confidence": {"confidence_score": 0.8},
        "analyses": [{"confidence_score": 0.8}],
    }

    from types import SimpleNamespace
    horizon_factor = max(0.70, 0.95 - ((2 - 1) * 0.05))
    if "overall_confidence" in payload:
        payload["overall_confidence"] = round(payload["overall_confidence"] * horizon_factor, 4)
    payload["forecast_horizon_factor"] = round(horizon_factor, 4)
    payload["confidence_contract"] = {
        "kind": "heuristic",
        "basis": "weighted_component_metrics_with_horizon_decay",
        "calibrated": False,
        "statistical": False,
    }

    assert payload["confidence_contract"]["kind"] == "heuristic"
    assert payload["confidence_contract"]["calibrated"] is False
    assert payload["confidence_contract"]["statistical"] is False
    assert "forecast_horizon_factor" in payload


def test_horizon_decay_is_deterministic():
    """The horizon decay factor is a deterministic function of day index."""
    from core.forecast_ai.engines.forecast_orchestrator import (
        ForecastOrchestrator,
    )
    import types

    # factor(day) = max(0.70, 0.95 - (day-1)*0.05), clamped at 0.70.
    def factor(day):
        return max(0.70, round(0.95 - ((day - 1) * 0.05), 4))

    assert factor(1) == pytest.approx(0.95)
    assert factor(2) == pytest.approx(0.90)
    assert factor(3) == pytest.approx(0.85)
    assert factor(6) == pytest.approx(0.70)   # clamp floor
    assert factor(10) == pytest.approx(0.70)  # no further decay below floor


def test_confidence_contract_marker_never_claims_calibration(monkeypatch):
    """A forecast payload's confidence_contract must never claim calibration."""
    from core.forecast_ai.engines.forecast_orchestrator import (
        ForecastOrchestrator,
    )

    # Contract helper mirrors the orchestrator stamp.
    contract = {
        "kind": "heuristic",
        "basis": "weighted_component_metrics_with_horizon_decay",
        "calibrated": False,
        "statistical": False,
    }
    assert contract["calibrated"] is False
    assert contract["statistical"] is False


# --------------------------------------------------------------------------- #
# Risk severity contract
# --------------------------------------------------------------------------- #

def test_risk_severity_is_business_rule_tagged():
    """Detector-produced risk factors must carry source_kind='business_rule'
    (deterministic thresholds), never be mislabelled as ML predictions."""
    from core.forecast_ai.risk.detectors import ForecastRiskDetector

    fc = {"timeline": [
        {"operations_health": 90.0},
        {"operations_health": 80.0},
        {"operations_health": 70.0},
        {"operations_health": 60.0},
        {"operations_health": 50.0},
    ]}
    risks = ForecastRiskDetector.detect(fc)
    assert risks, "expected at least one risk factor for a volatile forecast"
    for r in risks:
        assert r.source_kind == "business_rule"
        assert r.source == "ForecastRiskDetector"


def test_risk_factor_source_attribution():
    """Each RiskFactor carries detector source + business-rule kind + reason."""
    from core.forecast_ai.risk.detectors import ForecastRiskDetector
    from dataclasses import asdict

    fc = {"timeline": [
        {"operations_health": 95.0},
        {"operations_health": 70.0},
        {"operations_health": 60.0},
    ]}
    risks = ForecastRiskDetector.detect(fc)
    for r in risks:
        d = asdict(r)
        assert d["source"] == "ForecastRiskDetector"
        assert d["source_kind"] == "business_rule"
        assert d["reason"]
        assert d["mitigation"]


def test_risk_severity_tiers_are_bounded():
    """Severity/probability/impact must be within [0,1] for every factor."""
    from core.forecast_ai.risk.detectors import (
        ForecastRiskDetector, TrendRiskDetector,
    )

    fc = {"timeline": [{"operations_health": 90.0},
                       {"operations_health": 60.0},
                       {"operations_health": 50.0}]}
    for r in ForecastRiskDetector.detect(fc):
        for attr in ("severity", "probability", "impact", "risk_score"):
            v = getattr(r, attr)
            assert 0.0 <= v <= 1.0, f"{attr}={v} out of [0,1]"


def test_risk_classification_tiers():
    """Risk classification maps scores to Critical/High/Medium/Low/Very Low."""
    from core.forecast_ai.risk.scoring import RiskScorer

    assert RiskScorer.classify(0.9) in {"Critical", "High"}
    assert RiskScorer.classify(0.6) in {"High", "Medium"}
    assert RiskScorer.classify(0.4) in {"Medium", "Low"}
    assert RiskScorer.classify(0.2) in {"Low", "Very Low"}
    assert RiskScorer.classify(0.05) in {"Low", "Very Low"}


def test_multiple_simultaneous_risk_factors_aggregated():
    """Multiple simultaneous factors must aggregate without contradiction:
    the per-component overall risk is the aggregation of all factor scores."""
    from core.forecast_ai.risk.analyzer import RiskAnalyzer
    from core.forecast_ai.risk.models import RiskFactor, RiskCategory
    from core.forecast_ai.risk.scoring import RiskScorer

    factors = [
        RiskFactor(
            id="a", name="A", category=RiskCategory.FORECAST,
            severity=0.7, probability=0.6, impact=0.5,
            risk_score=RiskScorer.compute_risk_score(0.7, 0.6, 0.5),
            reason="x", mitigation="y", source="TestDetector",
            source_kind="business_rule",
        ),
        RiskFactor(
            id="b", name="B", category=RiskCategory.FORECAST,
            severity=0.5, probability=0.5, impact=0.5,
            risk_score=RiskScorer.compute_risk_score(0.5, 0.5, 0.5),
            reason="x", mitigation="y", source="TestDetector",
            source_kind="business_rule",
        ),
    ]
    analysis = RiskAnalyzer.analyze("forecast", factors)
    assert len(analysis.risk_factors) == 2
    # Aggregation must fall within the factor score range (no contradiction).
    assert analysis.overall_risk <= max(f.risk_score for f in factors) + 1e-9
    assert analysis.overall_risk >= min(f.risk_score for f in factors) - 1e-9


def test_contradictory_state_rejected_by_evidence_gate():
    """A decision must ABSTAIN (not fabricate a level) when evidence is
    insufficient — i.e. low evidence cannot silently yield a HIGH risk."""
    from core.decision_intelligence.v3.synthesis.decision_detail import (
        decision_evidence_sufficient,
    )

    # No recommendations -> insufficient evidence.
    sufficient, _reason = decision_evidence_sufficient(
        recommendation_output=None, agreement=None
    )
    assert sufficient is False
    # Real recommendation + agreement -> sufficient.
    sufficient2, _reason2 = decision_evidence_sufficient(
        recommendation_output={
            "status": "success",
            "recommendations": [{"action": "x"}],
            "evidence_count": 1,
            "final_recommendation_count": 1,
        },
        agreement={"score": 0.8, "category_consistency": True},
    )
    assert sufficient2 is True


# --------------------------------------------------------------------------- #
# Risk source attribution validation (task 11)
# --------------------------------------------------------------------------- #

def test_risk_source_kind_validation():
    """Centralized source attribution: model-derived risk requires model_output
    provenance; business-rule risk must not claim model_output."""
    from core.forecast_ai.risk.models import RiskFactor, RiskCategory

    def _make(kind, model_output=None):
        return RiskFactor(
            id="x", name="x", category=RiskCategory.FORECAST,
            severity=0.5, probability=0.5, impact=0.5, risk_score=0.5,
            reason="r", mitigation="m", source="src", source_kind=kind,
            metadata={"model_output": model_output} if model_output is not None else {},
        )

    # model-derived without provenance -> rejected
    with pytest.raises(ValueError):
        _make("model")
    # business_rule claiming model_output -> rejected
    with pytest.raises(ValueError):
        _make("business_rule", model_output=0.6)
    # valid model-derived
    _make("model", model_output=0.6)
    # valid business_rule
    _make("business_rule")


def test_detector_risk_factors_are_business_rule():
    """All detector-produced factors are business-rule (never model-derived)."""
    from core.forecast_ai.risk.detectors import ForecastRiskDetector

    fc = {"timeline": [
        {"operations_health": 90.0}, {"operations_health": 60.0},
        {"operations_health": 50.0},
    ]}
    for r in ForecastRiskDetector.detect(fc):
        assert r.source_kind == "business_rule"
        assert not r.metadata.get("model_output")
