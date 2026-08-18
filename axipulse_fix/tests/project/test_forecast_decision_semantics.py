"""
Regression tests for Forecast/Decision output semantics.

Guarantees:
- ``expected_score`` is the mean 0..10 survey score (NEVER the -100..100 NPS).
- ``score_p05/p95`` are 0..10 score quantiles.
- ``nps_p05/p95`` are -100..100 NPS quantiles computed by Monte Carlo.
- Displayed NPS matches the distribution (promoters - detractors)*100.
- A flat point forecast with horizon confidence decay is labelled explicitly.
- Agreement/consistency returns an explicit insufficient-evidence state when
  evidence is missing (never an apparently confident block of —/0).
"""
import math

import numpy as np
import pytest

from core.decision_intelligence.v3.bayesian import inference as bayes
from core.decision_intelligence.v3.synthesis.decision_detail import build_adie_detail
from core.forecast_ai.engines import forecast_orchestrator


def _nps_distribution(center=8):
    weights = {s: math.exp(-0.5 * (s - center) ** 2) for s in range(0, 11)}
    total = sum(weights.values())
    return {f"score_{s}": w / total for s, w in weights.items()}


def _scenarios(n=4):
    scenarios = []
    for i in range(n):
        oh = 90.0
        dist = _nps_distribution(center=8)
        nps = bayes.expected_nps_business(dist)  # consistent with distribution
        scenarios.append({
            "name": f"forecast_day_{i}",
            "_predicted": True,
            "operations_health": oh,
            "nps": nps,
            "confidence": max(0.9 - i * 0.05, 0.1),  # declining -> confidence decay
            "risk_severity": 0.1,
            "expected_score": bayes.expected_nps_from_distribution(dist),
            "score_p05": 6.0,
            "score_p95": 10.0,
            "nps_p05": 60.0,
            "nps_p95": 100.0,
            "bayesian_score_distribution": dist,
        })
    return scenarios


def _package():
    return {"scenarios": _scenarios(), "semantics": {}, "decision": None}


# --------------------------------------------------------------------------- #
# 1/6. expected_score vs NPS
# --------------------------------------------------------------------------- #
def test_expected_score_is_0_10_and_never_nps():
    detail = build_adie_detail(_package(), horizon=4)
    first = detail["forecast_summary"]["per_day_table"][0]
    assert "expected_score" in first
    assert "expected_nps" not in first
    # expected_score is a 0..10 survey score.
    assert 0.0 <= first["expected_score"] <= 10.0
    # nps (the point forecast) is on the -100..100 scale and distinct.
    assert abs(first["nps"]) <= 100.0


def test_expected_score_matches_distribution_mean():
    dist = _nps_distribution(center=9)
    expected = bayes.expected_nps_from_distribution(dist)
    # The 0..10 mean of a distribution peaked at 9 is ~9 (never ~90+).
    assert 0.0 <= expected <= 10.0
    assert expected > 8.0


# --------------------------------------------------------------------------- #
# 2/6. score_p05/p95 vs nps_p05/p95
# --------------------------------------------------------------------------- #
def test_score_and_nps_quantiles_are_distinct_scales():
    detail = build_adie_detail(_package(), horizon=4)
    first = detail["forecast_summary"]["per_day_table"][0]
    assert "score_p05" in first and "score_p95" in first
    assert "nps_p05" in first and "nps_p95" in first
    # score quantiles are 0..10.
    assert 0.0 <= first["score_p05"] <= 10.0
    assert 0.0 <= first["score_p95"] <= 10.0
    # nps quantiles are -100..100.
    assert -100.0 <= first["nps_p05"] <= 100.0
    assert -100.0 <= first["nps_p95"] <= 100.0
    # They are different numbers on different scales.
    assert first["score_p05"] != first["nps_p05"]


def test_nps_monte_carlo_percentiles_are_nps_scale():
    dist = _nps_distribution(center=9)  # high promoter mass -> high NPS
    lo, hi = bayes.nps_monte_carlo_percentiles(dist)
    assert lo is not None and hi is not None
    assert -100.0 <= lo <= hi <= 100.0
    # Score quantiles of the same distribution are 0..10.
    slo, shi = bayes.nps_score_percentiles(dist)
    assert 0.0 <= slo <= shi <= 10.0


# --------------------------------------------------------------------------- #
# 5. Independent NPS validation
# --------------------------------------------------------------------------- #
def test_displayed_nps_matches_distribution():
    detail = build_adie_detail(_package(), horizon=4)
    for entry in detail["forecast_summary"]["per_day_table"]:
        dist = entry["nps_distribution"]
        p = np.array([dist[f"score_{i}"] for i in range(11)])
        detractors = p[0:7].sum()
        promoters = p[9:11].sum()
        expected_business = (promoters - detractors) * 100.0
        # The displayed nps point forecast matches the distribution-derived NPS.
        assert entry["nps"] == pytest.approx(expected_business, abs=1.0)


def test_nps_validation_consistent_with_expected_nps_business():
    dist = _nps_distribution(center=9)
    p = np.array([dist[f"score_{i}"] for i in range(11)])
    manual = (p[9:11].sum() - p[0:7].sum()) * 100.0
    assert bayes.expected_nps_business(dist) == pytest.approx(manual, abs=1e-6)


# --------------------------------------------------------------------------- #
# 3. Point forecast + confidence decay explicit
# --------------------------------------------------------------------------- #
def test_flat_point_forecast_labelled_explicitly():
    detail = build_adie_detail(_package(), horizon=4)
    character = detail["forecast_summary"]["character"]
    assert character["flat_oh"] is True
    assert character["flat_nps"] is True
    assert character["horizon_confidence_decay"] is True
    assert "point" in character["note"].lower()


def test_varying_forecast_not_marked_flat():
    scenarios = _scenarios()
    for i, s in enumerate(scenarios):
        s["nps"] = 80.0 - i * 2.0  # varying NPS
    detail = build_adie_detail({"scenarios": scenarios, "semantics": {}, "decision": None}, horizon=4)
    assert detail["forecast_summary"]["character"]["flat_nps"] is False


# --------------------------------------------------------------------------- #
# 4. Agreement insufficient evidence
# --------------------------------------------------------------------------- #
def test_agreement_insufficient_evidence_when_missing():
    detail = build_adie_detail(_package(), agreement=None, horizon=4)
    ag = detail["agreement"]
    assert ag["available"] is False
    assert ag["status"] == "insufficient_evidence"
    # Never an apparently confident 0/None block.
    assert ag.get("score") is None
    assert "conflict_count" not in ag


def test_agreement_available_when_evidence_present():
    detail = build_adie_detail(
        _package(),
        agreement={"score": 0.8, "category_consistency": 1.0, "conflicts": []},
        horizon=4,
    )
    ag = detail["agreement"]
    assert ag["available"] is True
    assert ag["status"] == "available"
    assert ag["score"] == pytest.approx(0.8)
    assert ag["conflict_count"] == 0


# --------------------------------------------------------------------------- #
# 7. Trained model invocation unchanged / no heuristic substitution
# --------------------------------------------------------------------------- #
def test_model_invocation_path_unchanged():
    # The forecast still calls the prediction service (not a heuristic) to build
    # each day's OH/NPS. Verify the orchestrator uses self.service.predict.
    import inspect
    src = inspect.getsource(forecast_orchestrator.ForecastOrchestrator.execute)
    assert "self.service.predict(pred_req)" in src
