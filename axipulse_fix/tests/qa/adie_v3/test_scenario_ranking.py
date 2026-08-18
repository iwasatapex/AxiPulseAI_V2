"""Scenario ranking (Phase 2) regression tests.

CASE A: day 1 materially worse than day 3 -> day 3 can outrank day 1.
CASE B: days genuinely identical -> deterministic tie preserved.
CASE C: higher predicted KPI but materially higher risk -> the policy handles
        the tradeoff consistently (safety component).
"""

from core.decision_intelligence.v3.scenario.scoring import (
    compute_scenario_score,
    rank_scenarios,
)
from core.decision_intelligence.v3.integration.probabilistic_decision import (
    ProbabilisticDecisionService,
)


def _day(name, oh, confidence, risk_severity, delta_oh):
    return {
        "name": name,
        "operations_health": oh,
        "confidence": confidence,
        "risk_severity": risk_severity,
        "delta_oh": delta_oh,
        "_predicted": True,
    }


def test_case_a_day3_can_outrank_day1():
    days = [
        _day("forecast_day_1", oh=72.0, confidence=0.5, risk_severity=0.7, delta_oh=-2.0),
        _day("forecast_day_2", oh=80.0, confidence=0.7, risk_severity=0.4, delta_oh=1.0),
        _day("forecast_day_3", oh=91.0, confidence=0.9, risk_severity=0.1, delta_oh=3.0),
    ]
    ranked = rank_scenarios([dict(d) for d in days])
    assert ranked[0]["name"] == "forecast_day_3"
    assert ranked[0]["rank"] == 1
    # Deterministic: re-ranking yields the same order.
    again = rank_scenarios([dict(d) for d in days])
    assert [s["name"] for s in again] == [s["name"] for s in ranked]


def test_case_b_identical_days_preserve_tie():
    a = _day("forecast_day_1", oh=82.0, confidence=0.8, risk_severity=0.3, delta_oh=0.0)
    b = _day("forecast_day_2", oh=82.0, confidence=0.8, risk_severity=0.3, delta_oh=0.0)
    ranked = rank_scenarios([dict(a), dict(b)])
    # Identical evidence -> identical scores; input order preserved (stable),
    # and the later equal-scored scenario is flagged as a tie.
    assert ranked[0]["score"] == ranked[1]["score"]
    assert ranked[0]["name"] == "forecast_day_1"
    assert ranked[1]["name"] == "forecast_day_2"
    assert ranked[1]["tie"] is True


def test_case_c_higher_kpi_but_higher_risk_tradeoff():
    # Day 3 has better OH but much worse risk; the safety component tempers it.
    day_low_risk = _day("forecast_day_2", oh=85.0, confidence=0.8, risk_severity=0.1, delta_oh=1.0)
    day_high_risk = _day("forecast_day_3", oh=93.0, confidence=0.5, risk_severity=0.95, delta_oh=1.0)
    ranked = rank_scenarios([dict(day_low_risk), dict(day_high_risk)])
    # The low-risk day outranks the higher-KPI/high-risk day given the weights.
    assert ranked[0]["name"] == "forecast_day_2"
    scores = {s["name"]: s["score"] for s in ranked}
    assert scores["forecast_day_2"] > scores["forecast_day_3"]


def test_no_evidence_scenario_gets_default_score_and_stable_order():
    s1 = {"name": "a"}
    s2 = {"name": "b"}
    ranked = rank_scenarios([dict(s1), dict(s2)])
    assert ranked[0]["score"] == 0.0
    assert ranked[0]["name"] == "a"
    assert ranked[1]["score"] == 0.0
    assert ranked[1]["tie"] is True


def test_scenario_ranking_through_service_differentiates_days():
    service = ProbabilisticDecisionService()
    days = [
        _day("forecast_day_1", oh=70.0, confidence=0.4, risk_severity=0.8, delta_oh=-3.0),
        _day("forecast_day_2", oh=88.0, confidence=0.9, risk_severity=0.2, delta_oh=4.0),
    ]
    result = service.analyze(
        scenarios=[dict(d) for d in days],
        observations=[1, 1, 0, 1, 1],
        baseline=0.8,
        samples=2000,
    )
    assert result.scenarios[0]["name"] == "forecast_day_2"
    assert result.scenarios[0]["rank"] == 1
    # Different days -> different scores (no longer degenerate).
    scores = {s["name"]: s["score"] for s in result.scenarios}
    assert scores["forecast_day_2"] > scores["forecast_day_1"]
