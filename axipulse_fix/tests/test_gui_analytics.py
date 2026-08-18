"""Tests for the AxiPulseAI V2 GUI analytics layer.

These test the pure analysis functions (the analytics data contract) so they
can run headlessly without a browser. Representative canonical output
fixtures are used; no simulator mathematics is duplicated here.
"""
from __future__ import annotations

import pandas as pd
import pytest

from gui.analytics import common as a
from gui.analytics import adie, dashboard, forecast, prediction, reverse, target_state, training
from gui import contracts as ct


# =====================================================================
# KPI target calculations / met logic / transfer inverse semantics
# =====================================================================

def test_kpi_met_high_direction():
    assert ct.kpi_met("quality", 90.0) is True
    assert ct.kpi_met("quality", 80.0) is False
    assert ct.kpi_met("release", 50.0) is False  # target 60
    assert ct.kpi_met("release", 60.0) is True


def test_kpi_met_transfer_inverse():
    # Lower transfer is better.
    assert ct.kpi_met("transfer", 5.0) is True
    assert ct.kpi_met("transfer", 9.0) is True   # == target
    assert ct.kpi_met("transfer", 12.0) is False


def test_kpi_met_returns_none_when_no_target_or_bad_value():
    assert ct.kpi_met("nps", 50.0) is None          # no canonical target
    assert ct.kpi_met("operations_health", 90.0) is None
    assert ct.kpi_met("quality", "high") is None


def test_kpi_comparison_rows_respect_transfer():
    state = {"quality": 90.0, "transfer": 12.0, "release": 55.0}
    rows = {r["kpi"]: r for r in a.kpi_comparison_rows(state)}
    assert rows["Quality"]["met"] is True
    assert rows["Transfer Rate"]["met"] is False
    assert rows["Release Rate"]["met"] is False
    # Transfer gap is target - value (positive means above target -> worse).
    assert rows["Transfer Rate"]["gap"] == pytest.approx(9.0 - 12.0)


def test_day_kpi_met_requires_3_of_4():
    # quality, competency, release met; transfer not -> 3 of 4 -> met.
    day = {"quality": 90.0, "competency": 95.0, "release": 70.0, "transfer": 15.0}
    assert a.day_kpi_met(day) is True
    # Only 2 met -> not met.
    day2 = {"quality": 90.0, "competency": 95.0, "release": 40.0, "transfer": 15.0}
    assert a.day_kpi_met(day2) is False


# =====================================================================
# NPS normalization / negative NPS rendering
# =====================================================================

def test_nps_normalization_via_analytics_contracts():
    assert ct.normalize_nps_distribution({"score_5": 0.5, 6: 0.5}) == {5: 0.5, 6: 0.5}


def test_forecast_chart_negative_nps_axis():
    from gui import charts
    timeline = [
        {"operations_health": 80.0, "nps": -30.0},
        {"operations_health": 82.0, "nps": -15.0},
    ]
    fig = charts.forecast_timeline_chart(timeline, 2)
    assert fig is not None
    y2 = fig.layout.yaxis2.range
    assert (y2[0], y2[1]) == (ct.NPS_MIN, ct.NPS_MAX) or (y2[0], y2[1]) == (-100.0, 100.0)


# =====================================================================
# Missing optional fields must not crash
# =====================================================================

def test_analytics_tolerate_empty_result():
    assert prediction.confidence_info({})["available"] is False
    assert prediction.nps_analytics({})["nps"] is None
    # Diagnostics on an empty result flags missing output (a warning), but
    # must not crash.
    diag = prediction.prediction_diagnostics({}, {})
    assert isinstance(diag, list)
    assert any("no OH/NPS" in w for w in diag)
    assert forecast.kpi_trajectories([]) != {}
    assert all(v.get("available") is False for v in forecast.kpi_trajectories([]).values())
    assert forecast.target_attainment([])["pct_horizon_met"] is None
    assert adie.decision_summary({})["action"] == "No recommendation produced"
    assert adie.decision_drivers({}) == []
    assert target_state.gap_analysis({}) == []
    assert reverse.required_state({}) == []
    assert dashboard.kpi_overview({}) == []
    m = training.model_metrics_analytics({})
    assert set(m) == {"oh", "nps"}
    assert all(v.get("mae") is None and v.get("r2") is None for v in m.values())


# =====================================================================
# Training analytics
# =====================================================================

def _train_df():
    return pd.DataFrame({
        "actual_quality": [80.0, 85.0, 90.0, 95.0, 88.0],
        "actual_competency": [70.0, 75.0, 80.0, 85.0, 78.0],
        "promoters": [10, 12, 8, 15, 11],
        "passives": [20, 22, 18, 25, 21],
        "detractors": [5, 6, 4, 7, 5],
    })


def test_training_dataset_profile():
    p = a.dataset_profile(_train_df())
    assert p["rows"] == 5
    assert p["columns"] == 5
    assert p["n_numeric"] == 5
    assert p["missing_total"] == 0
    assert p["duplicate_rows"] == 0


def test_training_data_quality_flags_constant_and_missing():
    df = pd.DataFrame({"a": [1, 1, 1], "b": [1, None, 3]})
    q = a.data_quality_report(df)
    qa = {r["feature"]: r for r in q}
    assert qa["a"]["constant"] is True
    assert qa["b"]["missing_pct"] == pytest.approx(33.33, abs=0.01)


def test_training_target_analysis():
    d = a.describe_target(_train_df(), "actual_quality")
    assert d is not None
    assert d["min"] == 80.0 and d["max"] == 95.0
    assert d["mean"] == pytest.approx(sum([80, 85, 90, 95, 88]) / 5)


def test_training_correlations():
    top = a.top_correlations(_train_df(), "actual_quality")
    assert isinstance(top, list)
    names = {t["feature"] for t in top}
    assert "actual_competency" in names


def test_training_health_unknown_fit_without_r2():
    profile = {"rows": 100, "missing_total": 0, "duplicate_rows": 0}
    metrics = training.model_metrics_analytics({})
    h = training.training_health(profile, metrics)
    assert h["model_fit"]["level"] == "Unknown"


def test_training_health_uses_r2_when_present():
    profile = {"rows": 100, "missing_total": 0, "duplicate_rows": 0}
    metrics = {"oh": {"r2": 0.85, "mae": 0.2}, "nps": {"r2": 0.75, "mae": 0.3}}
    h = training.training_health(profile, metrics)
    assert h["model_fit"]["level"] == "Good"
    assert h["overall"] == "Good"


def test_training_model_metrics_extraction():
    result = {
        "oh_algorithm": "CatBoost",
        "oh_metrics": {"CatBoost": {"mae": 0.5, "rmse": 0.7, "r2": 0.8}},
        "nps_metrics": {"RandomForest": {"mae": 0.4, "r2": 0.75}},
    }
    m = training.model_metrics_analytics(result)
    assert m["oh"]["mae"] == pytest.approx(0.5)
    assert m["oh"]["r2"] == pytest.approx(0.8)
    assert m["nps"]["mae"] == pytest.approx(0.4)


# =====================================================================
# Prediction analytics
# =====================================================================

def _pred_result():
    return {
        "operational_health": 88.5,
        "nps": -12.5,
        "oh_confidence": 0.91,
        "oh_lower": 85.0, "oh_upper": 92.0,
        "bayesian_score_distribution": {"score_5": 0.2, "score_9": 0.8},
        "promoters": 0.6, "passives": 0.3, "detractors": 0.1,
        "active_family": "alpha",
    }


def test_prediction_input_summary_ranges():
    state = {"quality": 90.0, "transfer": 12.0, "release": 55.0, "operations_health": 95.0}
    rows = prediction.input_summary(state)
    by = {r["kpi"]: r for r in rows}
    assert by["Quality"]["lo"] == 60.0 and by["Quality"]["hi"] == 100.0
    assert by["Transfer Rate"]["hi"] == 20.0
    assert by["Release Rate"]["lo"] == 50.0
    assert by["Operational Health"]["hi"] == 100.0


def test_prediction_confidence_info():
    c = prediction.confidence_info(_pred_result())
    assert c["available"] is True
    assert c["oh_interval"] == (85.0, 92.0)


def test_prediction_nps_analytics_causal_order():
    n = prediction.nps_analytics(_pred_result())
    assert n["nps"] == pytest.approx(-12.5)
    assert n["distribution"] == {5: 0.2, 9: 0.8}
    assert n["survey_probabilities"]["promoters"] == pytest.approx(0.6)
    # Accurate wording: NPS is ML-predicted from the score distribution.
    assert "ML-predicted" in n["causal_order"]
    assert "Surveys" not in n["causal_order"]


def test_prediction_diagnostics_flags_misses():
    state = {"quality": 90.0, "release": 40.0}
    warns = prediction.prediction_diagnostics(_pred_result(), state)
    assert any("below target" in w for w in warns)


# =====================================================================
# Forecast analytics
# =====================================================================

def _timeline():
    # 3 days. Day 0 observed, days 1-2 predicted.
    return [
        {"operations_health": 80.0, "quality": 90.0, "competency": 95.0,
         "release": 70.0, "transfer": 8.0, "nps": -20.0},
        {"operations_health": 82.0, "quality": 88.0, "competency": 94.0,
         "release": 66.0, "transfer": 9.0, "nps": -15.0},
        {"operations_health": 85.0, "quality": 85.0, "competency": 93.0,
         "release": 62.0, "transfer": 9.0, "nps": -5.0},
    ]


def test_forecast_horizon_summary():
    fc = {"horizon": 3, "scenario": "baseline", "active_family": "alpha",
          "_timestamp": "2026-01-01T00:00:00", "timeline": _timeline()}
    hs = forecast.horizon_summary(fc)
    assert hs["horizon"] == 3 and hs["days"] == 3


def test_forecast_kpi_trajectories_ranges():
    traj = forecast.kpi_trajectories(_timeline())
    assert traj["operations_health"]["range"] == (0.0, 100.0)
    assert traj["nps"]["range"] == (-100.0, 100.0)
    assert traj["transfer"]["range"] == (0.0, 20.0)
    assert traj["quality"]["range"] == (60.0, 100.0)
    assert traj["operations_health"]["direction"] == "up"
    assert traj["nps"]["direction"] == "up"


def test_forecast_target_attainment():
    att = forecast.target_attainment(_timeline())
    assert att["total_days"] == 3
    assert att["met_days"] == 3
    assert att["pct_horizon_met"] == pytest.approx(100.0)


def test_forecast_trend_analytics():
    tr = forecast.trend_analytics(_timeline())
    assert tr["operations_health"]["available"] is True
    assert tr["operations_health"]["deltas"] == [2.0, 3.0]
    assert tr["nps"]["deltas"] == [5.0, 10.0]


def test_forecast_uncertainty_unavailable_by_default():
    unc = forecast.uncertainty({"timeline": _timeline()})
    assert unc["available"] is False


def test_forecast_risk_flags_detect_transfer_and_release_issues():
    bad = [
        {"operations_health": 80.0, "nps": 20.0, "transfer": 10.0, "release": 60.0},
        {"operations_health": 70.0, "nps": 15.0, "transfer": 25.0, "release": 40.0},
    ]
    flags = forecast.risk_flags(bad)
    joined = " | ".join(flags).lower()
    assert "transfer" in joined
    assert "release" in joined
    assert "declining operational health" in joined


def test_forecast_scenario_comparison():
    fc_a = {"scenario": "baseline", "timeline": _timeline()}
    fc_b = {"scenario": "training", "timeline": _timeline()}
    comp = forecast.scenario_comparison([fc_a, fc_b])
    assert len(comp) == 2
    assert {c["scenario"] for c in comp} == {"baseline", "training"}
    assert comp[0]["kpi_met_pct"] == pytest.approx(100.0)


# =====================================================================
# Scenario comparison (extended)
# =====================================================================

def _fc(sid, oh_final=80.0, nps_final=-5.0, transfer_final=9.0, release_final=62.0,
        quality_final=85.0, competency_final=93.0, horizon=2):
    return {
        "scenario": sid, "horizon": horizon, "active_family": "alpha",
        "success": True,
        "timeline": [
            {"operations_health": 78.0, "nps": -10.0, "quality": 90.0,
             "competency": 95.0, "release": 70.0, "transfer": 8.0},
            {"operations_health": oh_final, "nps": nps_final, "quality": quality_final,
             "competency": competency_final, "release": release_final, "transfer": transfer_final},
        ],
    }


def test_scenario_comparison_includes_finals_and_deltas():
    base = _fc("baseline", oh_final=80.0, nps_final=-5.0)
    alt = _fc("training", oh_final=85.0, nps_final=-15.0, transfer_final=12.0)
    comp = {c["scenario"]: c for c in forecast.scenario_comparison([base, alt])}
    assert comp["training"]["oh_delta"] == pytest.approx(5.0)
    assert comp["training"]["nps_delta"] == pytest.approx(-10.0)
    assert comp["training"]["transfer_final"] == pytest.approx(12.0)
    assert comp["baseline"]["oh_delta"] == pytest.approx(0.0)  # reference vs itself


def test_scenario_comparison_negative_nps_not_clipped():
    alt = _fc("training", nps_final=-25.0)
    comp = {c["scenario"]: c for c in forecast.scenario_comparison([_fc("baseline"), alt])}
    assert comp["training"]["nps_final"] == pytest.approx(-25.0)


def test_kpi_met_spec_95_105_thresholds():
    # Rule 2: quality/competency/release meet at >= 95% of target.
    assert ct.kpi_met("quality", 82.65) is True      # 0.95 * 87
    assert ct.kpi_met("quality", 82.64) is False
    assert ct.kpi_met("competency", 88.35) is True   # 0.95 * 93
    assert ct.kpi_met("competency", 88.0) is False
    assert ct.kpi_met("release", 57.0) is True       # 0.95 * 60
    assert ct.kpi_met("release", 56.9) is False
    # transfer meets at <= 105% of target (inverse).
    assert ct.kpi_met("transfer", 9.45) is True      # 1.05 * 9
    assert ct.kpi_met("transfer", 9.46) is False


def test_scenario_comparison_transfer_inverse_kpi_met():
    # Per-KPI: transfer uses inverse semantics with a 105%-of-target threshold.
    assert ct.kpi_met("transfer", 8.0) is True
    assert ct.kpi_met("transfer", 9.45) is True      # == 105% of target 9
    assert ct.kpi_met("transfer", 12.0) is False
    # Day-level: a single failed transfer does not fail a day when the other
    # 3 checked KPIs meet (>= 3 of 4 rule).
    good = _fc("baseline", transfer_final=8.0)
    bad = _fc("training", transfer_final=12.0)
    assert a.day_kpi_met(good["timeline"][1]) is True
    assert a.day_kpi_met(bad["timeline"][1]) is True  # 3 of 4 still met


def test_scenario_comparison_tolerates_missing_timeline():
    comp = forecast.scenario_comparison([{"scenario": "baseline"}, {"scenario": "training"}])
    assert len(comp) == 2
    assert comp[0]["oh_final"] is None
    assert comp[0]["kpi_met_pct"] is None


def test_scenario_comparison_malformed_output_tolerated():
    comp = forecast.scenario_comparison([
        {"scenario": "baseline", "timeline": None},
        {"scenario": "training", "timeline": [{"operations_health": "oops"}]},
    ])
    assert len(comp) == 2


def test_run_scenario_comparison_executes_each_enabled_once():
    calls = []

    def fake_fn(state, horizon, scenario=None, family=None):
        calls.append(scenario)
        return _fc(scenario)

    results, cache = forecast.run_scenario_comparison(
        {"quality": 87.0}, 3, "alpha", ["baseline", "training", "training", "baseline"],
        forecast_fn=fake_fn,
    )
    assert calls.count("baseline") == 1
    assert calls.count("training") == 1
    assert [r["scenario"] for r in results].count("baseline") == 1  # exactly one baseline


def test_run_scenario_comparison_does_not_rerun_cached():
    calls = []
    payload = _fc("training")

    def fake_fn(state, horizon, scenario=None, family=None):
        calls.append(scenario)
        return payload

    cache = {}
    results1, cache = forecast.run_scenario_comparison(
        {"quality": 87.0}, 3, "alpha", ["baseline", "training"], forecast_fn=fake_fn, cache=cache)
    n1 = len(calls)
    # Rerun with same inputs + reused cache -> no new calls.
    results2, cache = forecast.run_scenario_comparison(
        {"quality": 87.0}, 3, "alpha", ["baseline", "training"], forecast_fn=fake_fn, cache=cache)
    assert len(calls) == n1
    assert all(r["cached"] for r in results2)


def test_run_scenario_comparison_rejects_disabled_without_executing():
    from core.forecast_ai.scenarios.registry import ScenarioRegistry
    from core.forecast_ai.scenarios.models import Scenario

    ScenarioRegistry.register(Scenario(
        id="__disabled__", name="x", description="x", modifiers=[], enabled=False,
    ))
    calls = []

    def fake_fn(state, horizon, scenario=None, family=None):
        calls.append(scenario)
        return _fc(scenario)

    try:
        results, _ = forecast.run_scenario_comparison(
            {"quality": 87.0}, 3, "alpha", ["baseline", "__disabled__"], forecast_fn=fake_fn)
    finally:
        ScenarioRegistry.reset()
    by = {r["scenario"]: r for r in results}
    assert "__disabled__" not in calls          # never executed
    assert by["__disabled__"]["payload"] is None
    assert "disabled or unknown" in by["__disabled__"]["error"]


def test_run_scenario_comparison_is_session_pure():
    """The comparison must not touch gui.state (session isolation)."""
    from gui import state as gui_state
    before = set(gui_state._FALLBACK.keys())

    def fake_fn(state, horizon, scenario=None, family=None):
        return _fc(scenario)

    forecast.run_scenario_comparison(
        {"quality": 87.0}, 3, "alpha", ["baseline", "training"], forecast_fn=fake_fn)
    assert set(gui_state._FALLBACK.keys()) == before


# =====================================================================
# ADIE analytics
# =====================================================================

def _adie_result():
    return {
        "decision_intelligence": {
            "package": {
                "recommendations": [
                    {"action": "Increase training", "affected_kpi": "competency",
                     "direction": "up", "confidence": 0.87, "risk": "low",
                     "expected_effect": {"oh_gain": 2.0, "nps_lift": 1.0},
                     "evidence": ["a", "b"]}
                ],
                "risks": [{"kpi": "transfer", "severity": "medium"}],
            },
            "details": {"contributions": {"competency": 0.6, "transfer": -0.3}},
        },
        "forecast": {"scenario": "baseline"},
    }


def test_adie_decision_summary():
    s = adie.decision_summary(_adie_result())
    assert s["action"] == "Increase training"
    assert s["affected_kpi"] == "competency"
    assert s["confidence"] == pytest.approx(0.87)


def test_adie_decision_drivers_ranked():
    d = adie.decision_drivers(_adie_result())
    assert d[0]["factor"] == "competency"
    assert d[0]["direction"] == "positive"


def test_adie_risk_and_recommendation_quality():
    assert len(adie.risk_analysis(_adie_result())) == 1
    recs = adie.recommendation_quality(_adie_result())
    assert recs[0]["action"] == "Increase training"
    assert recs[0]["evidence_count"] == 2


def test_adie_explainability_deterministic():
    lines = adie.explainability_text(_adie_result())
    assert any("Increase training" in l for l in lines)


# =====================================================================
# Target State analytics
# =====================================================================

def _target_state_result():
    return {
        "targets": {"operational_health": 90.0, "release": 60.0, "transfer": 9.0},
        "recommended_state": {"quality": 92.0, "release": 62.0, "transfer": 8.0},
        "consensus": {"oh": 90.2, "release": 62.0, "transfer": 8.0},
        "distance": 0.03,
        "leaderboards": {"OH": [{"model": "CatBoost"}], "NPS": [{"model": "RF"}]},
    }


def test_target_gap_analysis():
    gaps = target_state.gap_analysis(_target_state_result())
    by = {g["target"]: g for g in gaps}
    assert by["Operational Health"]["desired"] == pytest.approx(90.0)
    assert by["Operational Health"]["achieved"] == pytest.approx(90.2)
    assert by["Operational Health"]["delta"] == pytest.approx(0.2)


def test_target_feasibility_feasible():
    f = target_state.feasibility(_target_state_result())
    assert f["feasible"] is True
    assert f["conflicts"] == []


def test_target_feasibility_conflict_on_out_of_range():
    r = _target_state_result()
    r["recommended_state"] = {"transfer": 25.0}
    f = target_state.feasibility(r)
    assert f["feasible"] is False
    assert f["conflicts"]


# =====================================================================
# Reverse optimizer analytics
# =====================================================================

def _reverse_result():
    return {
        "metric": "OH", "target": 90.0, "predicted": 90.1, "distance": 0.05,
        "found": True,
        "recommended_state": {"quality": 92.0, "release": 62.0, "transfer": 8.0, "attendance": 90.0},
        "consensus": {"oh": 90.1},
    }


def test_reverse_target_vs_predicted():
    t = reverse.target_vs_predicted(_reverse_result())
    assert t["delta"] == pytest.approx(0.1)


def test_reverse_required_state_and_constraints():
    req = reverse.required_state(_reverse_result())
    keys = {r["kpi"] for r in req}
    assert "Transfer Rate" in keys
    assert reverse.constraint_analysis(_reverse_result()) == []


def test_reverse_feasibility_classification():
    assert reverse.feasibility_classification(_reverse_result())["class"] == "Feasible"


def test_reverse_feasibility_infeasible_when_no_solution():
    r = _reverse_result()
    r["found"] = False
    r["recommended_state"] = {}
    assert reverse.feasibility_classification(r)["class"] == "Infeasible"


# =====================================================================
# Dashboard analytics
# =====================================================================

def test_dashboard_model_inventory():
    models = [{"family": "alpha", "oh": {"model_name": "X"}, "nps": {"model_name": "Y"}, "active": True}]
    inv = dashboard.model_inventory(models)
    assert inv[0]["status"] == "Ready"
    assert inv[0]["active"] is True


def test_dashboard_health_breakdown():
    health = {"status": "Degraded",
              "checks": {"models": {"status": "Ready", "available_families": ["alpha"]},
                         "active_model": {"status": "Degraded", "detail": "invalid"}}}
    hb = dashboard.health_breakdown(health)
    assert len(hb) == 3
    assert hb[1]["status"] == "Degraded"


def test_dashboard_kpi_overview_uses_final_forecast_day():
    last_fc = {"timeline": _timeline()}
    rows = dashboard.kpi_overview(last_fc)
    by = {r["kpi"]: r for r in rows}
    assert "Transfer Rate" in by
    assert by["Quality"]["latest"] == pytest.approx(85.0)


# =====================================================================
# Session isolation (analytics must not create global state)
# =====================================================================

def test_analytics_functions_are_pure():
    """Analytics pure functions must not mutate any session state."""
    from gui import state as gui_state
    before = set(gui_state._FALLBACK.keys())
    _train_df()
    a.dataset_profile(_train_df())
    prediction.confidence_info(_pred_result())
    forecast.kpi_trajectories(_timeline())
    target_state.gap_analysis(_target_state_result())
    reverse.feasibility_classification(_reverse_result())
    adie.decision_summary(_adie_result())
    after = set(gui_state._FALLBACK.keys())
    assert after == before, "analytics must not touch session state"
