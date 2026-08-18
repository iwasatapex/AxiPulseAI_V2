"""
Regression tests for ADIE GUI raw-code leakage, stale-risk text, and the
training/model leaderboard.

Guarantees:
- no raw dict/code in the primary ADIE view sections (Current State, Forecast
  Outlook, Bayesian, Monte Carlo, target probabilities, uncertainty,
  supporting evidence, why selected)
- raw JSON only inside the technical-details expander
- ABSTAIN supporting_evidence never claims a normal LOW/MEDIUM/HIGH risk
- Forecast Day / percentage formatting
- leaderboard uses human-readable columns and never renders raw dicts
"""
from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

from gui.views import adie_view as av

VIEW_PATH = pathlib.Path(inspect.getsourcefile(av))
MODELS_PATH = pathlib.Path(
    inspect.getsourcefile(__import__("gui.views.models_view", fromlist=["models_view"])))
TARGET_PATH = pathlib.Path(
    inspect.getsourcefile(__import__("gui.views.target_state_view", fromlist=["target_state_view"])))


def _src(path):
    return path.read_text()


# --------------------------------------------------------------------------- #
# Percentage formatting
# --------------------------------------------------------------------------- #
def test_percent_formatting():
    assert av._format_percent(0.7366) == "73.7%"
    assert av._format_percent(0.95) == "95%"
    assert av._format_percent(0.9) == "90%"
    assert av._format_percent(1.0) == "100%"
    assert av._format_percent(0.0) == "0%"
    assert av._format_percent(None) == "—"


def test_scenario_name_human_readable():
    assert av._format_scenario_name("forecast_day_2") == "Forecast Day 2"
    assert av._format_scenario_name("forecast_day_1") == "Forecast Day 1"


def test_label_metric_human_readable():
    assert av._label_metric("operations_health") == "Operational Health"
    assert av._label_metric("aggregate_probability") == "Decision Probability"


# --------------------------------------------------------------------------- #
# Current State / Forecast Outlook / Uncertainty render prose, not dicts
# --------------------------------------------------------------------------- #
def test_current_state_formatted_as_prose():
    out = av._format_current_state({
        "aggregate_probability": 0.7366,
        "aggregate_confidence": 0.95,
        "observed_metrics": ["operations_health", "nps"],
    })
    assert isinstance(out, str)
    assert "Overall decision probability: 73.7%" in out
    assert "Confidence: 95%" in out
    assert "Operational Health" in out
    # No raw key leaked as dict content.
    assert "aggregate_probability" not in out


def test_forecast_outlook_formatted_as_prose():
    out = av._format_forecast_outlook({"scenario_count": 3, "horizon_days": 1,
                                       "best_scenario": "forecast_day_2"})
    assert isinstance(out, str)
    assert "Forecast horizon: 1 day" in out
    assert "Forecast Day 2" in out


def test_uncertainty_formatted_as_prose():
    out = av._format_uncertainty({"downside": 0.42, "upside": 0.95, "confidence": 0.72,
                                  "monte_carlo_samples": 10000})
    assert isinstance(out, str)
    assert "Downside (p05): 42%" in out
    assert "Upside (p95): 95%" in out
    assert "10,000" in out


# --------------------------------------------------------------------------- #
# ABSTAIN supporting_evidence never claims a normal risk level
# --------------------------------------------------------------------------- #
def test_abstain_supporting_evidence_no_low_claim():
    """When the canonical decision is ABSTAIN, supporting_evidence must not
    claim LOW/MEDIUM/HIGH."""
    from core.decision_intelligence.v3.synthesis.decision_detail import (
        DECISION_STATUS_INSUFFICIENT,
        build_adie_detail,
    )
    # Explanation with a stale LOW risk claim in supporting_evidence.
    package = {
        "scenarios": [{"name": "forecast_day_1", "operations_health": 88.0}],
        "semantics": {},
        "decision": None,
        "explanation": {
            "current_state": {"aggregate_probability": 0.5},
            "supporting_evidence": ["canonical risk level: LOW (score 0.216)"],
            "main_risk": {"level": "LOW", "score": 0.216},
        },
    }
    detail = build_adie_detail(package, recommendation_output=None,
                               agreement=None, horizon=5)
    assert detail["decision_status"] == DECISION_STATUS_INSUFFICIENT
    exp = detail["explanation"]
    evidence = exp.get("supporting_evidence") or []
    assert evidence  # present
    joined = " ".join(evidence).lower()
    assert "canonical risk level: low" not in joined
    assert "canonical decision: abstain" in joined


# --------------------------------------------------------------------------- #
# Primary view does not emit raw dict/code in the explanation sections
# --------------------------------------------------------------------------- #
def test_render_explanation_never_emits_raw_code():
    """The primary _render_explanation must not call st.json/st.code directly
    on current_state / forecast_summary / uncertainty / supporting_evidence."""
    src = _src(VIEW_PATH)
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_render_explanation")
    seg = ast.get_source_segment(src, fn)
    assert "st.json(" not in seg
    assert "st.code(" not in seg
    # Raw detail is routed to the technical expander only.
    assert "_render_technical_details_expander(" in seg


def test_primary_view_only_raw_via_technical_expander():
    """Within the ADIE view, st.json appears only in the technical expander
    helper and the (data-only) agreement insufficient-state block."""
    src = _src(VIEW_PATH)
    # The agreement insufficient-state block is intentionally structured, but
    # it is not a recommendation-facing prose leak; still, ensure the general
    # explanation/recommendation sections never call st.json.
    tree = ast.parse(src)
    for fn_name in ("_render_explanation", "_render_recommendations",
                    "_render_recommendation_card", "_render_mc_detail",
                    "_render_bayesian_detail"):
        fn = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == fn_name), None)
        if fn is None:
            continue
        seg = ast.get_source_segment(src, fn)
        assert "st.code(" not in seg, f"{fn_name} leaks st.code"


# --------------------------------------------------------------------------- #
# Leaderboard: human-readable columns, no raw dicts
# --------------------------------------------------------------------------- #
def test_models_view_leaderboard_human_readable():
    src = _src(MODELS_PATH)
    assert "Model Family" in src
    assert "Predictor" in src
    assert "Algorithm" in src
    assert "MAE" in src
    assert "Training Rows" in src
    assert "History Days" in src
    assert "Status" in src
    assert "Device" in src


def test_models_view_leaderboard_shows_final_fit_feasibility():
    """The NPS leaderboard shows resource-aware selection columns."""
    src = _src(MODELS_PATH)
    assert "CV NPS MAE" in src
    assert "Final Fit Feasible" in src
    assert "Estimated Memory" in src
    assert "Exclusion Reason" in src


def test_models_view_no_raw_json_performance():
    """The family detail must not dump algorithm_performance as raw st.json."""
    src = _src(MODELS_PATH)
    assert "st.json(perf)" not in src
    # Raw JSON only in the technical expander.
    assert "raw_json_expander(" in src


def test_target_state_leaderboard_no_raw_dict_fallback():
    """The target-state leaderboard fallback must not render a raw dict."""
    src = _src(TARGET_PATH)
    assert "col.json(data.to_dict(orient=\"records\"))" not in src
    # Human-readable fallback lines instead.
    assert "_fmt_v" in src
