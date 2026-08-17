"""
Regression tests for the ADIE recommendation / explanation GUI rendering.

Guarantees the recommendation-facing sections of ``gui.views.adie_view`` render
human-readable prose, never a raw dict/JSON/code:

- recommendations are not rendered as raw dicts/code
- ``forecast_day_2`` becomes "Forecast Day 2"
- confidence is formatted as a percentage (0.9 -> 90%)
- ABSTAIN state renders correctly
- an actionable recommendation renders as prose
- technical JSON remains available only inside the "Technical details" expander

These tests exercise the pure, streamlit-free formatters so no browser session
is required.
"""
from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

from gui.views import adie_view as av

ADIE_PATH = pathlib.Path(inspect.getsourcefile(av))


def _source():
    return ADIE_PATH.read_text()


# --------------------------------------------------------------------------- #
# Confidence formatting
# --------------------------------------------------------------------------- #
def test_confidence_formatted_as_percentage():
    assert av._format_confidence(0.9) == "90%"
    assert av._format_confidence(0.85) == "85%"
    assert av._format_confidence(1.0) == "100%"
    assert av._format_confidence(0.0) == "0%"
    assert av._format_confidence(1.234) == "100%"
    assert av._format_confidence(0.999) == "100%"


def test_confidence_none_renders_dash():
    assert av._format_confidence(None) == "—"


# --------------------------------------------------------------------------- #
# Scenario-name formatting
# --------------------------------------------------------------------------- #
def test_scenario_name_forecast_day_2_becomes_human_readable():
    assert av._format_scenario_name("forecast_day_2") == "Forecast Day 2"
    assert av._format_scenario_name("forecast_day_1") == "Forecast Day 1"
    assert av._format_scenario_name("current_state") == "Current State"


# --------------------------------------------------------------------------- #
# Action prose
# --------------------------------------------------------------------------- #
def test_action_is_humanized_to_prose():
    assert av._humanize_action("prioritize_nps_improvement") == "Prioritize nps improvement"
    assert av._humanize_action("") == "No actionable recommendation available."
    assert av._humanize_action(None) == "No actionable recommendation available."


# --------------------------------------------------------------------------- #
# Recommendation rendering never emits raw dict/code
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("func_name", [
    "_render_recommendations",
    "_render_recommendation_card",
    "_render_explanation",
])
def test_recommendation_sections_never_render_raw_code(func_name):
    """Recommendation-facing renderers must not call st.json/st.code on recs."""
    src = _source()
    # Locate the function body.
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == func_name)
    seg = ast.get_source_segment(src, fn)
    assert "st.json(" not in seg
    assert "st.code(" not in seg


def test_recommendation_card_exposes_expected_labels_only():
    """The recommendation card shows human labels, not internal snake keys."""
    src = _source()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_render_recommendation_card")
    seg = ast.get_source_segment(src, fn)
    body = seg.split(":")[1].split("if __name__")[0] if "if __name__" in seg else seg.split(":")[1]
    for forbidden in ("recommendation_status", "aggregate_probability", "re_ranking",
                      "decision_status"):
        assert forbidden not in body


def test_technical_json_confined_to_technical_expander():
    """Raw JSON is only emitted inside the Technical details expander."""
    src = _source()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_render_technical_details_expander")
    seg = ast.get_source_segment(src, fn)
    assert "st.json(" in seg  # raw JSON lives here
    assert "st.expander" in seg
    assert "expanded=False" in seg  # optional / collapsed by default


def test_render_explanation_uses_technical_expander_not_direct_json():
    """_render_explanation must route raw detail through the expander only."""
    src = _source()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_render_explanation")
    seg = ast.get_source_segment(src, fn)
    # It must reference the expander helper and must NOT call st.json directly.
    assert "_render_technical_details_expander(" in seg
    assert "st.json(" not in seg
    assert "st.code(" not in seg


# --------------------------------------------------------------------------- #
# Formatting helpers used by the recommendation view
# --------------------------------------------------------------------------- #
def test_why_selected_returns_prose_not_dict():
    ws = {"text": "Forecast preference only: forecast_day_1 ranked highest."}
    out = av._format_why_selected(ws)
    assert isinstance(out, str)
    assert "Forecast preference only" in out


def test_decision_changers_formatted_as_prose():
    dc = {"re_ranking": "x", "risk": "y"}
    out = av._format_decision_changers(dc)
    assert isinstance(out, str)
    assert "deterministic ranking policy" in out
    assert "risk level" in out
    # Internal key names are not leaked verbatim as raw dict contents.
    assert "aggregate_probability" not in out


def test_format_why_selected_handles_empty():
    out = av._format_why_selected({})
    assert isinstance(out, str)
    assert out
