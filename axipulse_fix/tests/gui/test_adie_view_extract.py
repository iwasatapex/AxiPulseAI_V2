"""
Focused tests for the ADIE view extraction helper.

Validates that ``extract_adie_display`` correctly resolves
(probabilistic, details) under all payload shapes without
a browser session.
"""
from __future__ import annotations

import pytest

from gui.views.adie_view import extract_adie_display


def test_extract_via_di_details():
    """Priority 1: decision_intelligence.details + probabilistic."""
    result = {
        "decision_intelligence": {
            "details": {"recommendations": [{"rank": 1}]},
            "probabilistic": {"probability": 0.7},
        },
    }
    prob, det = extract_adie_display(result)
    assert prob["probability"] == 0.7
    assert det["recommendations"][0]["rank"] == 1


def test_extract_via_di_no_probabilistic_falls_to_decision():
    """When di has details but no probabilistic, fall back to decision.probabilistic."""
    result = {
        "decision_intelligence": {
            "details": {"recommendations": []},
        },
        "decision": {
            "probabilistic": {"probability": 0.6, "confidence": 0.8},
        },
    }
    prob, det = extract_adie_display(result)
    assert prob["probability"] == 0.6
    assert prob["confidence"] == 0.8


def test_extract_via_decision_details():
    """Priority 2: decision.details with decision.probabilistic."""
    result = {
        "decision_intelligence": {},
        "decision": {
            "details": {"forecast_summary": {"oh_range": {"min": 80}}},
            "probabilistic": {"probability": 0.65},
        },
    }
    prob, det = extract_adie_display(result)
    assert prob["probability"] == 0.65
    assert det["forecast_summary"]["oh_range"]["min"] == 80


def test_extract_via_di_probabilistic_only():
    """Priority 3: di has probabilistic but no details at any level."""
    result = {
        "decision_intelligence": {
            "probabilistic": {"probability": 0.66, "risk": "LOW"},
        },
    }
    prob, det = extract_adie_display(result)
    assert prob["probability"] == 0.66
    assert prob["risk"] == "LOW"
    assert det == {}


def test_extract_fallback_probe():
    """Fallback — raw decision or di.package becomes probabilistic."""
    result = {
        "decision": {
            "recommendation": "improve",
            "risk": "MEDIUM",
        },
    }
    prob, det = extract_adie_display(result)
    assert prob["recommendation"] == "improve"
    assert det == {}


def test_extract_empty_result():
    """Empty result -> empty prob and det."""
    prob, det = extract_adie_display({})
    assert prob == {}
    assert det == {}


def test_extract_no_decision_key():
    """Result without any decision key -> empty."""
    prob, det = extract_adie_display({"some_other_key": 1})
    assert prob == {}
    assert det == {}


def test_extract_di_package_fallback():
    """When di contains package dict (no details/probabilistic), package becomes prob."""
    result = {
        "decision_intelligence": {
            "package": {"recommendation": "hold"},
        },
    }
    prob, det = extract_adie_display(result)
    assert prob["recommendation"] == "hold"
    assert det == {}


def test_extract_with_details_and_di_package_preserves_details():
    """Full realistic payload: di has details and probabilistic."""
    result = {
        "success": True,
        "decision": {
            "probabilistic": {"probability": 0.72},
            "details": {"forecast_summary": {"oh_range": {"min": 85}}},
        },
        "decision_intelligence": {
            "package": {"probabilistic": {"probability": 0.72}},
            "probabilistic": {"probability": 0.72},
            "details": {"recommendations": [], "forecast_summary": {"oh_range": {"min": 85}}},
        },
    }
    prob, det = extract_adie_display(result)
    assert prob["probability"] == 0.72
    assert det["forecast_summary"]["oh_range"]["min"] == 85
    assert det["recommendations"] == []