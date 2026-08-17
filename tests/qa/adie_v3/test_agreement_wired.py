"""Conflict-aware recommendation agreement (Phase 10 / W6)."""

from core.forecast_ai.recommendations.conflicts import ConflictDetector
from core.forecast_ai.recommendations.models import (
    Recommendation,
    Category,
    Priority,
    Difficulty,
)


def _rec(title, desc, cat=Category.QUALITY, kpi=None, direction=None, mag=None):
    return Recommendation(
        id="r", title=title, description=desc, category=cat,
        priority=Priority.MEDIUM, difficulty=Difficulty.MEDIUM,
        actions=[], target_kpi=kpi, direction=direction, magnitude=mag,
    )


def test_structured_fields_detect_genuine_conflict():
    a = _rec("Improve quality", "", kpi="quality", direction="increase", mag=2.0)
    b = _rec("Cut quality", "", kpi="quality", direction="decrease", mag=1.0)
    conflicts = ConflictDetector.detect_conflicts([a, b])
    assert len(conflicts) == 1
    assert "quality" in conflicts[0][2]


def test_no_conflict_same_direction_structured():
    a = _rec("Improve quality", "", kpi="quality", direction="increase", mag=2.0)
    b = _rec("More quality", "", kpi="quality", direction="improve", mag=3.0)
    assert ConflictDetector.detect_conflicts([a, b]) == []


def test_negation_not_treated_as_direction():
    # "do not increase" should not count as an increase direction.
    a = _rec("Hold quality", "Do not increase quality spending", kpi=None, direction=None)
    b = _rec("Improve quality", "Increase quality", kpi="quality", direction="increase", mag=1.0)
    # 'a' has no structured fields; keyword fallback must ignore the negated
    # "increase" in 'a', so no conflict.
    assert ConflictDetector.detect_conflicts([a, b]) == []


def test_keyword_fallback_still_detects_conflict():
    # Both lack structured fields; keyword fallback finds opposite directions.
    a = _rec("Raise quality", "Increase quality by improving processes")
    b = _rec("Cut quality", "Decrease quality-related cost", cat=Category.OPERATIONS)
    conflicts = ConflictDetector.detect_conflicts([a, b])
    assert len(conflicts) == 1
    assert "quality" in conflicts[0][2]


def test_empty_no_conflicts():
    assert ConflictDetector.detect_conflicts([]) == []
