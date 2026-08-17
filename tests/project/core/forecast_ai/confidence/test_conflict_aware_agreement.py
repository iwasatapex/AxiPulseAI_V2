"""Regression tests for the recommendation-agreement conflict detector.

Implements the TODO at core/forecast_ai/confidence/metrics.py: recommendation
agreement now uses the existing ConflictDetector + consistency metrics instead
of a placeholder.
"""

from core.forecast_ai.confidence.metrics import ConfidenceMetrics
from core.forecast_ai.recommendations.conflicts import ConflictDetector
from core.forecast_ai.recommendations.models import (
    Recommendation,
    Category,
    Priority,
    Difficulty,
)


def _rec(title, desc, cat):
    return Recommendation(
        id="r",
        title=title,
        description=desc,
        category=cat,
        priority=Priority.MEDIUM,
        difficulty=Difficulty.MEDIUM,
        actions=[],
    )


def test_conflict_detector_finds_opposite_directions():
    """Two recommendations driving the same KPI in opposite directions conflict."""
    r1 = _rec("Raise quality", "Increase quality by improving processes", Category.QUALITY)
    r2 = _rec("Cut quality spend", "Decrease quality-related cost", Category.OPERATIONS)
    conflicts = ConflictDetector.detect_conflicts([r1, r2])
    assert len(conflicts) == 1
    assert "quality" in conflicts[0][2]


def test_conflict_detector_no_conflict_same_direction():
    """Same-direction recommendations are not conflicts."""
    r1 = _rec("Improve quality", "Increase quality by improving processes", Category.QUALITY)
    r2 = _rec("More quality", "Improve quality further", Category.QUALITY)
    assert ConflictDetector.detect_conflicts([r1, r2]) == []


def test_recommendation_agreement_reduced_by_conflict():
    """Agreement must be lower when a conflict is present."""
    r1 = _rec("Raise quality", "Increase quality by improving processes", Category.QUALITY)
    r2 = _rec("Cut quality spend", "Decrease quality-related cost", Category.OPERATIONS)
    r3 = _rec("More quality", "Improve quality further", Category.QUALITY)

    with_conflict = ConfidenceMetrics.recommendation_agreement([r1, r2])
    without_conflict = ConfidenceMetrics.recommendation_agreement([r1, r3])
    assert with_conflict < without_conflict


def test_recommendation_agreement_single_category_high():
    """Agreement is 1.0 for a single-category, conflict-free recommendation set."""
    r1 = _rec("Improve quality", "Increase quality", Category.QUALITY)
    r2 = _rec("More quality", "Improve quality further", Category.QUALITY)
    assert ConfidenceMetrics.recommendation_agreement([r1, r2]) == 1.0


def test_recommendation_agreement_empty_zero():
    """Empty recommendation list yields zero agreement."""
    assert ConfidenceMetrics.recommendation_agreement([]) == 0.0
