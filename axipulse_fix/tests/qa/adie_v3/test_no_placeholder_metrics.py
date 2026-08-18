"""No fabricated placeholder metrics (Phase 11 / W7)."""

from core.forecast_ai.recommendations.engine import RecommendationEngine
from core.forecast_ai.optimization.models import (
    OptimizationResult,
    OptimizationSolution,
)
from core.forecast_ai.recommendations.models import RecommendationResult
from core.forecast_ai.strategy.engine import StrategyEngine as StrategyCore
from core.forecast_ai.recommendations import (
    Recommendation,
    Category,
    Priority,
    Difficulty,
)


def _solution():
    return OptimizationSolution(
        predicted_operations_health=88.0,
        predicted_nps=72.0,
        state_changes={"quality": 5.0},
        applied_scenarios=[],
        optimization_score=0.5,
        distance_to_target=1.0,
        iterations_used=5,
        constraints_satisfied=True,
        state={"quality": 85.0},
    )


def test_recommendations_have_no_fabricated_gains_or_0_7_confidence():
    result = OptimizationResult(
        success=True, solutions=[_solution()], best_solution=_solution(),
        warnings=[], errors=[], metadata={},
    )
    rec_result = RecommendationEngine().generate(result)
    assert rec_result.success
    for rec in rec_result.recommendations:
        # Gains are None (explicitly assumptions), never fabricated numbers.
        assert rec.estimated_operations_health_gain is None
        assert rec.estimated_nps_gain is None
        # Confidence is DERIVED (0.9 here from distance 1.0), not hard-coded 0.7.
        assert rec.confidence != 0.7
        assert 0.0 <= rec.confidence <= 1.0
        # Structured conflict fields populated from optimizer deltas.
        assert rec.target_kpi is not None
        assert rec.direction in ("increase", "decrease")
        assert rec.magnitude is not None


def test_strategy_has_no_hardcoded_80_70_0_7():
    recs = RecommendationResult(
        success=True,
        recommendations=[
            Recommendation(
                id="rec-1", title="Improve Quality", description="Increase QA",
                category=Category.QUALITY, priority=Priority.HIGH,
                difficulty=Difficulty.MEDIUM, confidence=0.9,
                metadata={"predicted_operations_health": 88.0, "predicted_nps": 72.0},
            )
        ],
    )
    result = StrategyCore().generate(recs)
    assert result.success
    for strat in result.strategies:
        # Derived from the real recommendation predictions, or None — never 80/70.
        assert strat.estimated_operations_health is None or strat.estimated_operations_health == 88.0
        assert strat.estimated_nps is None or strat.estimated_nps == 72.0
        assert strat.confidence != 0.7
        assert 0.0 <= strat.confidence <= 1.0
        # Assumptions explicitly labelled in metadata.
        assert "assumptions" in strat.metadata
