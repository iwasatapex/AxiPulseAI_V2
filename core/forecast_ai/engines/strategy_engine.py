"""
StrategyEngine – ForecastAI engine for operational strategy generation.
Consumes RecommendationResult, does NOT run optimization or predictions.
"""
import datetime
import logging
from typing import Dict, Any, Optional
from dataclasses import asdict

from ..base_engine import ForecastAIEngine
from ..models import ForecastRequest, ForecastResponse
from ..strategy import StrategyEngine as StrategyCore
from ..recommendations import RecommendationResult

logger = logging.getLogger(__name__)

class StrategyEngine(ForecastAIEngine):
    def __init__(self, strategy_core: Optional[StrategyCore] = None):
        self.strategy_core = strategy_core or StrategyCore()

    def execute(self, request: ForecastRequest) -> ForecastResponse:
        if request.parameters is None:
            return self._error_response("Missing parameters")

        # Expect recommendation_result to be passed in
        rec_result_data = request.parameters.get('recommendation_result')
        if rec_result_data is None:
            return self._error_response("Missing 'recommendation_result' in parameters")

        # Reconstruct RecommendationResult
        try:
            if hasattr(rec_result_data, 'success'):
                rec_result = rec_result_data
            else:
                # Try to parse as dict
                from ..recommendations import RecommendationResult as RecResult
                from ..recommendations import (
                    Recommendation as Rec,
                    Category as RecCategory,
                    Priority as RecPriority,
                    Difficulty as RecDifficulty,
                )

                raw_recommendations = rec_result_data.get('recommendations', [])

                def _reconstruct_rec(item):
                    if isinstance(item, Rec):
                        return item
                    if not isinstance(item, dict):
                        return item
                    cat = item.get('category')
                    pri = item.get('priority')
                    diff = item.get('difficulty')
                    try:
                        cat_enum = (
                            RecCategory(cat) if not isinstance(cat, RecCategory) else cat
                        )
                    except Exception:
                        cat_enum = RecCategory.GENERAL
                    try:
                        pri_enum = (
                            RecPriority(pri) if not isinstance(pri, RecPriority) else pri
                        )
                    except Exception:
                        pri_enum = RecPriority.MEDIUM
                    try:
                        diff_enum = (
                            RecDifficulty(diff) if not isinstance(diff, RecDifficulty) else diff
                        )
                    except Exception:
                        diff_enum = RecDifficulty.MEDIUM
                    return Rec(
                        id=item.get('id', ''),
                        title=item.get('title', ''),
                        description=item.get('description', ''),
                        category=cat_enum,
                        priority=pri_enum,
                        difficulty=diff_enum,
                        estimated_operations_health_gain=item.get(
                            'estimated_operations_health_gain'
                        ),
                        estimated_nps_gain=item.get('estimated_nps_gain'),
                        estimated_disruption=item.get('estimated_disruption', 0.0),
                        confidence=item.get('confidence', 0.5),
                        actions=item.get('actions', []),
                        reasoning=item.get('reasoning', ''),
                        optimization_score=item.get('optimization_score', 0.0),
                        metadata=item.get('metadata', {}),
                    )

                rec_result = RecResult(
                    success=rec_result_data.get('success', False),
                    recommendations=[
                        _reconstruct_rec(r) for r in raw_recommendations
                    ],
                    warnings=rec_result_data.get('warnings', []),
                    errors=rec_result_data.get('errors', []),
                    metadata=rec_result_data.get('metadata', {})
                )
        except Exception as e:
            return self._error_response(f"Failed to parse recommendation_result: {str(e)}")

        try:
            strategy_result = self.strategy_core.generate(rec_result)
        except Exception as e:
            logger.exception("Strategy generation failed")
            return self._error_response(f"Strategy error: {str(e)}")

        payload = {
            "recommendation_result": asdict(rec_result) if hasattr(rec_result, 'success') else rec_result,
            "strategies": {
                "success": strategy_result.success,
                "strategies": [asdict(s) for s in strategy_result.strategies],
                "best_strategy": asdict(strategy_result.best_strategy) if strategy_result.best_strategy else None,
                "warnings": strategy_result.warnings,
                "errors": strategy_result.errors,
                "metadata": strategy_result.metadata
            }
        }

        return ForecastResponse(
            success=strategy_result.success,
            operation="strategy",
            engine="StrategyEngine",
            timestamp=datetime.datetime.now().isoformat(),
            warnings=strategy_result.warnings,
            errors=strategy_result.errors,
            metadata={"phase": "9", "mode": "pure_transformation"},
            payload=payload
        )

    def _error_response(self, message: str) -> ForecastResponse:
        return ForecastResponse(
            success=False,
            operation="strategy",
            engine="StrategyEngine",
            timestamp=datetime.datetime.now().isoformat(),
            warnings=[],
            errors=[message],
            metadata={},
            payload=None
        )

# Module-level compatibility surface.
# Delegates to the existing implementation; no logic changed.
execute = StrategyEngine.execute
