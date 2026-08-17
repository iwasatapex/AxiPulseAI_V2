"""Optimizer timeout enforcement (Phase 5)."""

import time

import pytest

from core.forecast_ai.optimization.search import DeterministicHillClimb
from core.forecast_ai.optimization.models import TargetGoal
from core.forecast_ai.engines.recommendation_engine import RecommendationEngine
from core.forecast_ai.models import ForecastRequest as FR
from core.forecast_ai.optimization import ReverseOptimizer
from core.forecast_ai.prediction.service import PredictionService


def test_timeout_aborts_within_bounded_margin():
    state = __import__("core.forecast_ai.state.models", fromlist=["OperationalState"]).OperationalState(
        quality=85.0, competency=78.0, transfer=9.0, release=60.0, attendance=90.0
    )
    target = TargetGoal(target_operations_health=90.0)

    def slow_eval(state):
        time.sleep(0.2)
        return 82.0, 70.0

    searcher = DeterministicHillClimb()
    start = time.time()
    searcher.iterate(state, slow_eval, target, [], max_iterations=25, timeout_seconds=0.1)
    wall = time.time() - start
    assert searcher.timed_out is True
    # 0.1s budget with 0.2s/eval -> must abort after ~1 evaluation, far below
    # the 25-iteration budget it would otherwise burn.
    assert wall < 3.0


class _SlowOH:
    def predict(self, state):
        time.sleep(0.25)
        return 82.0


class _SlowNPS:
    def predict(self, state):
        time.sleep(0.25)
        return {"nps": 70.0}


def test_recommendation_timeout_is_structured_not_silent():
    service = PredictionService(oh_predictor=_SlowOH(), nps_predictor=_SlowNPS())
    engine = RecommendationEngine(
        optimizer_core=ReverseOptimizer(prediction_service=service)
    )
    resp = engine.execute(FR(operation="recommend", parameters={
        "state": {"quality": 85.0, "competency": 78.0, "transfer": 9.0,
                  "release": 60.0, "attendance": 90.0, "operations_health": 82.0},
        "target_oh": 95.0,
        "max_iterations": 25,
        "timeout_seconds": 0.1,
    }))
    assert resp.success is False
    assert resp.metadata.get("reason") == "optimization_timeout"
