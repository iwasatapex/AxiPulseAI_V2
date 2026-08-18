"""Focused tests for the ReverseOptimizer reliability fix (V2).

Covers the required behaviors: already-satisfied target, feasible OH
improvement, feasible combined target, infeasible target (honest failure),
best-effort result (never claims achieved), timeout enforcement,
deterministic repeat, recommendation -> strategy, and state immutability.
"""
import time
from copy import deepcopy
from dataclasses import asdict

import pytest

from core.forecast_ai.optimization import (
    ReverseOptimizer,
    TargetGoal,
    OptimizationRequest,
    ScoreCalculator,
)
from core.forecast_ai.prediction.service import PredictionService
from core.forecast_ai.engines.recommendation_engine import (
    RecommendationEngine as EngineRecEngine,
)
from core.forecast_ai.engines.strategy_engine import StrategyEngine as EngineStratEngine


class MonotonicOH:
    """OH rises monotonically; max=105, min=50."""

    def predict(self, state):
        q = float(state.get("quality", 0.0))
        c = float(state.get("competency", 0.0))
        a = float(state.get("attendance", 0.0))
        return 50.0 + 0.4 * q + 0.1 * c + 0.05 * a


class MonotonicNPS:
    """NPS rises monotonically; max=100."""

    def predict(self, state):
        q = float(state.get("quality", 0.0))
        c = float(state.get("competency", 0.0))
        a = float(state.get("attendance", 0.0))
        return {"nps": 40.0 + 0.3 * q + 0.2 * c + 0.1 * a}


class FlatOH:
    def predict(self, state):
        return 80.0


class FlatNPS:
    def predict(self, state):
        return {"nps": 70.0}


class SlowOH:
    def predict(self, state):
        time.sleep(0.2)
        return 80.0


class SlowNPS:
    def predict(self, state):
        time.sleep(0.2)
        return {"nps": 70.0}


BASE = {"quality": 60.0, "competency": 60.0, "attendance": 60.0,
        "release": 50.0, "transfer": 10.0}


def _opt(oh_pred, nps_pred):
    service = PredictionService(oh_predictor=oh_pred(), nps_predictor=nps_pred())
    return ReverseOptimizer(prediction_service=service)


def _run(opt, target, mi=40, to=40):
    return opt.optimize(OptimizationRequest(
        initial_state=dict(BASE), target_goal=target,
        max_iterations=mi, timeout_seconds=to))


# --- directional distance (root cause) -------------------------------------- #
def test_distance_is_directional_overshoot_is_success():
    t = TargetGoal(target_operations_health=90.0, target_nps=65.0)
    assert ScoreCalculator.compute_distance(BASE, 95.0, 70.0, t) == 0.0
    assert ScoreCalculator.compute_distance(BASE, 90.0, 65.0, t) == 0.0
    assert ScoreCalculator.compute_distance(BASE, 85.0, 65.0, t) == pytest.approx(5.0)
    assert ScoreCalculator.compute_distance(BASE, 90.59, 83.33, t) == 0.0


# --- already-satisfied target ----------------------------------------------- #
def test_already_satisfied_target_succeeds_immediately():
    opt = _opt(MonotonicOH, MonotonicNPS)
    target = TargetGoal(target_operations_health=70.0, target_nps=60.0, tolerance=0.5)
    res = _run(opt, target)
    assert res.success is True
    assert res.metadata.get("target_achieved") is True
    assert res.metadata.get("best_effort") is False
    assert res.best_solution.distance_to_target == pytest.approx(0.0, abs=1e-6)


# --- feasible OH improvement ------------------------------------------------- #
def test_feasible_oh_improvement():
    opt = _opt(MonotonicOH, MonotonicNPS)
    target = TargetGoal(target_operations_health=95.0, tolerance=0.5)
    res = _run(opt, target)
    assert res.success is True
    assert res.metadata.get("target_achieved") is True
    assert res.best_solution.predicted_operations_health >= 95.0 - 0.5


# --- feasible combined target ------------------------------------------------ #
def test_feasible_combined_target():
    opt = _opt(MonotonicOH, MonotonicNPS)
    target = TargetGoal(target_operations_health=95.0, target_nps=80.0, tolerance=0.5)
    res = _run(opt, target)
    assert res.success is True
    bs = res.best_solution
    assert bs.predicted_operations_health >= 95.0 - 0.5
    assert bs.predicted_nps >= 80.0 - 0.5
# --- infeasible target fails honestly (no fabricated recommendations) ------- #
def test_infeasible_target_fails_honestly():
    opt = _opt(FlatOH, FlatNPS)  # model cannot move OH toward 90 at all
    target = TargetGoal(target_operations_health=90.0, tolerance=0.5)
    res = _run(opt, target)
    assert res.success is False
    assert res.metadata.get("best_effort") is False  # no improvement possible
    assert any("No solution within tolerance" in e for e in res.errors)
    from core.forecast_ai.recommendations import RecommendationEngine
    rec = RecommendationEngine().generate(res)
    assert rec.success is False
    assert rec.recommendations == []


# --- best-effort result ------------------------------------------------------ #
def test_best_effort_result_never_claims_achieved():
    opt = _opt(MonotonicOH, MonotonicNPS)
    target = TargetGoal(target_operations_health=130.0, tolerance=0.5)  # beyond max 105
    res = _run(opt, target)
    assert res.success is False
    assert res.metadata.get("target_achieved") is False
    assert res.metadata.get("best_effort") is True
    assert res.best_solution.predicted_operations_health <= 105.0 + 1e-6

    from core.forecast_ai.recommendations import RecommendationEngine
    rec = RecommendationEngine().generate(res)
    assert rec.success is True
    assert rec.metadata.get("goal_achieved") is False
    assert rec.metadata.get("best_effort") is True
    for r in rec.recommendations:
        assert r.metadata.get("goal_achieved") is False
        assert r.metadata.get("best_effort") is True
        assert r.metadata.get("gain_basis") == "assumption_target_not_reached"


# --- timeout stays enforced --------------------------------------------------- #
def test_timeout_still_enforced():
    service = PredictionService(oh_predictor=SlowOH(), nps_predictor=SlowNPS())
    engine = EngineRecEngine(optimizer_core=ReverseOptimizer(prediction_service=service))
    from core.forecast_ai.models import ForecastRequest as FR
    start = time.time()
    resp = engine.execute(FR(operation="recommend", parameters={
        "state": dict(BASE), "target_oh": 90.0,
        "max_iterations": 50, "timeout_seconds": 0.1}))
    wall = time.time() - start
    assert resp.success is False
    assert resp.metadata.get("reason") == "optimization_timeout"
    assert wall < 3.0


# --- deterministic repeat ------------------------------------------------------ #
def test_deterministic_repeat():
    opt = _opt(MonotonicOH, MonotonicNPS)
    target = TargetGoal(target_operations_health=95.0, target_nps=80.0, tolerance=0.5)
    r1 = _run(opt, target)
    r2 = _run(opt, target)
    assert r1.success == r2.success is True
    assert r1.best_solution.predicted_operations_health == \
        r2.best_solution.predicted_operations_health
    assert r1.best_solution.predicted_nps == r2.best_solution.predicted_nps
    assert r1.best_solution.distance_to_target == r2.best_solution.distance_to_target
# --- recommendation -> strategy ------------------------------------------------ #
def test_recommendation_to_strategy():
    service = PredictionService(oh_predictor=MonotonicOH(), nps_predictor=MonotonicNPS())
    engine = EngineRecEngine(optimizer_core=ReverseOptimizer(prediction_service=service))
    from core.forecast_ai.models import ForecastRequest as FR
    resp = engine.execute(FR(operation="recommend", parameters={
        "state": dict(BASE), "target_oh": 95.0, "target_nps": 80.0,
        "max_iterations": 40, "timeout_seconds": 40}))
    assert resp.success is True
    assert resp.metadata.get("goal_achieved") is True
    recs_block = resp.payload["recommendations"]
    assert recs_block.get("success") is True
    assert len(recs_block.get("recommendations") or []) >= 1

    strat_resp = EngineStratEngine().execute(FR(operation="strategy", parameters={
        "recommendation_result": recs_block}))
    assert strat_resp.success is True
    assert len(strat_resp.payload["strategies"]["strategies"]) >= 1


# --- state immutability --------------------------------------------------------- #
def test_state_immutability():
    opt = _opt(MonotonicOH, MonotonicNPS)
    target = TargetGoal(target_operations_health=95.0, target_nps=80.0, tolerance=0.5)
    state_before = deepcopy(BASE)
    res = _run(opt, target)
    assert BASE == state_before  # caller state untouched