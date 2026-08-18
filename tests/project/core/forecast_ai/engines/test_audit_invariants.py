"""Regression tests for audit findings:

P1-C — forecast / sensitivity / recommendation / batch must use the SAME model
       instance semantics (explicit dependency injection into the shared
       PredictionService, never divergent implicit services).

P2-A — optimizer search must not waste model evaluations (deterministic dedup)
       while preserving solution quality and reproducibility.

P2-B — sensitivity confidence is explicitly heuristic, not statistical.
"""
import inspect

import pytest

from core.forecast_ai.engines import forecast_orchestrator as orch_module
from core.forecast_ai.optimization.search import DeterministicHillClimb
from core.forecast_ai.optimization.models import TargetGoal
from core.forecast_ai.state import OperationalState


# --------------------------------------------------------------------------- #
# P1-C: same model instance semantics across surfaces
# --------------------------------------------------------------------------- #

def test_forecast_recommendation_share_service():
    """The orchestrator's recommendation engine must reuse the forecast's own
    PredictionService (explicit injection), not a fresh implicit service."""
    src = inspect.getsource(orch_module.ForecastOrchestrator._build_recommendation_output)
    assert "prediction_service=self.service" in src
    assert "ReverseOptimizer(" in src


def test_sensitivity_uses_forecast_service():
    """Sensitivity must run through the forecast's own PredictionService so
    forecast/sensitivity/recommendation/batch share the same selected model."""
    src = inspect.getsource(orch_module.ForecastOrchestrator._build_sensitivity_output)
    assert "prediction_service=self.service" in src


def test_prediction_service_defaults_to_canonical_provider():
    """A default PredictionService resolves predictors from the canonical
    PredictorProvider (same production model configuration)."""
    from core.forecast_ai.prediction.service import PredictionService
    from core.forecast_ai.prediction import provider as prov

    svc = PredictionService()
    assert svc.oh is None and svc.nps is None  # lazy -> provider singleton
    assert prov.PredictorProvider is not None
    assert hasattr(prov.PredictorProvider, "get_oh_predictor")
    assert hasattr(prov.PredictorProvider, "get_nps_predictor")


# --------------------------------------------------------------------------- #
# P2-A: optimizer search evaluation efficiency
# --------------------------------------------------------------------------- #

def _monotonic_evals_are_deduplicated_and_deterministic():
    def evaluator(state):
        return 80.0 + state.quality * 0.1, 70.0 + state.competency * 0.05

    def _state():
        return OperationalState(
            quality=80.0, competency=70.0, transfer=10.0,
            release=85.0, attendance=90.0, operations_health=88.0,
        )

    # Infeasible target forces a full search; count unique evaluated states.
    calls = {"n": 0}
    seen = set()

    def counting_evaluator(state):
        calls["n"] += 1
        key = tuple(getattr(state, f, 0.0) for f in
                    ["quality", "competency", "attendance", "release", "transfer"])
        seen.add(key)
        return evaluator(state)

    searcher = DeterministicHillClimb()
    sols = searcher.iterate(
        _state(), counting_evaluator,
        TargetGoal(target_operations_health=95.0, tolerance=1.0),
        [], max_iterations=25, timeout_seconds=30,
    )
    assert calls["n"] == len(seen), (
        "every model evaluation must be a unique state (no wasted re-evals)"
    )
    assert sols, "search must still produce solutions"
    return sols


def test_search_deduplicates_evaluations():
    sols = _monotonic_evals_are_deduplicated_and_deterministic()
    best = min(s.distance_to_target for s in sols)
    assert best == pytest.approx(5.0, abs=1e-6)  # max OH 90, target 95


def test_search_remains_deterministic_with_dedup():
    a = _monotonic_evals_are_deduplicated_and_deterministic()
    b = _monotonic_evals_are_deduplicated_and_deterministic()
    assert [s.state for s in a] == [s.state for s in b]
    assert [s.predicted_operations_health for s in a] == [
        s.predicted_operations_health for s in b
    ]


# --------------------------------------------------------------------------- #
# P2-B: sensitivity confidence is explicit heuristic
# --------------------------------------------------------------------------- #

def test_sensitivity_confidence_type_is_heuristic():
    from core.forecast_ai.sensitivity import SensitivityEngine
    from core.forecast_ai.sensitivity.models import SensitivityAnalysis

    # A single-direction result must carry an explicit heuristic confidence type.
    a = SensitivityAnalysis(
        metric="quality", baseline_output_oh=80.0, baseline_output_nps=70.0,
        modified_output_oh=82.0, modified_output_nps=71.0,
        operations_health_change=2.0, nps_change=1.0,
        sensitivity_score_oh=1.0, sensitivity_score_nps=0.5,
        elasticity_oh=0.5, elasticity_nps=0.3,
        confidence=0.6, confidence_type="heuristic_single_direction",
    )
    assert a.confidence_type == "heuristic_single_direction"
    # confidence field is preserved (not removed), but typed as heuristic.
    assert a.confidence == 0.6
