"""Regression tests for the joint OH/NPS reverse-optimizer generated-candidates
fix (AxiPulseAI_reverse_optimizer_joint_oh_nps_fix).

Covers (in order):
  A. OH and NPS are jointly optimized.
  B. Reverse optimizer generates NEW candidate states (does not scan/select
     from an existing model/state catalogue).
  C. Multiple candidates are returned (not only best_solution).
  D. Each candidate prediction contains both OH and NPS.
  E. NPS uncertainty comes from the canonical Monte Carlo NPS percentiles over
     the 0..10 survey-score distribution.
  F. Changing the scalar NPS point forecast does NOT change the uncertainty
     interval when the underlying 0..10 distribution is unchanged.
  G. No target objective preserves abstention/no-action semantics.
  H. GUI/API recommendation output preserves multiple distinct generated
     candidates.
"""
from types import SimpleNamespace

import pytest

from core.forecast_ai.optimization import (
    ReverseOptimizer,
    OptimizationRequest,
    TargetGoal,
)
from core.forecast_ai.prediction.service import PredictionService


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


BASE = {"quality": 60.0, "competency": 60.0, "attendance": 60.0,
        "release": 50.0, "transfer": 10.0}


def _opt(oh, nps, service=None):
    if service is None:
        service = PredictionService(oh_predictor=oh, nps_predictor=nps)
    return ReverseOptimizer(prediction_service=service)


def _run(target, max_iterations=60, timeout=40):
    opt = _opt(MonotonicOH(), MonotonicNPS())
    return opt.optimize(OptimizationRequest(
        initial_state=dict(BASE), target_goal=target,
        max_iterations=max_iterations, timeout_seconds=timeout,
    ))


def _dist(center=8):
    import math
    weights = {s: math.exp(-0.5 * (s - center) ** 2) for s in range(0, 11)}
    total = sum(weights.values())
    return {f"score_{s}": w / total for s, w in weights.items()}


# --- A. Joint OH + NPS optimization ----------------------------------------- #
def test_oh_and_nps_are_jointly_optimized():
    target = TargetGoal(target_operations_health=95.0, target_nps=85.0, tolerance=0.5)
    res = _run(target)
    cands = res.metadata.get("ranked_candidates") or []
    feasible = [c for c in cands if c["feasible"]]
    assert feasible, "a feasible joint OH+NPS candidate must exist"
    best = feasible[0]
    assert best["predicted_operations_health"] >= 95.0
    assert best["predicted_nps"] >= 85.0


def test_joint_objective_requires_both_oh_and_nps():
    """Both target_oh and target_nps are accepted and appear on candidates."""
    target = TargetGoal(target_operations_health=95.0, target_nps=85.0, tolerance=0.5)
    res = _run(target)
    cands = res.metadata.get("ranked_candidates") or []
    for c in cands:
        assert c["target_operations_health"] == 95.0
        assert c["target_nps"] == 85.0
        assert c["optimization_basis"] == "joint_operations_health_and_nps"


# --- B. Generates new states, does not scan a catalogue --------------------- #
def test_optimizer_generates_new_candidate_states_not_scan():
    """The optimizer must call the forward model on NEW generated states, not
    scan/select from a pre-existing model/state catalogue."""
    evaluated_states = []

    class _OH:
        def predict(self, state):
            evaluated_states.append(dict(state))
            return 50.0 + 0.4 * float(state.get("quality", 0))

    class _NPS:
        def predict(self, state):
            evaluated_states.append(dict(state))
            return {"nps": 40.0 + 0.3 * float(state.get("quality", 0))}

    opt = _opt(_OH(), _NPS())
    target = TargetGoal(target_operations_health=120.0, tolerance=0.5)  # infeasible -> many states
    res = opt.optimize(OptimizationRequest(
        initial_state=dict(BASE), target_goal=target,
        max_iterations=60, timeout_seconds=40,
    ))

    assert res.metadata.get("ranked_candidates"), "must expose generated candidates"
    # The search evaluated the original PLUS generated variants -> >1 distinct.
    # Hashable key over scalar KPI values only (state may carry list values).
    def _key(s):
        return tuple(
            (k, tuple(v) if isinstance(v, list) else v)
            for k, v in sorted(s.items())
        )
    distinct_states = {_key(s) for s in evaluated_states}
    assert len(distinct_states) >= 2, "optimizer must generate multiple distinct states"
    # Every exposed candidate is explicitly a reverse-optimizer-generated state.
    for c in res.metadata["ranked_candidates"]:
        assert c["generated"] is True
        assert c["source"] == "reverse_optimizer_generated_state"


# --- C. Multiple candidates returned ---------------------------------------- #
def test_multiple_candidates_returned():
    target = TargetGoal(target_operations_health=120.0, tolerance=0.5)  # infeasible
    res = _run(target, max_iterations=60)
    cands = res.metadata.get("ranked_candidates") or []
    assert len(cands) >= 2, "more than one candidate must be exposed"
    # Distinct candidate states (not the same state repeated).
    states = [tuple(sorted((c["state"] or {}).items())) for c in cands]
    assert len(set(states)) >= 2


# --- D. Candidate predictions contain both OH and NPS ----------------------- #
def test_candidate_predictions_contain_oh_and_nps():
    target = TargetGoal(target_operations_health=95.0, target_nps=85.0, tolerance=0.5)
    res = _run(target)
    for c in res.metadata.get("ranked_candidates") or []:
        assert "predicted_operations_health" in c
        assert "predicted_nps" in c
        assert c["predicted_operations_health"] is not None
        assert c["predicted_nps"] is not None


# --- E. NPS uncertainty from canonical MC percentiles ----------------------- #
def test_candidate_nps_uncertainty_from_canonical_mc():
    """The candidate interval is exactly the canonical Monte Carlo percentiles
    the production path computed and preserved on the result — never an
    independent reverse-optimizer re-derivation."""
    opt = _opt(MonotonicOH(), MonotonicNPS())
    # Carried production MC percentiles (arbitrary but stable values that a
    # re-derivation from the distribution would NOT reproduce, proving the
    # optimizer reads the preserved interval rather than recomputing it).
    pred_result = SimpleNamespace(
        bayesian_score_distribution=_dist(8),
        score_counts={f"score_{i}": 5 for i in range(11)},
        nps=80.0,
        monte_carlo_nps_p05=-40.0,
        monte_carlo_nps_p50=-5.0,
        monte_carlo_nps_p95=30.0,
    )
    interval = opt._probabilistic_interval(pred_result, dict(BASE))
    assert interval is not None
    assert interval["basis"] == "monte_carlo_survey_score_distribution"
    # Exactly the carried canonical values -> proves no independent re-derivation.
    assert interval["p05"] == -40.0
    assert interval["p50"] == -5.0
    assert interval["p95"] == 30.0
    assert interval["p05"] <= interval["p50"] <= interval["p95"]


# --- F. Scalar NPS change does not change interval -------------------------- #
def test_scalar_nps_change_does_not_change_interval():
    """The candidate interval comes from the preserved production MC percentiles
    (0..10 distribution), so changing the scalar NPS point forecast does not
    change the uncertainty interval when the distribution is unchanged."""
    opt = _opt(MonotonicOH(), MonotonicNPS())
    dist = _dist(8)
    lo = SimpleNamespace(
        bayesian_score_distribution=dist, score_counts=None, nps=20.0,
        monte_carlo_nps_p05=-40.0, monte_carlo_nps_p50=-5.0, monte_carlo_nps_p95=30.0,
    )
    hi = SimpleNamespace(
        bayesian_score_distribution=dist, score_counts=None, nps=95.0,
        monte_carlo_nps_p05=-40.0, monte_carlo_nps_p50=-5.0, monte_carlo_nps_p95=30.0,
    )
    i_lo = opt._probabilistic_interval(lo, dict(BASE))
    i_hi = opt._probabilistic_interval(hi, dict(BASE))
    assert i_lo is not None and i_hi is not None
    # Same distribution -> identical interval, regardless of the scalar NPS.
    assert i_lo == i_hi
    assert i_lo["p05"] == -40.0


# --- Production interval is preserved through PredictionService ------------ #
def test_production_mc_interval_preserved_on_prediction_result():
    """The production PredictionService must preserve the canonical Monte Carlo
    NPS percentiles on PredictionResult so the reverse optimizer reads exactly
    the interval production computed."""
    from core.nps_predictor.inference import postprocess_predictions
    from core.forecast_ai.prediction.service import PredictionService

    dist = _dist(8)
    row = {"total_calls_received": 200, "actual_release_rate": 60.0,
           "operational_health": 80.0}
    raw = postprocess_predictions(
        [dist[f"score_{i}"] for i in range(11)], row
    )
    extracted = PredictionService._extract_nps_result(raw)
    assert extracted["monte_carlo_nps_p05"] is not None
    assert extracted["monte_carlo_nps_p50"] is not None
    assert extracted["monte_carlo_nps_p95"] is not None
    assert extracted["monte_carlo_nps_p05"] <= extracted["monte_carlo_nps_p50"] <= extracted["monte_carlo_nps_p95"]

    opt = _opt(MonotonicOH(), MonotonicNPS())
    pred_result = SimpleNamespace(
        bayesian_score_distribution=extracted["bayesian_score_distribution"],
        score_counts=extracted["score_counts"],
        nps=raw["nps"],
        monte_carlo_nps_p05=extracted["monte_carlo_nps_p05"],
        monte_carlo_nps_p50=extracted["monte_carlo_nps_p50"],
        monte_carlo_nps_p95=extracted["monte_carlo_nps_p95"],
    )
    interval = opt._probabilistic_interval(pred_result, dict(BASE))
    assert interval is not None
    assert interval["p05"] == extracted["monte_carlo_nps_p05"]
    assert interval["p50"] == extracted["monte_carlo_nps_p50"]
    assert interval["p95"] == extracted["monte_carlo_nps_p95"]


# --- G. No target -> no fabricated objective -------------------------------- #
# --- #1: joint_target_distance is display-only, not an optimizer objective --- #
def test_joint_target_distance_is_display_only_not_optimization_objective():
    """`_joint_target_distance` must ONLY order the exposed candidate display
    list. It must never be part of the search/best-solution selection, which is
    governed by the single canonical ScoreCalculator objective."""
    import inspect

    import core.forecast_ai.optimization.optimizer as optmod
    from core.forecast_ai.optimization.scoring import ScoreCalculator

    optimize_src = inspect.getsource(optmod.ReverseOptimizer.optimize)
    summaries_src = inspect.getsource(optmod.ReverseOptimizer._build_candidate_summaries)

    # _joint_target_distance is referenced only in the display summary builder.
    assert "_joint_target_distance" in summaries_src
    # It must not appear in the optimize() selection path (the code before the
    # candidate-summary build), i.e. it never drives best_solution/search.
    body_before_summaries = optimize_src.split("_build_candidate_summaries")[0]
    assert "_joint_target_distance" not in body_before_summaries

    # The authoritative best_solution selection is the canonical feasible list.
    assert "best_solution = acceptable[0]" in optimize_src
    # The canonical objective is ScoreCalculator's joint directional distance.
    scoring_src = inspect.getsource(ScoreCalculator.compute_distance)
    assert "target.target_operations_health" in scoring_src
    assert "target.target_nps" in scoring_src


def test_best_solution_meets_both_targets_via_canonical_objective():
    """The canonical best_solution must jointly satisfy OH and NPS targets,
    chosen by ScoreCalculator (not by the display-only joint distance)."""
    target = TargetGoal(target_operations_health=95.0, target_nps=85.0, tolerance=0.5)
    res = _run(target)
    assert res.best_solution is not None
    assert res.best_solution.distance_to_target <= target.tolerance
    assert res.best_solution.predicted_operations_health >= 95.0
    assert res.best_solution.predicted_nps >= 85.0
    # Display candidates still expose joint OH/NPS distance as metadata.
    cands = res.metadata.get("ranked_candidates") or []
    assert any(c["feasible"] and c["joint_oh_nps_distance"] is not None for c in cands)


def test_no_target_preserves_no_action_semantics():
    opt = _opt(MonotonicOH(), MonotonicNPS())
    res = opt.optimize(OptimizationRequest(
        initial_state=dict(BASE), target_goal=TargetGoal(),  # no objective
    ))
    assert res.success is False
    assert res.metadata.get("no_objective") is True
    assert res.metadata.get("ranked_candidates") == []
    assert res.best_solution is None
    assert res.errors, "must explain why no action was taken"


# --- H. Recommendation output preserves multiple distinct candidates -------- #
def test_recommendation_output_preserves_multiple_distinct_candidates():
    from core.forecast_ai.recommendations import RecommendationEngine

    target = TargetGoal(target_operations_health=120.0, tolerance=0.5)  # infeasible
    res = _run(target, max_iterations=60)
    rec = RecommendationEngine().generate(res)
    assert rec.success is True
    # Multiple candidates preserved on the RecommendationResult.
    assert len(rec.candidates) >= 2
    cand_states = [tuple(sorted((c.get("state") or {}).items())) for c in rec.candidates]
    assert len(set(cand_states)) >= 2, "candidates must be distinct, not duplicated"
    # Multiple distinct alternative recommendations (not the same one repeated).
    alt_recs = [r for r in rec.recommendations if r.metadata.get("candidate")]
    alt_states = [
        tuple(sorted((r.metadata.get("state") or {}).items())) for r in alt_recs
    ]
    assert len(set(alt_states)) >= 2


def test_recommendation_facade_preserves_candidates():
    """The ForecastAI recommendation facade payload must carry candidates so
    the data contract survives to the API/GUI."""
    from core.forecast_ai.engines.recommendation_engine import (
        RecommendationEngine as Facade,
    )
    from core.forecast_ai.models import ForecastRequest

    service = PredictionService(oh_predictor=MonotonicOH(), nps_predictor=MonotonicNPS())
    facade = Facade(optimizer_core=ReverseOptimizer(prediction_service=service))
    resp = facade.execute(ForecastRequest(operation="recommend", parameters={
        "state": dict(BASE), "target_oh": 120.0,
        "max_iterations": 60, "timeout_seconds": 40,
    }))
    recs_block = resp.payload["recommendations"]
    assert len(recs_block.get("candidates") or []) >= 2
    cand_states = [
        tuple(sorted((c.get("state") or {}).items()))
        for c in (recs_block.get("candidates") or [])
    ]
    assert len(set(cand_states)) >= 2
    # Also surfaced on the optimization metadata.
    ranked = (resp.payload.get("optimization", {}).get("metadata") or {}).get("ranked_candidates")
    assert len(ranked or []) >= 2
