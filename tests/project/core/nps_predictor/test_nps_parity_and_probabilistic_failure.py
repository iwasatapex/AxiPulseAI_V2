"""Regression tests for:

P1-D — batch NPS prediction must match single-prediction semantics
        (same model/ensemble vector path => identical NPS, counts, metadata).

P1-F — probabilistic NPS failure must be visible (degraded/unavailable status,
        never a normal-looking probabilistic interval, no scalar-NPS fallback).
"""
import numpy as np
import pytest

from core.nps_predictor import inference as inf


class _RawModel:
    """Raw selected model returning a constant 11-vector (score 8-heavy)."""

    def __init__(self, vec):
        self._vec = np.asarray(vec, dtype=float)

    def predict(self, X):
        return [self._vec for _ in range(len(X))]


class _FakePredictor:
    def __init__(self, raw_vec, models, weights):
        self.model = _RawModel(raw_vec)
        self._all_models = {
            name: _RawModel(vec) for name, vec in models.items()
        }
        self.ensemble_weights = dict(weights)
        self.model_name = "fake"


def _row():
    return {
        "total_calls_received": 200,
        "actual_release_rate": 60.0,
        "operational_health": 80.0,
    }


# --------------------------------------------------------------------------- #
# P1-D: single == batch semantics
# --------------------------------------------------------------------------- #

def test_predict_single_vector_honors_weighted_ensemble():
    """predict_single_vector must apply the persisted weighted ensemble when
    present, matching what single prediction uses — never the raw model."""
    raw = np.zeros(11); raw[8] = 1.0  # raw model: all mass at score 8
    b = np.zeros(11); b[9] = 1.0      # model B: all mass at score 9 (promoter)
    c = np.zeros(11); c[6] = 1.0      # model C: all mass at score 6 (detractor)
    pred = _FakePredictor(raw, {"B": b, "C": c}, {"B": 0.6, "C": 0.4})

    vec = inf.predict_single_vector(pred, np.zeros((1, 1)))
    assert vec.shape == (11,)
    # 0.6*score9 + 0.4*score6 => weighted ensemble, NOT the raw score-8 mass.
    assert vec[9] == pytest.approx(0.6, abs=1e-9)
    assert vec[6] == pytest.approx(0.4, abs=1e-9)
    assert vec[8] == pytest.approx(0.0, abs=1e-9)


def test_single_and_batch_share_canonical_vector_path():
    """The batch helper (predict_single_vector + postprocess) and single
    prediction (predict_single) produce identical results for the same row."""
    raw = np.zeros(11); raw[8] = 1.0
    b = np.zeros(11); b[9] = 1.0
    c = np.zeros(11); c[6] = 1.0
    pred = _FakePredictor(raw, {"B": b, "C": c}, {"B": 0.6, "C": 0.4})
    X = np.zeros((1, 1))

    single = inf.predict_single(pred, X, _row())
    batch = inf.postprocess_predictions(
        inf.predict_single_vector(pred, X), _row()
    )

    assert single["nps"] == batch["nps"]
    assert single["score_counts"] == batch["score_counts"]
    assert single["promoters"] == batch["promoters"]
    assert single["passives"] == batch["passives"]
    assert single["detractors"] == batch["detractors"]
    assert single["bayesian_score_distribution"] == batch["bayesian_score_distribution"]


def test_batch_does_not_duplicate_probabilistic_logic():
    """The batch vector helper is a pure pre-postprocess vector builder; the
    probabilistic derivation lives only in postprocess_predictions."""
    raw = np.zeros(11); raw[8] = 1.0
    pred = _FakePredictor(raw, {}, {})
    X = np.zeros((1, 1))
    vec = inf.predict_single_vector(pred, X)
    # No probabilistic metadata appears on the raw vector.
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (11,)


# --------------------------------------------------------------------------- #
# P1-F: probabilistic failure is visible
# --------------------------------------------------------------------------- #

def test_probabilistic_success_has_full_metadata(monkeypatch):
    raw = np.zeros(11); raw[9] = 1.0
    pred = _FakePredictor(raw, {}, {})
    result = inf.postprocess_predictions(
        inf.predict_single_vector(pred, np.zeros((1, 1))), _row()
    )
    assert result["probabilistic_uncertainty"] == "available"
    # Canonical probabilistic fields from the Bayesian/Monte-Carlo layer.
    assert "bayesian_posterior_alpha" in result
    assert "monte_carlo_nps_p05" in result
    assert "monte_carlo_nps_p95" in result
    # Interval must be present and consistent.
    assert result["prediction_interval"]["low"] <= result["prediction_interval"]["high"]
    # NPS derived from 0..10 scores.
    assert -100.0 <= result["nps"] <= 100.0


def test_probabilistic_failure_is_degraded_and_visible(monkeypatch):
    """If the Bayesian/Monte-Carlo layer fails, the result must explicitly
    indicate probabilistic uncertainty is degraded/unavailable, keep a
    deterministic point result, and NOT fabricate a probabilistic interval."""
    def _boom(*a, **k):
        raise RuntimeError("injected probabilistic failure")

    monkeypatch.setattr(inf, "attach_probabilistic_analysis", _boom)

    raw = np.zeros(11); raw[8] = 1.0
    pred = _FakePredictor(raw, {}, {})
    result = inf.postprocess_predictions(
        inf.predict_single_vector(pred, np.zeros((1, 1))), _row()
    )
    assert result["probabilistic_uncertainty"] == "degraded_unavailable"
    assert "probabilistic_error" in result
    # Deterministic point interval only (low == high), never a fabricated band.
    assert result["prediction_interval"]["low"] == result["prediction_interval"]["high"]
    # NPS still derives from the 0..10 distribution.
    assert -100.0 <= result["nps"] <= 100.0
    assert "score_counts" in result


def test_nps_derived_from_0_10_scores_not_scalar_fallback():
    """NPS must be computed from the 0..10 survey-score distribution/counts,
    never from a scalar-NPS Bayesian/Monte-Carlo fallback."""
    raw = np.zeros(11)
    raw[0] = 1.0  # all detractors -> NPS = -100
    pred = _FakePredictor(raw, {}, {})
    result = inf.postprocess_predictions(
        inf.predict_single_vector(pred, np.zeros((1, 1))), _row()
    )
    assert result["detractors"] > 0
    assert result["promoters"] == 0
    assert result["nps"] == pytest.approx(-100.0, abs=1e-6)
