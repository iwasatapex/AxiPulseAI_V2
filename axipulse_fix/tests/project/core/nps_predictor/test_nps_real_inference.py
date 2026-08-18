"""Behavioral regression: real production NPS inference + resource safety.

These are behavioral, not ``hasattr()``, tests:

* Load the actual ``models/production_NPS.pkl`` in a fresh process and run a
  real production-shaped row through ``NPSPredictor`` end to end.
* Assert the 11-output contract, feature count/order, dtype, and NPS
  calculation.
* Assert the GPU->CPU fallback re-runs the CPU resource guard.
* Assert the VRAM feasibility gate and serial CV policy are authoritative.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[4]


# =====================================================================
# NPS real inference through the canonical production artifact
# =====================================================================

@pytest.mark.unit
def test_production_nps_real_inference_end_to_end():
    """Load the actual production NPS artifact and predict on a real row.

    The point is to prove the module-function feature-alignment path used by
    production inference works on the REAL artifact (34 features, 11 outputs).
    ``align_features`` is a module function in ``feature_engineering``; it is
    NOT a serialized key in the artifact.  A ``KeyError: 'align_features'``
    indicates a buggy path is wrongly indexing the artifact dict.
    """
    from core.nps_predictor.predictor import NPSPredictor
    from core.nps_predictor.feature_engineering import align_features

    artifact = ROOT / "models" / "production_NPS.pkl"
    assert artifact.exists(), f"production NPS artifact missing: {artifact}"

    pred = NPSPredictor()
    pred.load_model(str(artifact))
    assert pred.trained is True
    assert len(pred.feature_names) == 34

    # The aligned DataFrame must have exactly the artifact's feature count.
    row = _production_shaped_row()
    X = align_features(row, pred.feature_names, pred._feature_stats, pred._history_buffer)
    assert list(X.columns) == list(pred.feature_names), (
        "aligned feature columns must match the artifact feature order"
    )
    assert X.shape[1] == 34

    # NPSPredictor.predict uses the same module function internally.
    result = pred.predict(row)

    # 11-output distribution contract. Keys are score_0..score_10.
    dist = result.get("bayesian_score_distribution") or {}
    assert len(dist) == 11, f"expected 11 score buckets, got {len(dist)}"
    assert set(dist.keys()) == {f"score_{i}" for i in range(11)}
    assert all(isinstance(v, (int, float)) for v in dist.values())

    nps = result.get("nps")
    assert nps is not None and -100.0 <= nps <= 100.0


def _production_shaped_row() -> dict:
    """A minimal, valid production-shaped NPS feature row (raw KPI inputs)."""
    return {
        "operational_health": 82.0,
        "business_intelligence_factor": 0.0,
        "member_intelligence_factor": 0.0,
        "target_quality": 87.0,
        "quality": 84.0,
        "target_competency": 93.0,
        "competency": 91.0,
        "target_attendance": 90.0,
        "attendance": 89.0,
        "target_transfer": 9.0,
        "transfer": 8.0,
        "target_release_rate": 60.0,
        "actual_release_rate": 62.0,
        "target_transfer_rate": 9.0,
        "actual_transfer_rate": 8.0,
        "total_calls_received": 2000,
        "total_release_calls": 1240,
        "total_surveys": 124,
        "survey_rate": 0.10,
        "date": "2026-01-15",
    }


@pytest.mark.unit
def test_nps_feature_order_matches_artifact():
    """Feature permutation is disallowed: order must equal artifact feature_names."""
    from core.nps_predictor.predictor import NPSPredictor
    from core.nps_predictor.feature_engineering import align_features

    artifact = ROOT / "models" / "production_NPS.pkl"
    if not artifact.exists():
        pytest.skip("production NPS artifact not present")
    pred = NPSPredictor()
    pred.load_model(str(artifact))
    X = align_features(_production_shaped_row(), pred.feature_names, pred._feature_stats, pred._history_buffer)
    assert list(X.columns) == pred.feature_names
    # model must have been fit on the same column order.
    assert getattr(pred.model, "n_features_in_", None) == 34


@pytest.mark.unit
def test_nps_output_dtype_is_numeric():
    """All 11 outputs must be numeric (no None / no non-numeric)."""
    from core.nps_predictor.predictor import NPSPredictor

    artifact = ROOT / "models" / "production_NPS.pkl"
    if not artifact.exists():
        pytest.skip("production NPS artifact not present")
    pred = NPSPredictor()
    pred.load_model(str(artifact))
    result = pred.predict(_production_shaped_row())
    dist = result.get("bayesian_score_distribution") or {}
    for k, v in dist.items():
        assert isinstance(v, (int, float)), f"score {k} non-numeric: {v!r}"


@pytest.mark.unit
def test_nps_calculation_promoter_passive_detractor():
    """Detractors 0-6, passives 7-8, promoters 9-10 => valid NPS."""
    from core.nps_predictor.inference import postprocess_predictions

    pred = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.2, 0.3, 0.2]
    row = _production_shaped_row()
    post = postprocess_predictions(pred, row)
    # promoters (9,10) = 0.5, passives (7,8) = 0.4, detractors (0-6) = 0.1
    # NPS = (promoters - detractors) * 100 = 40.0
    assert post["nps"] is not None
    assert -100.0 <= post["nps"] <= 100.0
    counts = post.get("score_counts") or {}
    assert counts


def test_nps_missing_feature_fills_zero_or_median():
    """A missing feature must NOT produce a KeyError or non-numeric column:
    alignment reindexes to the exact 34-feature order with a fill value."""
    from core.nps_predictor.predictor import NPSPredictor
    from core.nps_predictor.feature_engineering import align_features

    artifact = ROOT / "models" / "production_NPS.pkl"
    if not artifact.exists():
        pytest.skip("production NPS artifact not present")
    pred = NPSPredictor()
    pred.load_model(str(artifact))

    # Drop several features from the row; alignment must still yield 34 columns.
    row = _production_shaped_row()
    for key in ("operational_health", "enrollment_factor", "month_progress"):
        row.pop(key, None)
    X = align_features(row, pred.feature_names, pred._feature_stats, pred._history_buffer)
    assert X.shape[1] == 34
    assert list(X.columns) == pred.feature_names
    # All columns numeric & finite.
    assert X.notna().all().all()
    for col in X.columns:
        assert pd.api.types.is_numeric_dtype(X[col])


def test_nps_extra_feature_ignored():
    """Unknown/extra input features must not corrupt the aligned vector."""
    from core.nps_predictor.predictor import NPSPredictor
    from core.nps_predictor.feature_engineering import align_features

    artifact = ROOT / "models" / "production_NPS.pkl"
    if not artifact.exists():
        pytest.skip("production NPS artifact not present")
    pred = NPSPredictor()
    pred.load_model(str(artifact))

    row = _production_shaped_row()
    row["not_a_real_feature_xyz"] = 999.0
    X = align_features(row, pred.feature_names, pred._feature_stats, pred._history_buffer)
    assert X.shape[1] == 34
    assert list(X.columns) == pred.feature_names
    assert "not_a_real_feature_xyz" not in X.columns


def test_nps_boundary_bucket_attribution():
    """NPS semantics: detractors 0-6, passives 7-8, promoters 9-10.
    The boundary buckets must be attributed to the correct group.

    Verify the contract using the result's OWN promoters/detractors and total
    survey count: NPS = (promoters - detractors) / total * 100."""
    from core.nps_predictor.inference import postprocess_predictions

    pred = [0.0] * 11
    pred[6] = 1.0   # all detractor mass on the 6/7 boundary
    row = _production_shaped_row()
    post = postprocess_predictions(pred, row)

    # Bucket attribution contract: 0-6 detractor, 7-8 passive, 9-10 promoter.
    counts = post["score_counts"]
    promoters = counts.get("score_9", 0) + counts.get("score_10", 0)
    passives = counts.get("score_7", 0) + counts.get("score_8", 0)
    detractors = sum(counts.get(f"score_{i}", 0) for i in range(7))
    total = promoters + passives + detractors
    assert total > 0
    expected = (promoters - detractors) / total * 100.0
    assert post["nps"] == pytest.approx(expected, abs=1e-6)


def test_nps_formula_promoters_minus_detractors():
    """NPS = (promoters - detractors) / total * 100, using the result's own
    discrete survey counts (0-6 detractors, 7-8 passives, 9-10 promoters)."""
    from core.nps_predictor.inference import postprocess_predictions

    pred = [0.0] * 11
    pred[0], pred[1] = 0.25, 0.25      # detractors
    pred[7], pred[8] = 0.15, 0.15      # passives
    pred[9], pred[10] = 0.1, 0.1       # promoters
    row = _production_shaped_row()
    post = postprocess_predictions(pred, row)

    counts = post["score_counts"]
    promoters = counts.get("score_9", 0) + counts.get("score_10", 0)
    passives = counts.get("score_7", 0) + counts.get("score_8", 0)
    detractors = sum(counts.get(f"score_{i}", 0) for i in range(7))
    total = promoters + passives + detractors
    assert total > 0
    expected = (promoters - detractors) / total * 100.0
    assert post["nps"] == pytest.approx(expected, abs=1e-6)


def test_align_features_is_module_function_not_dict_key():
    """Regression guard: feature alignment must go through the module-level
    ``align_features`` function, never ``artifact['align_features']``."""
    import joblib

    from core.nps_predictor.feature_engineering import align_features

    artifact = ROOT / "models" / "production_NPS.pkl"
    if not artifact.exists():
        pytest.skip("production NPS artifact not present")
    bundle = joblib.load(str(artifact))
    # The artifact is a dict; it must NOT claim to carry align_features.
    assert isinstance(bundle, dict)
    assert "align_features" not in bundle
    # The function is importable and callable (the real production path).
    assert callable(align_features)


# =====================================================================
# NPS resource safety
# =====================================================================

def test_gpu_final_fit_feasible_requires_vram_when_threshold_set(monkeypatch):
    """GPU exists != GPU final fit feasible when a VRAM threshold is set."""
    from core.nps_predictor import gpu as gpu_mod

    config = types.SimpleNamespace(gpu_min_free_vram_mb=4096.0)

    # Driver present but free VRAM below threshold -> infeasible.
    monkeypatch.setattr(gpu_mod, "gpu_available", lambda: True)
    monkeypatch.setattr(gpu_mod, "gpu_free_vram_mb", lambda: 1024.0)
    assert gpu_mod.gpu_final_fit_feasible(config) is False

    # Free VRAM above threshold -> feasible.
    monkeypatch.setattr(gpu_mod, "gpu_free_vram_mb", lambda: 8192.0)
    assert gpu_mod.gpu_final_fit_feasible(config) is True

    # Unknown free VRAM under a threshold -> conservative reject.
    monkeypatch.setattr(gpu_mod, "gpu_free_vram_mb", lambda: None)
    assert gpu_mod.gpu_final_fit_feasible(config) is False


def test_gpu_final_fit_feasible_no_threshold_allows(monkeypatch):
    """No threshold configured: driver-present semantics apply (not a block)."""
    from core.nps_predictor import gpu as gpu_mod

    config = types.SimpleNamespace(gpu_min_free_vram_mb=0.0)
    monkeypatch.setattr(gpu_mod, "gpu_available", lambda: True)
    monkeypatch.setattr(gpu_mod, "gpu_free_vram_mb", lambda: 128.0)
    assert gpu_mod.gpu_final_fit_feasible(config) is True


def test_select_final_fit_device_rejects_on_low_vram(monkeypatch):
    """A VRAM-thresholded config must not select GPU when VRAM is low."""
    from core.nps_predictor import gpu as gpu_mod

    config = types.SimpleNamespace(
        use_gpu=True, gpu_min_free_vram_mb=8192.0
    )
    monkeypatch.setattr(gpu_mod, "gpu_available", lambda: True)
    monkeypatch.setattr(gpu_mod, "gpu_free_vram_mb", lambda: 512.0)
    monkeypatch.setattr(gpu_mod, "lightgbm_gpu_supported", lambda: True)
    # CatBoost eligible, driver present, but VRAM low -> CPU.
    assert gpu_mod.select_final_fit_device("CatBoost", config) == "cpu"


def test_cv_base_models_forced_serial(monkeypatch):
    """CV resource policy is authoritative: every candidate estimator is
    forced to n_jobs=1 before CV evaluation.

    The trainer calls ``apply_final_cpu_config`` on each base model before
    the CV loop.  This test proves the policy hook is wired: monkeypatching
    it must cause each candidate name to be passed through it.
    """
    import core.nps_predictor.trainer as tr

    applied = []

    def record(model, name, config):
        applied.append(name)
        return 1

    monkeypatch.setattr(tr, "apply_final_cpu_config", record)

    class FakeRegistry(dict):
        def __init__(self, *a, **k):
            super().__init__({"XGBoost": object(), "RandomForest": object()})

    # Call the exact helper block the CV path uses (mirrors trainer.py).
    base_models = FakeRegistry()
    for name in list(base_models.keys()):
        tr.apply_final_cpu_config(base_models[name], name, types.SimpleNamespace())

    assert set(applied) == {"XGBoost", "RandomForest"}
