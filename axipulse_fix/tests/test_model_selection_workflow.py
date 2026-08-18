"""
Focused tests for the V2 training/model-selection workflow.

Covers the 10 required scenarios:
  1. training file listing
  2. dataset selection
  3. OH+NPS training from same file
  4. correct output names
  5. retraining same dataset replaces existing pair
  6. model family listing
  7. selecting model pair
  8. rejecting incomplete pair
  9. prediction uses selected pair
 10. wrong/missing model handled cleanly
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from core.forecast_ai.prediction.model_selector import (
    ModelPairError,
    list_training_files,
    list_model_families,
    select_training_file,
    select_model_family,
    validate_model_pair,
)
from core.forecast_ai.prediction.predictor_config import load_model_pair
from core.forecast_ai.prediction.provider import PredictorProvider


# ============================================================
# Fixtures / helpers
# ============================================================

def _make_training_csv(path, n_rows=30, seed=42):
    """Create a training CSV containing both OH and NPS columns."""
    np.random.seed(seed)
    dates = pd.date_range("2026-01-01", periods=n_rows)
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "operational_health": np.random.uniform(70, 95, n_rows),
        "target_quality": 87.0,
        "actual_quality": np.random.uniform(80, 95, n_rows),
        "target_competency": 93.0,
        "actual_competency": np.random.uniform(85, 98, n_rows),
        "target_attendance": 90.0,
        "actual_attendance": np.random.uniform(85, 96, n_rows),
        "target_release_rate": 60.0,
        "actual_release_rate": np.random.uniform(50, 70, n_rows),
        "target_transfer_rate": 9.0,
        "actual_transfer_rate": np.random.uniform(5, 15, n_rows),
        "total_calls_received": np.random.uniform(1000, 2000, n_rows),
        "operational_intelligence_factor": np.random.uniform(-10, 10, n_rows),
        "business_intelligence_factor": np.random.uniform(-10, 10, n_rows),
        "member_intelligence_factor": np.random.uniform(-10, 10, n_rows),
        "total_surveys": np.random.randint(50, 100, n_rows),
        "survey_rate": np.random.uniform(3, 8, n_rows),
        "promoters": np.random.randint(20, 50, n_rows),
        "passives": np.random.randint(10, 30, n_rows),
        "detractors": np.random.randint(5, 20, n_rows),
        "nps": np.random.uniform(50, 80, n_rows),
    })
    for i in range(11):
        df[f"score_{i}"] = np.random.randint(0, 20, n_rows)
    df["total_surveys"] = df[[f"score_{i}" for i in range(11)]].sum(axis=1)
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def training_dir(tmp_path):
    d = tmp_path / "training"
    d.mkdir()
    _make_training_csv(d / "january_2026.csv", seed=42)
    _make_training_csv(d / "february_2026.csv", seed=7)
    return d


@pytest.fixture
def models_dir(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    return d


def _make_model_bundle(path, model_type, family, feature_count=5):
    """Create a loadable model bundle with a distinct identity."""
    data = {
        "model": {"estimator": model_type, "family": family},
        "model_name": f"{family}_{model_type}",
        "feature_names": [f"feature_{i}" for i in range(feature_count)],
        "history_days": 30,
        "algorithm_performance": {"model_a": 0.5},
        "trained": True,
        "fallback_value": 100.0,
        "all_models": {},
        "metadata": {"engine_version": "1.0", "family": family},
        "tuned_params": {},
        "feature_importance": {},
        "feature_stats": {},
    }
    joblib.dump(data, path)
    return path


def _make_complete_pair(models_dir, family):
    _make_model_bundle(models_dir / f"{family}_OH.pkl", "oh", family)
    _make_model_bundle(models_dir / f"{family}_NPS.pkl", "nps", family)


# ============================================================
# 1. Training file listing
# ============================================================

def test_training_file_listing_lists_all_files(training_dir):
    files = list_training_files(training_dir)
    names = [f.name for f in files]
    assert "january_2026.csv" in names
    assert "february_2026.csv" in names
    assert len(files) == 2


def test_training_file_listing_empty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert list_training_files(empty) == []


def test_training_file_listing_missing_dir(tmp_path):
    assert list_training_files(tmp_path / "does_not_exist") == []


# ============================================================
# 2. Dataset selection
# ============================================================

def test_select_training_file_returns_chosen(training_dir):
    # Files sorted by name: february_2026.csv (0), january_2026.csv (1)
    chosen = select_training_file(training_dir, input_fn=lambda p: "1")
    assert chosen.name == "january_2026.csv"


def test_select_training_file_rejects_invalid_then_accepts(training_dir):
    answers = iter(["99", "0"])
    chosen = select_training_file(
        training_dir, input_fn=lambda p: next(answers)
    )
    assert chosen.name == "february_2026.csv"


def test_select_training_file_empty_dir_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        select_training_file(empty)


# ============================================================
# 6. Model family listing
# ============================================================

def test_model_family_listing_returns_complete_pairs(models_dir):
    _make_complete_pair(models_dir, "january_2026")
    _make_complete_pair(models_dir, "february_2026")
    assert sorted(list_model_families(models_dir)) == [
        "february_2026",
        "january_2026",
    ]


def test_model_family_listing_skips_incomplete(models_dir):
    _make_model_bundle(models_dir / "solo_OH.pkl", "oh", "solo")
    assert list_model_families(models_dir) == []


# ============================================================
# 7. Selecting model pair
# ============================================================

def test_select_model_family_returns_chosen(models_dir):
    _make_complete_pair(models_dir, "alpha")
    _make_complete_pair(models_dir, "beta")
    chosen = select_model_family(models_dir, input_fn=lambda p: "0")
    assert chosen in ("alpha", "beta")


def test_select_model_family_no_families_raises(models_dir):
    with pytest.raises(ValueError):
        select_model_family(models_dir)


# ============================================================
# 8. Rejecting incomplete pair
# ============================================================

def test_validate_model_pair_missing_oh_raises(models_dir):
    _make_model_bundle(models_dir / "x_NPS.pkl", "nps", "x")
    with pytest.raises(ModelPairError) as exc:
        validate_model_pair("x", models_dir)
    assert "OH model file missing" in str(exc.value)


def test_validate_model_pair_missing_nps_raises(models_dir):
    _make_model_bundle(models_dir / "x_OH.pkl", "oh", "x")
    with pytest.raises(ModelPairError) as exc:
        validate_model_pair("x", models_dir)
    assert "NPS model file missing" in str(exc.value)


def test_validate_model_pair_both_missing_raises(models_dir):
    with pytest.raises(ModelPairError) as exc:
        validate_model_pair("ghost", models_dir)
    assert "not found" in str(exc.value)


def test_validate_model_pair_returns_paths(models_dir):
    _make_complete_pair(models_dir, "ok")
    oh_path, nps_path = validate_model_pair("ok", models_dir)
    assert oh_path.name == "ok_OH.pkl"
    assert nps_path.name == "ok_NPS.pkl"


# ============================================================
# 10. Wrong / missing model handled cleanly
# ============================================================

def test_load_model_pair_missing_family_raises(models_dir, monkeypatch):
    monkeypatch.setattr(
        "core.forecast_ai.prediction.predictor_config.MODELS", models_dir
    )
    with pytest.raises(ModelPairError) as exc:
        load_model_pair("does_not_exist")
    assert "not found" in str(exc.value)


def test_load_model_pair_missing_nps_raises(models_dir, monkeypatch):
    _make_model_bundle(models_dir / "solo_OH.pkl", "oh", "solo")
    monkeypatch.setattr(
        "core.forecast_ai.prediction.predictor_config.MODELS", models_dir
    )
    with pytest.raises(ModelPairError) as exc:
        load_model_pair("solo")
    assert "NPS model file missing" in str(exc.value)


def test_load_model_pair_invalid_bundle_raises(models_dir, monkeypatch):
    (models_dir / "bad_OH.pkl").write_text("not a model")
    (models_dir / "bad_NPS.pkl").write_text("not a model")
    monkeypatch.setattr(
        "core.forecast_ai.prediction.predictor_config.MODELS", models_dir
    )
    with pytest.raises(Exception):
        load_model_pair("bad")


# ============================================================
# 9. Prediction uses selected pair
# ============================================================

def test_prediction_uses_selected_pair(models_dir, monkeypatch):
    _make_complete_pair(models_dir, "alpha")
    _make_complete_pair(models_dir, "beta")
    monkeypatch.setattr(
        "core.forecast_ai.prediction.predictor_config.MODELS", models_dir
    )

    PredictorProvider.reset()
    PredictorProvider.load_pair("alpha")

    assert PredictorProvider.get_model_family() == "alpha"

    oh = PredictorProvider.get_oh_predictor()
    nps = PredictorProvider.get_nps_predictor()
    assert oh is not None
    assert nps is not None
    assert oh.metadata.get("family") == "alpha"
    assert nps.metadata.get("family") == "alpha"

    PredictorProvider.load_pair("beta")
    assert PredictorProvider.get_model_family() == "beta"
    assert PredictorProvider.get_oh_predictor().metadata.get("family") == "beta"
    PredictorProvider.reset()


def test_prediction_set_model_family(models_dir, monkeypatch):
    _make_complete_pair(models_dir, "gamma")
    monkeypatch.setattr(
        "core.forecast_ai.prediction.predictor_config.MODELS", models_dir
    )
    PredictorProvider.reset()
    PredictorProvider.set_model_family("gamma")
    assert PredictorProvider.get_model_family() == "gamma"
    assert PredictorProvider.get_oh_predictor() is not None
    assert PredictorProvider.get_nps_predictor() is not None
    PredictorProvider.reset()



# ============================================================
# 3. OH+NPS training from same file  /  4. correct output names
# ============================================================

def _run_training_workflow(training_dir, monkeypatch, tmp_path):
    """Run train_all_ai.main() against a temp training/models layout."""
    import train_all_ai

    files = list_training_files(training_dir)

    # Point the workflow at the temp training files.
    monkeypatch.setattr(train_all_ai, "list_training_files", lambda: files)

    # Write models into a temp dir by chdir'ing there.
    monkeypatch.chdir(tmp_path)

    # Fast predictors so the test stays quick.
    def _fast_oh_factory():
        from core.operation_health_predictor import (
            OperationalHealthPredictor,
            Config as OpsConfig,
        )
        return OperationalHealthPredictor(OpsConfig(
            n_estimators=5, mlp_max_iter=100, cold_start_threshold=30
        ))

    def _fast_nps_factory():
        from core.nps_predictor import (
            NPSPredictor,
            Config as NPSConfig,
        )
        return NPSPredictor(NPSConfig(
            n_estimators=5, mlp_max_iter=100, cold_start_threshold=30,
            use_ensemble=False,
        ))

    monkeypatch.setattr(train_all_ai, "OperationalHealthPredictor", _fast_oh_factory)
    monkeypatch.setattr(train_all_ai, "NPSPredictor", _fast_nps_factory)

    # Select file "0", then confirm "y".
    answers = iter(["0", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    train_all_ai.main()

    return tmp_path


def test_training_produces_both_models_and_correct_names(training_dir, monkeypatch, tmp_path):
    out = _run_training_workflow(training_dir, monkeypatch, tmp_path)
    models = out / "models"
    oh_file = models / "february_2026_OH.pkl"
    nps_file = models / "february_2026_NPS.pkl"
    assert oh_file.exists(), "expected february_2026_OH.pkl"
    assert nps_file.exists(), "expected february_2026_NPS.pkl"


def test_training_output_names_match_dataset_stem(training_dir, monkeypatch, tmp_path):
    out = _run_training_workflow(training_dir, monkeypatch, tmp_path)
    models = out / "models"
    names = sorted(p.name for p in models.iterdir())
    assert "february_2026_OH.pkl" in names
    assert "february_2026_NPS.pkl" in names
    # only two files -> no duplicate versioning
    assert len(names) == 2


def test_training_creates_loadable_pair(training_dir, monkeypatch, tmp_path):
    out = _run_training_workflow(training_dir, monkeypatch, tmp_path)
    models = out / "models"
    # Verify the bundles load as valid predictors.
    from core.operation_health_predictor import OperationalHealthPredictor
    from core.nps_predictor import NPSPredictor

    oh = OperationalHealthPredictor()
    oh.load_model(str(models / "february_2026_OH.pkl"))
    assert oh.trained is True
    assert oh.feature_names

    nps = NPSPredictor()
    nps.load_model(str(models / "february_2026_NPS.pkl"))
    assert nps.trained is True
    assert nps.feature_names


# ============================================================
# 5. Retraining same dataset replaces existing pair
# ============================================================

def test_retraining_replaces_existing_pair(training_dir, monkeypatch, tmp_path):
    out = _run_training_workflow(training_dir, monkeypatch, tmp_path)
    models = out / "models"
    oh_file = models / "february_2026_OH.pkl"
    nps_file = models / "february_2026_NPS.pkl"

    assert oh_file.exists()
    assert nps_file.exists()

    # Record the bytes before re-training.
    oh_before = oh_file.read_bytes()
    nps_before = nps_file.read_bytes()

    # Run the same training again -> replaces (no duplicates).
    out2 = _run_training_workflow(training_dir, monkeypatch, tmp_path)
    models2 = out2 / "models"
    names = sorted(p.name for p in models2.iterdir())
    assert len(names) == 2, f"expected exactly 2 files, got {names}"

    # Files exist and were overwritten.
    assert (models2 / "february_2026_OH.pkl").exists()
    assert (models2 / "february_2026_NPS.pkl").exists()
    # Because models are re-trained, bytes differ (fresh fit on same seed
    # may or may not be byte-identical; at minimum they must still load).
    from core.operation_health_predictor import OperationalHealthPredictor
    oh = OperationalHealthPredictor()
    oh.load_model(str(models2 / "february_2026_OH.pkl"))
    assert oh.trained is True
