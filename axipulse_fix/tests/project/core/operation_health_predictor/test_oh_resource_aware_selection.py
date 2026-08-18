"""
Regression tests: resource-aware OH final-fit feasibility in model selection.

Guarantees:
- A candidate can win OH CV yet be FINAL-FIT INFEASIBLE under the configured
  budget at the full training row count (the audited gap in trainer.py).
- Such infeasible candidates are excluded from winner selection with an
  explicit reason; the best CV-MAE FEASIBLE candidate wins.
- All candidates remain in the registry / algorithm_performance (resource-
  aware selection, not candidate deletion).
- The selected model is still trained on all rows (no row dropping).
- No post-selection model substitution; exactly one final full-data fit.
- If every candidate is infeasible, training fails with a clear reason listing
  each candidate's estimate.
- Resource diagnostics (cv_score, final_fit_estimated_memory_mb,
  final_fit_feasible, reason_if_not_feasible) are produced per candidate and
  persisted in the OH model metadata and saved payload.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone as real_clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

import core.operation_health_predictor.trainer as trainer_mod
from core.operation_health_predictor.config import Config
from core.operation_health_predictor.predictor import OperationalHealthPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cv_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, 5)).astype("float32"),
        columns=[f"f{i}" for i in range(5)],
    )
    y = pd.Series(rng.uniform(0.0, 100.0, n), dtype=np.float32)
    return X, y


def _tree_models():
    """One infeasible-at-scale tree family + two feasible tree families."""
    return {
        # 500 trees: estimated final fit at 100k rows is far above 4 GiB.
        "ExtraTrees": ExtraTreesRegressor(n_estimators=500, random_state=42, n_jobs=1),
        # 2 trees: feasible under 4096 MiB even at 100k rows.
        "RandomForest": RandomForestRegressor(n_estimators=2, random_state=42, n_jobs=1),
        # 3 trees: feasible but with a WORSE CV MAE than RandomForest.
        "RandomForestWorse": RandomForestRegressor(n_estimators=3, random_state=42, n_jobs=1),
    }


def _all_infeasible_models():
    return {
        "ExtraTrees": ExtraTreesRegressor(n_estimators=500, random_state=42, n_jobs=1),
        "RandomForest": RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=1),
    }


def _fake_predictor(full_rows=100_620):
    predictor = _FakePredictor()
    predictor.config = Config(
        n_estimators=5,
        mlp_hidden_layers=(4,),
        mlp_max_iter=2,
        use_cyclical_dates=False,
        clip_outliers=False,
    )
    predictor.config.cv_folds = 2
    predictor.config.cv_timeout = 5.0
    predictor._training_dates = pd.Series(
        pd.date_range("2026-01-01", periods=120, freq="D")
    ).reset_index(drop=True)
    predictor._full_training_rows = full_rows
    predictor._full_training_cols = 5
    return predictor


def _ok_score(name):
    # ExtraTrees wins CV (lowest MAE); RandomForest second; RandomForestWorse last.
    if name == "ExtraTrees":
        return {"status": "ok", "score": 0.01, "elapsed": 0.05}
    if name == "RandomForest":
        return {"status": "ok", "score": 0.2, "elapsed": 0.05}
    return {"status": "ok", "score": 0.9, "elapsed": 0.05}


def _fake_subprocess(name, model, Xtr, ytr, Xva, yva, timeout, metric="mae", heartbeat=None):
    return _ok_score(name)


class _FitSpy:
    """Wrap an estimator so final-fit calls are observable."""
    def __init__(self, est, observer):
        self._est = est
        self._observer = observer

    def fit(self, X, y, **kwargs):
        self._observer(X, y)
        return self._est.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return self._est.predict(X, **kwargs)

    def get_params(self, deep=True):
        return self._est.get_params(deep)

    def set_params(self, **params):
        self._est.set_params(**params)
        return self

    def __getattr__(self, name):
        return getattr(self._est, name)


def _write_oh_csv(path, n_days=50, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
    records = []
    for day in dates:
        records.append(
            {
                "date": day.strftime("%Y-%m-%d"),
                "target_quality": float(rng.uniform(80, 100)),
                "actual_quality": float(rng.uniform(80, 100)),
                "target_competency": float(rng.uniform(80, 100)),
                "actual_competency": float(rng.uniform(80, 100)),
                "target_attendance": float(rng.uniform(80, 100)),
                "actual_attendance": float(rng.uniform(80, 100)),
                "target_release_rate": float(rng.uniform(50, 90)),
                "actual_release_rate": float(rng.uniform(50, 90)),
                "target_transfer_rate": float(rng.uniform(0, 20)),
                "actual_transfer_rate": float(rng.uniform(0, 20)),
                "total_calls_received": int(rng.uniform(2000, 5000)),
                "operational_intelligence_factor": float(rng.uniform(-100, 100)),
                "operational_health": float(rng.uniform(0, 100)),
            }
        )
    pd.DataFrame(records).to_csv(path, index=False)


# -------------------------------------------------------------------------
# Selection block: infeasible CV winner excluded, best feasible wins
# -------------------------------------------------------------------------

def test_oh_infeasible_cv_winner_excluded_and_best_feasible_wins(monkeypatch):
    predictor = _fake_predictor()
    predictor.create_model_registry = lambda cfg, cold_start=False: _tree_models()
    monkeypatch.setattr(trainer_mod, "evaluate_fold_in_subprocess", _fake_subprocess)

    X, y = _cv_data()
    trainer_mod.TrainerMixin._rolling_origin_train(predictor, X, y, progress=None)

    # ExtraTrees has the best CV MAE but can never final-fit under the default
    # budget at 100k rows -> excluded. RandomForest (2 trees) beats
    # RandomForestWorse (3 trees) on CV MAE -> wins.
    assert predictor.model_name == "RandomForest"
    assert set(predictor.algorithm_performance) == {
        "ExtraTrees", "RandomForest", "RandomForestWorse",
    }

    diag = predictor.model_selection_diagnostics
    assert diag["ExtraTrees"]["final_fit_feasible"] is False
    assert "budget" in (diag["ExtraTrees"]["reason_if_not_feasible"] or "").lower()
    assert diag["ExtraTrees"]["final_fit_estimated_memory_mb"] is not None
    assert diag["ExtraTrees"]["cv_score"] == 0.01
    assert diag["RandomForest"]["final_fit_feasible"] is True
    assert diag["RandomForestWorse"]["final_fit_feasible"] is True
    assert diag["RandomForest"]["cv_score"] == 0.2
    assert diag["RandomForestWorse"]["cv_score"] == 0.9
    assert diag["RandomForest"]["cv_score"] < diag["RandomForestWorse"]["cv_score"]


# -------------------------------------------------------------------------
# No candidate deleted: entire registry preserved
# -------------------------------------------------------------------------

def test_oh_all_candidates_remain_in_registry_after_selection(monkeypatch):
    predictor = _fake_predictor()
    predictor.create_model_registry = lambda cfg, cold_start=False: _tree_models()
    monkeypatch.setattr(trainer_mod, "evaluate_fold_in_subprocess", _fake_subprocess)
    X, y = _cv_data()
    trainer_mod.TrainerMixin._rolling_origin_train(predictor, X, y, progress=None)
    # Resource-aware selection does NOT delete candidates: every one that
    # produced a CV score stays visible in algorithm_performance and the
    # per-candidate diagnostics registry.
    assert set(predictor.algorithm_performance) == {
        "ExtraTrees", "RandomForest", "RandomForestWorse",
    }
    assert set(predictor.model_selection_diagnostics) == {
        "ExtraTrees", "RandomForest", "RandomForestWorse",
    }


# -------------------------------------------------------------------------
# All candidates infeasible -> clear diagnostic
# -------------------------------------------------------------------------

def test_oh_all_candidates_infeasible_raises_clear_error(monkeypatch):
    predictor = _fake_predictor()
    predictor.config.final_fit_memory_budget_mb = 0.001
    predictor.create_model_registry = lambda cfg, cold_start=False: _all_infeasible_models()
    monkeypatch.setattr(trainer_mod, "evaluate_fold_in_subprocess", _fake_subprocess)
    X, y = _cv_data()
    with pytest.raises(RuntimeError) as excinfo:
        trainer_mod.TrainerMixin._rolling_origin_train(predictor, X, y, progress=None)
    msg = str(excinfo.value)
    assert "No OH candidate is final-fit feasible" in msg
    assert "ExtraTrees" in msg and "RandomForest" in msg
    assert "budget" in msg.lower()


# -------------------------------------------------------------------------
# Diagnostics fields complete per candidate
# -------------------------------------------------------------------------

def test_oh_diagnostics_fields_present_for_every_candidate(monkeypatch):
    predictor = _fake_predictor()
    predictor.create_model_registry = lambda cfg, cold_start=False: _tree_models()
    monkeypatch.setattr(trainer_mod, "evaluate_fold_in_subprocess", _fake_subprocess)
    X, y = _cv_data()
    trainer_mod.TrainerMixin._rolling_origin_train(predictor, X, y, progress=None)
    for name, diag in predictor.model_selection_diagnostics.items():
        assert "cv_score" in diag
        assert "final_fit_estimated_memory_mb" in diag
        assert "final_fit_feasible" in diag
        assert "reason_if_not_feasible" in diag


# -------------------------------------------------------------------------
# Full train(): no rows dropped, no substitution, exactly one final fit,
# diagnostics persisted in metadata and in the saved payload
# -------------------------------------------------------------------------
def test_oh_train_single_full_fit_no_substitution_diagnostics_persisted(monkeypatch, tmp_path):
    csv_path = tmp_path / "oh_train.csv"
    _write_oh_csv(csv_path, n_days=50)

    predictor = OperationalHealthPredictor(
        config=Config(
            n_estimators=2,
            mlp_hidden_layers=(4,),
            mlp_max_iter=2,
            use_cyclical_dates=False,
            clip_outliers=False,
        )
    )
    # Pretend the full hosted matrix is 500k rows so ExtraTrees (400 trees)
    # is final-fit infeasible while RandomForest (2 trees) is feasible.
    predictor._full_training_rows = 500_000
    predictor._full_training_cols = 5
    predictor.create_model_registry = lambda cfg, cold_start=False: {
        "ExtraTrees": ExtraTreesRegressor(n_estimators=400, random_state=42, n_jobs=1),
        "RandomForest": RandomForestRegressor(n_estimators=2, random_state=42, n_jobs=1),
    }

    final_fit_rows = []

    def spy_clone(estimator):
        return _FitSpy(real_clone(estimator), lambda X, y: final_fit_rows.append(len(X)))

    def fake(name, model, Xtr, ytr, Xva, yva, timeout, metric="mae", heartbeat=None):
        return {"status": "ok", "score": 0.02 if name == "ExtraTrees" else 0.6, "elapsed": 0.05}

    monkeypatch.setattr(trainer_mod, "evaluate_fold_in_subprocess", fake)
    monkeypatch.setattr(trainer_mod, "clone", spy_clone)
    predictor._compute_shap = lambda X: None
    predictor._compute_feature_importance = lambda X, y: None

    predictor.train(str(csv_path))

    # Exactly one full-data fit, on the FULL aligned matrix (no row dropping).
    # The 50-day CSV yields 49 shifted rows -> the final fit sees all 49.
    assert len(final_fit_rows) == 1
    assert final_fit_rows == [49]
    # The infeasible CV winner (ExtraTrees) is NOT selected or substituted.
    assert predictor.model_name == "RandomForest"
    assert set(predictor._all_models) == {"RandomForest"}
    assert predictor._all_models["RandomForest"] is predictor.model
    # Diagnostics persisted in metadata.
    msd = predictor.metadata["model_selection_diagnostics"]
    assert msd["ExtraTrees"]["final_fit_feasible"] is False
    assert "budget" in (msd["ExtraTrees"]["reason_if_not_feasible"] or "").lower()
    assert msd["RandomForest"]["final_fit_feasible"] is True

    # And they ride along when the model is saved / loaded. The _FitSpy wrapper
    # is a test artifact (holds a local lambda) and cannot be pickled, so store
    # the real fitted estimator it wrapped — persistence shape identical to the
    # production model payload.
    predictor.model = predictor.model._est
    predictor._all_models = {predictor.model_name: predictor.model}
    pkl = tmp_path / "oh_model.pkl"
    predictor.save_model(str(pkl))
    loaded = OperationalHealthPredictor(config=Config())
    loaded.load_model(str(pkl))
    loaded_msd = loaded.metadata["model_selection_diagnostics"]
    assert loaded_msd["ExtraTrees"]["final_fit_feasible"] is False
    assert loaded_msd["RandomForest"]["final_fit_feasible"] is True


# -------------------------------------------------------------------------
# Fake predictor
# -------------------------------------------------------------------------

class _FakePredictor:
    def __init__(self):
        self.config = None
        self.model = None
        self.model_name = None
        self.algorithm_performance = {}
        self._all_models = {}
        self.cv_timing = {}
        self.model_selection_diagnostics = {}
        self._training_dates = None
        self._full_training_rows = None
        self._full_training_cols = None
