"""
Regression tests for the RAM-safe NPS training lifecycle.

Covered guarantees:

- Sample-fitted candidate models are released from ``predictor._all_models``
  BEFORE the final full-data refit (``_all_models`` is empty at the instant
  the full-data ``fit`` call begins).
- Only the selected winner is retained after training; candidates are not
  recreated, and their CV scores remain as leaderboard metadata only.
- Candidate CV runs serially (``n_jobs=1``) by default for the production
  training path.
- Prediction still works after the winner is saved and reloaded.
"""
import numpy as np
import pandas as pd
import pytest

from joblib import Parallel as RealParallel
from sklearn.base import clone as _real_clone

from core.nps_predictor import Config, NPSPredictor


def _write_training_csv(path, n_days=90, rows_per_day=1, seed=7):
    """Write a small deterministic NPS training CSV (repeated-date capable)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
    records = []
    for day in dates:
        for _ in range(rows_per_day):
            scores = rng.integers(0, 40, size=11)
            records.append(
                {
                    "date": day.strftime("%Y-%m-%d"),
                    "operational_health": float(rng.uniform(40, 100)),
                    "business_intelligence_factor": float(rng.uniform(-100, 100)),
                    "member_intelligence_factor": float(rng.uniform(-100, 100)),
                    "target_release_rate": float(rng.uniform(0, 100)),
                    "actual_release_rate": float(rng.uniform(0, 100)),
                    "total_calls_received": int(rng.integers(100, 2000)),
                    "total_surveys": int(scores.sum()),
                    "survey_rate": float(rng.uniform(0.2, 1.0)),
                    "target_quality": float(rng.uniform(0, 100)),
                    "quality": float(rng.uniform(0, 100)),
                    "target_competency": float(rng.uniform(0, 100)),
                    "competency": float(rng.uniform(0, 100)),
                    "target_attendance": float(rng.uniform(0, 100)),
                    "attendance": float(rng.uniform(0, 100)),
                    "target_transfer": float(rng.uniform(0, 100)),
                    "transfer": float(rng.uniform(0, 100)),
                    "promoters": int(scores[9:].sum()),
                    "passives": int(scores[7:9].sum()),
                    "detractors": int(scores[:7].sum()),
                    **{f"score_{i}": int(scores[i]) for i in range(11)},
                }
            )
    pd.DataFrame(records).to_csv(path, index=False)


def _small_config():
    return Config(
        n_estimators=5,
        mlp_hidden_layers=(4,),
        mlp_max_iter=2,
        use_cyclical_dates=False,
        clip_outliers=False,
        sample_for_selection=True,
        sample_size=40,
        history_buffer_days=3,
        cv_n_jobs=1,
        use_gpu=False,
    )


class _FitSpy:
    """Delegating wrapper that records the predictor state at fit time."""

    def __init__(self, estimator, observer):
        self._est = estimator
        self._observer = observer

    def fit(self, X, y, **kwargs):
        self._observer(X)
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


def test_candidates_released_before_full_data_refit(tmp_path, monkeypatch):
    """_all_models must be empty immediately before the final full-data fit."""
    import core.nps_predictor.trainer as trainer_mod

    n_days = 90
    full_rows = n_days - 1  # the last date has no T+1 target

    config = _small_config()
    predictor = NPSPredictor(config=config)

    csv_path = tmp_path / "nps_train.csv"
    _write_training_csv(csv_path, n_days=n_days)

    final_fit_snapshots = []
    parallel_workers = []

    def observe(X):
        if len(X) == full_rows:
            final_fit_snapshots.append(dict(predictor._all_models))

    def spy_clone(estimator):
        return _FitSpy(_real_clone(estimator), observe)

    class ParallelSpy:
        def __init__(self, *args, **kwargs):
            parallel_workers.append(kwargs.get("n_jobs"))
            self._impl = RealParallel(*args, **kwargs)

        def __call__(self, iterable):
            return self._impl(iterable)

    monkeypatch.setattr(trainer_mod, "clone", spy_clone)
    monkeypatch.setattr(trainer_mod, "Parallel", ParallelSpy)

    predictor.train(str(csv_path))

    assert predictor.trained is True

    # Exactly one full-data fit was observed, and at that instant the
    # sample-fitted candidates had already been released.
    assert len(final_fit_snapshots) == 1
    assert final_fit_snapshots == [{}]

    # Only the selected winner is retained after training.
    assert set(predictor._all_models) == {predictor.model_name}
    assert predictor._all_models[predictor.model_name] is predictor.model

    # Candidate CV ran serially by default (no worker-process base RAM).
    assert parallel_workers == [1]

    # Leaderboard performance survives as metadata only.
    assert set(predictor.algorithm_performance) >= {predictor.model_name}


def test_default_selection_sample_and_cv_folds_are_cheap():
    """Defaults: selection sample <=1000, CV uses <=2 folds, serial workers."""
    from core.nps_predictor.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG.sample_size <= 1000
    assert DEFAULT_CONFIG.cv_folds <= 2
    assert DEFAULT_CONFIG.cv_n_jobs in (1, 2)


def test_cv_uses_two_folds_by_default(tmp_path, monkeypatch):
    """Date-aware CV is requested with the configurable fold count (default 2)."""
    import core.nps_predictor.trainer as trainer_mod

    config = _small_config()
    predictor = NPSPredictor(config=config)

    csv_path = tmp_path / "nps_train.csv"
    _write_training_csv(csv_path, n_days=90)

    observed_folds = []
    original_splits = trainer_mod._build_date_aware_splits

    def spy_splits(dates, n_splits):
        observed_folds.append(n_splits)
        return original_splits(dates, n_splits)

    monkeypatch.setattr(trainer_mod, "_build_date_aware_splits", spy_splits)

    predictor.train(str(csv_path))

    assert predictor.trained is True
    assert observed_folds
    assert max(observed_folds) <= 2


def test_no_duplicate_candidate_refit_for_all_models(tmp_path, monkeypatch):
    """The winner is the ONLY full-data fit; no candidate is sample-refit to
    populate _all_models."""
    import core.nps_predictor.trainer as trainer_mod

    n_days = 90
    full_rows = n_days - 1

    config = _small_config()
    predictor = NPSPredictor(config=config)

    csv_path = tmp_path / "nps_train.csv"
    _write_training_csv(csv_path, n_days=n_days)

    full_size_fits = []

    def observe(X):
        if len(X) == full_rows:
            full_size_fits.append(len(X))

    def spy_clone(estimator):
        return _FitSpy(_real_clone(estimator), observe)

    monkeypatch.setattr(trainer_mod, "clone", spy_clone)

    predictor.train(str(csv_path))

    assert predictor.trained is True

    # Exactly one full-data fit on the complete dataset.
    assert len(full_size_fits) == 1
    assert full_size_fits == [full_rows]

    # No candidate ensemble/leaderboard models are retained post-training.
    assert set(predictor._all_models) == {predictor.model_name}

    # Selected model name and leaderboard MAE survive as metadata.
    assert predictor.model_name in predictor.algorithm_performance


def test_cv_parallelism_only_when_explicitly_requested(tmp_path, monkeypatch):
    """config.cv_n_jobs controls CV worker count; defaults to serial."""
    from core.nps_predictor.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG.cv_n_jobs == 1

    import core.nps_predictor.trainer as trainer_mod

    n_days = 90
    config = _small_config()
    config.cv_n_jobs = 2
    predictor = NPSPredictor(config=config)

    csv_path = tmp_path / "nps_train.csv"
    _write_training_csv(csv_path, n_days=n_days)

    parallel_workers = []

    class ParallelSpy:
        def __init__(self, *args, **kwargs):
            parallel_workers.append(kwargs.get("n_jobs"))
            self._impl = RealParallel(*args, **kwargs)

        def __call__(self, iterable):
            return self._impl(iterable)

    monkeypatch.setattr(trainer_mod, "Parallel", ParallelSpy)

    predictor.train(str(csv_path))

    assert predictor.trained is True
    assert parallel_workers == [2]


def test_prediction_works_after_winner_saved_and_reloaded(tmp_path):
    """Save/load round-trip; prediction still flows through the real engine."""
    n_days = 90
    config = _small_config()

    csv_path = tmp_path / "nps_train.csv"
    model_path = tmp_path / "nps_model.pkl"
    _write_training_csv(csv_path, n_days=n_days)

    trainer_predictor = NPSPredictor(config=config)
    trainer_predictor.train(str(csv_path))
    assert trainer_predictor.trained is True

    trainer_predictor.save_model(str(model_path))

    loaded = NPSPredictor(config=config)
    loaded.load_model(str(model_path))
    assert loaded.trained is True
    assert loaded.model is not None

    row = {
        "date": "2026-04-02",
        "operational_health": 90.0,
        "business_intelligence_factor": 10.0,
        "member_intelligence_factor": 10.0,
        "target_release_rate": 80.0,
        "actual_release_rate": 75.0,
        "total_calls_received": 2000,
        "total_surveys": 200,
        "survey_rate": 0.1,
        "target_quality": 80.0,
        "quality": 78.0,
        "target_competency": 85.0,
        "competency": 82.0,
        "target_attendance": 90.0,
        "attendance": 88.0,
        "target_transfer": 60.0,
        "transfer": 58.0,
    }
    result = loaded.predict(row)

    # score_counts / bayesian_score_distribution only exist on the REAL
    # engine path (fallback_predict does not emit them).
    assert isinstance(result, dict)
    for key in (
        "nps",
        "promoters",
        "passives",
        "detractors",
        "score_counts",
        "bayesian_score_distribution",
    ):
        assert key in result
    assert -100.0 <= result["nps"] <= 100.0
