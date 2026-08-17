"""
Behavioral tests for NPS training.

Exercise the real training lifecycle on a small synthetic dataset: training
completes, the model is trained, persistence round-trips, and selection
sampling respects the row-count limit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.nps_predictor.config import Config
from core.nps_predictor.predictor import NPSPredictor


def _write_training_csv(path, n_days=40, seed=7):
    rng = np.random.default_rng(seed)
    records = []
    for d in range(n_days):
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=d)
        for _ in range(3):
            rec = {
                "date": date,
                "operational_health": float(rng.integers(70, 95)),
                "business_intelligence_factor": float(rng.integers(0, 3)),
                "member_intelligence_factor": float(rng.integers(0, 3)),
                "target_quality": 87.0,
                "actual_quality": float(rng.integers(80, 92)),
                "target_competency": 93.0,
                "actual_competency": float(rng.integers(88, 96)),
                "target_attendance": 90.0,
                "actual_attendance": float(rng.integers(85, 95)),
                "target_release_rate": 60.0,
                "actual_release_rate": float(rng.integers(55, 70)),
                "target_transfer_rate": 9.0,
                "actual_transfer_rate": float(rng.integers(5, 12)),
                "total_calls_received": int(rng.integers(500, 2000)),
                "promoters": int(rng.integers(20, 60)),
                "passives": int(rng.integers(10, 30)),
                "detractors": int(rng.integers(5, 25)),
            }
            # A valid 0..10 score distribution summing to total_surveys.
            total = int(rec["total_calls_received"] * rec["actual_release_rate"] / 100.0 * 0.10)
            total = max(total, 11)
            rec["total_surveys"] = int(total)
            weights = rng.integers(1, 20, size=11).astype(float)
            weights /= weights.sum()
            counts = np.floor(weights * total).astype(int)
            counts[0] += int(total) - int(counts.sum())
            for i in range(11):
                rec[f"score_{i}"] = int(counts[i])
            records.append(rec)
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
        cv_folds=2,
        use_gpu=False,
        final_fit_auto_downscale=True,
    )


def test_training_lifecycle_and_persistence(tmp_path):
    """A full training run produces a trained, reloadable model."""
    csv_path = tmp_path / "train.csv"
    _write_training_csv(csv_path)
    predictor = NPSPredictor(config=_small_config())
    predictor.train(str(csv_path))
    assert predictor.trained is True
    assert predictor.model_name

    # Persistence / reload round-trip.
    out = tmp_path / "model.pkl"
    predictor.save_model(str(out))
    assert out.exists()
    loaded = NPSPredictor()
    loaded.load_model(str(out))
    assert loaded.trained is True
    assert loaded.model_name == predictor.model_name
    assert loaded.feature_names == predictor.feature_names


def test_selection_sampling_respects_row_limit(tmp_path, monkeypatch):
    """Selection CV must receive at most sample_size rows even when the
    dataset has more rows than the limit (row-count, not date-count)."""
    import core.nps_predictor.trainer as tr

    csv_path = tmp_path / "train.csv"
    _write_training_csv(csv_path, n_days=60)  # 60 days * 3 rows = 180 rows

    predictor = NPSPredictor(config=_small_config())
    observed = {"n_rows": None}

    original = tr.rolling_origin_train
    def _spy(*args, **kwargs):
        observed["n_rows"] = len(args[1]) if len(args) > 1 else None
        return original(*args, **kwargs)
    monkeypatch.setattr(tr, "rolling_origin_train", _spy)

    predictor.train(str(csv_path))
    # With 180 rows and sample_size=40, selection CV must be bounded to <=40.
    assert observed["n_rows"] is not None
    assert observed["n_rows"] <= 40
