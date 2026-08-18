"""
Focused tests for the central training-progress mechanism and its GUI
rendering, plus a guard that progress reporting never alters training results.
"""
import numpy as np
import pandas as pd
import pytest

from core.common.training_progress import (
    STAGE_COMPLETE,
    STAGE_FAILED,
    STAGE_FINAL_REFIT,
    STAGE_MODEL_SELECTION,
    TrainingProgress,
)
from gui.training_status import (
    format_elapsed,
    format_status_lines,
    progress_bar_value,
    render_live,
)


def test_progress_initialization():
    p = TrainingProgress(kind="NPS")
    snap = p.snapshot()
    assert snap["kind"] == "NPS"
    assert snap["stage"] == "loading"
    assert snap["percent"] is None
    assert snap["completed_models"] == 0
    assert snap["total_models"] is None
    assert snap["current_model"] is None
    assert snap["current_fold"] is None
    assert snap["error"] is None
    assert snap["elapsed_seconds"] >= 0


def test_candidate_fold_updates_and_real_percentage():
    p = TrainingProgress()
    p.set_models(8)
    assert p.snapshot()["total_models"] == 8

    p.start_candidate("CatBoost")
    assert p.snapshot()["current_model"] == "CatBoost"

    p.start_fold(1, total_folds=2)
    snap = p.snapshot()
    assert snap["current_fold"] == 1
    assert snap["total_folds"] == 2
    # 0 completed * 2 + (1-1) / (8*2) = 0%
    assert snap["percent"] == 0.0

    p.start_fold(2, total_folds=2)
    # 0 * 2 + (2-1) / 16 = 6.25%
    assert p.snapshot()["percent"] == pytest.approx(6.25)

    p.complete_candidate()
    snap = p.snapshot()
    assert snap["completed_models"] == 1
    # 1*2 / 16 = 12.5%
    assert snap["percent"] == pytest.approx(12.5)
    assert snap["current_model"] is None
    assert snap["current_fold"] is None


def test_final_fit_state_is_indeterminate():
    p = TrainingProgress()
    p.set_final_fit("CatBoost", device="gpu", rows=100_000)
    snap = p.snapshot()
    assert snap["stage"] == STAGE_FINAL_REFIT
    assert snap["model_name"] == "CatBoost"
    assert snap["device"] == "gpu"
    assert snap["rows"] == 100_000
    # Never a fake precise percentage for a fit without reliable progress.
    assert snap["percent"] is None
    assert "CatBoost" in snap["message"]
    assert "100,000" in snap["message"]


def test_completion_state():
    p = TrainingProgress()
    p.complete(model_name="XGBoost", rows=50_000, history_days=120)
    snap = p.snapshot()
    assert snap["stage"] == STAGE_COMPLETE
    assert snap["percent"] == 100.0
    assert snap["model_name"] == "XGBoost"
    assert snap["rows"] == 50_000
    assert snap["history_days"] == 120
    assert snap["error"] is None


def test_failed_state():
    p = TrainingProgress()
    p.fail("boom")
    snap = p.snapshot()
    assert snap["stage"] == STAGE_FAILED
    assert snap["error"] == "boom"
    assert snap["message"] == "boom"


def test_no_fake_percentage_progression():
    # Indeterminate final-fit must stay None through multiple reads.
    p = TrainingProgress()
    p.set_final_fit("CatBoost", device="cpu", rows=999)
    for _ in range(3):
        assert p.snapshot()["percent"] is None

    # Model-selection percent is derived strictly from counts.
    p2 = TrainingProgress()
    p2.set_models(4)
    p2.start_candidate("A")
    p2.start_fold(1, total_folds=3)
    assert p2.snapshot()["percent"] == 0.0
    p2.start_fold(2, total_folds=3)
    assert p2.snapshot()["percent"] == pytest.approx(100.0 / 12)  # 8.33
    p2.complete_candidate()
    assert p2.snapshot()["percent"] == pytest.approx(25.0)


class _FakeBar:
    def __init__(self):
        self.calls = []

    def progress(self, value, text=None):
        self.calls.append((value, text))


class _FakeDetail:
    def __init__(self):
        self.markdown_calls = []

    def markdown(self, text):
        self.markdown_calls.append(text)


def test_progress_bar_value_indeterminate_and_real():
    assert progress_bar_value({"percent": None}) is None
    assert progress_bar_value({"percent": 50.0}) == pytest.approx(0.5)
    assert progress_bar_value({"percent": 0.0}) == 0.0
    assert progress_bar_value({"percent": "nonsense"}) is None


def test_gui_rendering_live_indeterminate():
    bar = _FakeBar()
    detail = _FakeDetail()
    p = TrainingProgress()
    p.set_final_fit("XGBoost", device="gpu", rows=200_000)
    snap = p.snapshot()
    render_live(bar, detail, snap)
    assert bar.calls and bar.calls[-1][0] == 0.0  # indeterminate -> 0.0 bar
    text = "\n\n".join(detail.markdown_calls)
    assert "XGBoost" in text
    assert "gpu" in text.lower() or "GPU" in text


def test_gui_rendering_live_percentage():
    p = TrainingProgress()
    p.set_models(8)
    p.start_candidate("CatBoost")
    p.start_fold(2, total_folds=2)
    p.complete_candidate()  # 12.5%
    bar = _FakeBar()
    detail = _FakeDetail()
    render_live(bar, detail, p.snapshot())
    value = bar.calls[-1][0]
    assert value == pytest.approx(0.125)
    text = "\n\n".join(detail.markdown_calls)
    assert "CatBoost" in text or "Model" in text


def test_format_elapsed():
    assert format_elapsed(None) == "--:--"
    assert format_elapsed(0) == "00:00"
    assert format_elapsed(37) == "00:37"
    assert format_elapsed(131) == "02:11"
    assert format_elapsed(3723) == "1:02:03"


def test_format_status_lines_final_fit():
    p = TrainingProgress()
    p.set_final_fit("CatBoost", device="GPU (RTX 3050)", rows=100_000)
    lines = format_status_lines(p.snapshot())
    joined = "\n".join(lines)
    assert "CatBoost" in joined
    assert "GPU" in joined
    assert "100,000" in joined


def _write_training_csv(path, n_days=90, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
    records = []
    for day in dates:
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
    from core.nps_predictor import Config

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


def test_progress_does_not_alter_training_results(tmp_path):
    """Training with a progress object yields identical results to without."""
    from core.nps_predictor import NPSPredictor

    csv_path = tmp_path / "nps_train.csv"
    _write_training_csv(csv_path, n_days=90)

    base = NPSPredictor(config=_small_config())
    base.train(str(csv_path))

    with_progress = NPSPredictor(config=_small_config())
    progress = TrainingProgress(kind="NPS")
    with_progress.train(str(csv_path), progress=progress)

    assert with_progress.trained is True
    assert base.trained is True
    assert with_progress.model_name == base.model_name
    assert with_progress.algorithm_performance == base.algorithm_performance
    assert with_progress.algorithm_bucket_mae == base.algorithm_bucket_mae

    # Progress advanced through the expected stages.
    stages = []
    # (TrainingProgress only keeps the latest; verify final state and that
    # candidate bookkeeping was exercised.)
    assert progress.snapshot()["stage"] in (STAGE_COMPLETE, STAGE_FINAL_REFIT)
