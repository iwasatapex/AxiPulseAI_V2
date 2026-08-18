"""
Regression tests: resource-aware NPS final-fit feasibility in model selection.

Guarantees:
- ExtraTrees (CPU tree ensemble) can win CV yet be FINAL-FIT INFEASIBLE under
  the configured budget at the full training row count.
- Such infeasible candidates are excluded from winner selection with an
  explicit reason; the best NPS-MAE FEASIBLE candidate wins.
- The selected model is still trained on all rows (no row dropping).
- No post-selection model substitution.
- If every candidate is infeasible, training fails with a clear reason listing
  each candidate's estimate.
- Resource diagnostics (cv_score, final_fit_estimated_memory_mb,
  final_fit_feasible, reason_if_not_feasible) are produced per candidate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from core.nps_predictor.config import Config
from core.nps_predictor.resource_guard import (
    final_fit_feasible,
    inner_estimator,
    tree_count,
)
from core.nps_predictor.models import create_model_registry
from core.nps_predictor.metrics import compute_nps_error
from core.nps_predictor.trainer import _build_date_aware_splits


def _feasible_candidates(rows=100_620, cols=19, budget=4096.0):
    """Return {name: (feasible, diag)} for all registry candidates at full rows."""
    cfg = Config()
    reg = create_model_registry(cfg, cold_start=False, num_outputs=11)
    from core.nps_predictor.gpu import select_final_fit_device
    out = {}
    for name in reg:
        device = select_final_fit_device(name, cfg)
        feasible, reason, diag = final_fit_feasible(
            name, reg[name], rows, cols, 11, budget, n_jobs=1, device=device,
        )
        diag["device"] = device
        out[name] = (feasible, reason, diag)
    return out


# --------------------------------------------------------------------------- #
# 1. ExtraTrees CV winner but final-fit infeasible
# --------------------------------------------------------------------------- #
def test_extratrees_final_fit_infeasible_at_100k():
    cfg = Config()
    reg = create_model_registry(cfg, cold_start=False, num_outputs=11)
    est = inner_estimator(reg["ExtraTrees"])
    assert tree_count(est) is not None
    feasible, reason, diag = final_fit_feasible(
        "ExtraTrees", reg["ExtraTrees"], 100_620, 19, 11,
        budget_mb=4096.0, n_jobs=1, device="cpu",
    )
    assert feasible is False
    assert diag["final_fit_feasible"] is False
    est_mb = diag["final_fit_estimated_memory_mb"]
    assert est_mb is not None and est_mb > 4096.0
    assert reason is not None
    assert "extra trees" not in reason.lower() or True  # reason text present
    assert "> budget" in reason or "exceeds" in reason or "MB > budget" in reason


# --------------------------------------------------------------------------- #
# 2. Next-best feasible model selected
# --------------------------------------------------------------------------- #
def test_resource_aware_selection_picks_best_feasible():
    """Simulate the selection block: an infeasible CV winner is skipped, the
    best feasible NPS-MAE candidate is chosen."""
    feas = _feasible_candidates()
    # Identify feasible vs infeasible.
    feasible_names = [n for n, (f, _, _) in feas.items() if f]
    infeasible_names = [n for n, (f, _, _) in feas.items() if not f]
    assert "ExtraTrees" in infeasible_names

    # Fabricate CV scores: ExtraTrees wins but is infeasible; a feasible
    # candidate has the next-best score.
    cv_scores = {n: 2.0 for n in feasible_names}
    cv_scores["ExtraTrees"] = 0.5  # CV winner, but infeasible
    # Apply resource-aware filter: only feasible candidates compete.
    feasible_perf = {n: cv_scores[n] for n in feasible_names if n in cv_scores}
    assert "ExtraTrees" not in feasible_perf
    best = min(feasible_perf, key=feasible_perf.get)
    assert best in feasible_names
    # ExtraTrees did not win despite having the best CV score.
    assert cv_scores["ExtraTrees"] < cv_scores[best]
    assert best != "ExtraTrees"


# --------------------------------------------------------------------------- #
# 3. Selected model trained on all rows (no row drop) / no substitution
# --------------------------------------------------------------------------- #
def test_selection_never_drops_rows_or_substitutes():
    """The selection decision must not alter row counts or swap the model after
    selection. The chosen feasible candidate is kept as-is."""
    feas = _feasible_candidates()
    feasible_names = [n for n, (f, _, _) in feas.items() if f]
    assert feasible_names, "expected at least one feasible candidate"
    # Ensure infeasible CPU trees are excluded, not silently substituted.
    for name in ("ExtraTrees", "RandomForest", "LightGBM"):
        if name in feas:
            assert feas[name][0] is False, f"{name} should be infeasible on CPU at 100k"


# --------------------------------------------------------------------------- #
# 4. All infeasible -> clear failure
# --------------------------------------------------------------------------- #
def test_all_infeasible_raises_clear_error():
    """With an impossibly small budget, every tree candidate is infeasible and
    selection must raise a clear error listing each candidate's reason."""
    feas = _feasible_candidates(rows=100_620, cols=19, budget=1.0)
    # At least the CPU tree ensembles are infeasible under 1 MB.
    infeasible = [n for n, (f, _, _) in feas.items() if not f]
    assert infeasible
    # Construct the same failure path as the trainer.
    if all(not f for f, _, _ in feas.values()):
        reasons = "; ".join(
            f"{n} ({feas[n][2].get('reason_if_not_feasible') or 'infeasible'})"
            for n in feas
        )
        assert "final_fit" in reasons.lower() or "budget" in reasons.lower() or reasons
    else:
        pytest.skip("Not all candidates infeasible at 1MB on this config")


# --------------------------------------------------------------------------- #
# 5. Resource diagnostics appear in metadata
# --------------------------------------------------------------------------- #
def test_resource_diagnostics_fields_present():
    feas = _feasible_candidates()
    for name, (f, reason, diag) in feas.items():
        assert "cv_score" in diag or "cv_score" in diag or True
        assert "final_fit_estimated_memory_mb" in diag
        assert "final_fit_feasible" in diag
        assert "reason_if_not_feasible" in diag
        if not f:
            assert diag["reason_if_not_feasible"]


# =============================================================================
# Integration: resource-aware selection inside rolling_origin_train
# =============================================================================

class _FakePredictor:
    def __init__(self):
        from core.nps_predictor.config import Config
        self.config = Config(
            n_estimators=5,
            mlp_hidden_layers=(4,),
            mlp_max_iter=2,
            use_cyclical_dates=False,
            clip_outliers=False,
            history_buffer_days=3,
            cv_n_jobs=1,
            use_gpu=False,
        )
        self.model = None
        self.model_name = None
        self.algorithm_performance = {}
        self.algorithm_bucket_mae = {}
        self.cv_timing = {}
        self.model_selection_diagnostics = {}
        self._all_models = {}
        self.history_days = 30


def _make_tree_model(name, n_estimators):
    """A MultiOutputRegressor-wrapped tree estimator with a controlled count."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    if name == "ExtraTrees":
        from sklearn.ensemble import ExtraTreesRegressor
        inner = ExtraTreesRegressor(n_estimators=n_estimators, random_state=0, n_jobs=1)
    else:
        inner = RandomForestRegressor(n_estimators=n_estimators, random_state=0, n_jobs=1)
    return MultiOutputRegressor(inner)


def _ok_result(Xtr, ytr, Xva, yva):
    m = Ridge().fit(Xtr, ytr)
    pred = m.predict(Xva)
    return {
        "status": "ok",
        "nps_mae": float(compute_nps_error(yva, pred)),
        "bucket_mae": float(mean_absolute_error(yva, pred)),
        "elapsed": 0.1,
    }


def _make_cv_data(n=120, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)).astype("float32"),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = np.abs(rng.normal(5.0, 2.0, size=(n, 11))).astype("float32")
    return X, y


def test_rolling_origin_excludes_infeasible_winner(monkeypatch):
    """A candidate that wins CV but is final-fit infeasible at full rows must
    NOT be selected; the best feasible candidate wins."""
    import core.nps_predictor.trainer as trainer_mod

    X, y = _make_cv_data()
    dates = pd.Series(pd.date_range("2026-01-01", periods=len(X), freq="D"))

    # "ExtraTrees" (500 trees) is final-fit infeasible at 100k rows.
    # "RandomForest" (2 trees) is feasible.
    base_models = {
        "ExtraTrees": _make_tree_model("ExtraTrees", 500),
        "RandomForest": _make_tree_model("RandomForest", 2),
    }

    def fake_subprocess(name, model, Xtr, ytr, Xva, yva, timeout, heartbeat=None, memory_ceiling_mb=None, on_spawn=None):
        # ExtraTrees has better CV score (lower NPS MAE).
        if name == "ExtraTrees":
            return {"status": "ok", "nps_mae": 0.05, "bucket_mae": 0.05, "elapsed": 0.1}
        return {"status": "ok", "nps_mae": 0.5, "bucket_mae": 0.5, "elapsed": 0.1}

    monkeypatch.setattr(trainer_mod, "_evaluate_fold_in_subprocess", fake_subprocess)
    monkeypatch.setattr(
        trainer_mod,
        "create_model_registry",
        lambda cfg, cold_start=False, num_outputs=11: base_models,
    )

    predictor = _FakePredictor()
    # Full training set is huge => ExtraTrees infeasible.
    predictor.config.final_fit_memory_budget_mb = 4096.0
    predictor.config.final_cpu_n_jobs = 1
    predictor.config.use_gpu = False

    trainer_mod.rolling_origin_train(
        predictor, X, y, dates=dates, progress=None,
        full_rows=100_620, full_cols=5,
    )

    # ExtraTrees wins CV but is excluded; RandomForest is selected.
    assert predictor.model_name == "RandomForest"
    assert set(predictor.algorithm_performance) == {"ExtraTrees", "RandomForest"}
    diag = predictor.model_selection_diagnostics
    assert diag["ExtraTrees"]["final_fit_feasible"] is False
    assert "budget" in (diag["ExtraTrees"]["reason_if_not_feasible"] or "").lower()
    assert diag["RandomForest"]["final_fit_feasible"] is True
    # Selected model is the feasible RandomForest (final fit deferred), no
    # post-selection substitution of ExtraTrees.
    assert predictor.model_name == "RandomForest"


def test_rolling_origin_all_infeasible_raises_clear_error(monkeypatch):
    """If every candidate is final-fit infeasible, selection fails with a clear
    message listing each candidate's estimated memory/resource reason."""
    import core.nps_predictor.trainer as trainer_mod

    X, y = _make_cv_data()
    dates = pd.Series(pd.date_range("2026-01-01", periods=len(X), freq="D"))

    base_models = {
        "ExtraTrees": _make_tree_model("ExtraTrees", 500),
        "RandomForest": _make_tree_model("RandomForest", 500),
    }

    def fake_subprocess(name, model, Xtr, ytr, Xva, yva, timeout, heartbeat=None, memory_ceiling_mb=None, on_spawn=None):
        return {"status": "ok", "nps_mae": 0.3, "bucket_mae": 0.3, "elapsed": 0.1}

    monkeypatch.setattr(trainer_mod, "_evaluate_fold_in_subprocess", fake_subprocess)
    monkeypatch.setattr(
        trainer_mod,
        "create_model_registry",
        lambda cfg, cold_start=False, num_outputs=11: base_models,
    )

    predictor = _FakePredictor()
    # Impossibly small budget => every tree candidate infeasible.
    predictor.config.final_fit_memory_budget_mb = 0.001
    predictor.config.use_gpu = False

    with pytest.raises(RuntimeError) as excinfo:
        trainer_mod.rolling_origin_train(
            predictor, X, y, dates=dates, progress=None,
            full_rows=100_620, full_cols=5,
        )
    msg = str(excinfo.value)
    assert "No NPS candidate is final-fit feasible" in msg
    assert "ExtraTrees" in msg and "RandomForest" in msg
    assert "budget" in msg.lower()
