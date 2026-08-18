"""Behavioral regression: GUI state invalidation, leaderboard metric, role
isolation, and legacy-model isolation.

Proves:
* switching model family invalidates stale results from a prior family
* the leaderboard shows the SELECTED algorithm's metric, not a cross-algorithm
  min
* production/test families are separated using explicit role metadata
* default production inference cannot resolve to the legacy artifact
"""
from __future__ import annotations

import json
import hashlib
import json
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_production_manifest(models_dir: Path, source: str = "1mil-10yr") -> None:
    from core.forecast_ai.prediction import production_registry as pr

    oh = models_dir / "production_OH.pkl"
    nps = models_dir / "production_NPS.pkl"
    manifest = {
        oh.name: {"sha256": _file_hash(oh), "source": source, "role": "production"},
        nps.name: {"sha256": _file_hash(nps), "source": source, "role": "production"},
    }
    (models_dir / pr.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


class _FakeEstimator:
    n_features_in_ = 3
    n_outputs_ = 1
    def predict(self, X):
        return [0.0] * len(X)


class _FakeNPSEstimator:
    n_features_in_ = 34
    n_outputs_ = 11
    def predict(self, X):
        return [[0.0] * 11 for _ in range(len(X))]


def _write_production_bundles(models_dir: Path) -> None:
    """Write valid, loadable joblib production bundles (fail-closed integrity
    verification requires real deserialisable artifacts).  NPS is 11-output."""
    import joblib
    joblib.dump(
        {
            "model_name": "CatBoost", "trained": True,
            "feature_names": ["f1", "f2", "f3"], "model": _FakeEstimator(),
            "metadata": {"training_rows": 100},
        },
        str(models_dir / "production_OH.pkl"),
    )
    joblib.dump(
        {
            "model_name": "XGBoost", "trained": True,
            "feature_names": [f"nf{i}" for i in range(34)], "model": _FakeNPSEstimator(),
            "metadata": {"training_rows": 100, "num_scores": 11},
        },
        str(models_dir / "production_NPS.pkl"),
    )


# --------------------------------------------------------------------------- #
# GUI state: family switch invalidates stale results
# --------------------------------------------------------------------------- #

def test_family_switch_invalidates_prior_family_result(monkeypatch):
    from gui import services as svc
    from gui import state as gui_state

    store = {}
    monkeypatch.setattr(gui_state, "_store", lambda: store)

    svc.STATE.set_active_family("alpha")
    # Simulate a stored result produced under family alpha.
    svc.STATE.set_last_prediction({"active_family": "alpha", "nps": 50.0})
    assert svc.STATE.get_last_prediction() is not None

    # Switch to family beta: the alpha result must be invalidated.
    svc.STATE.set_active_family("beta")
    assert svc.STATE.get_last_prediction() is None
    svc.STATE.reset()


def test_family_switch_preserves_result_for_same_family(monkeypatch):
    from gui import services as svc
    from gui import state as gui_state

    store = {}
    monkeypatch.setattr(gui_state, "_store", lambda: store)

    svc.STATE.set_active_family("alpha")
    svc.STATE.set_last_forecast({"active_family": "alpha", "nps": 50.0})
    # Same family re-set must NOT clear the valid result.
    svc.STATE.set_active_family("alpha")
    assert svc.STATE.get_last_forecast() is not None
    svc.STATE.reset()


def test_family_switch_clears_forecast_and_adie(monkeypatch):
    from gui import services as svc
    from gui import state as gui_state

    store = {}
    monkeypatch.setattr(gui_state, "_store", lambda: store)

    svc.STATE.set_active_family("alpha")
    svc.STATE.set_last_forecast({"active_family": "alpha"})
    svc.STATE.set_last_adie({"family": "alpha"})
    svc.STATE.set_active_family("gamma")
    assert svc.STATE.get_last_forecast() is None
    assert svc.STATE.get_last_adie() is None
    svc.STATE.reset()


# --------------------------------------------------------------------------- #
# Leaderboard: selected-model metric is correct
# --------------------------------------------------------------------------- #

def test_leaderboard_shows_selected_algorithm_metric(monkeypatch):
    """The leaderboard must display the metric belonging to the SELECTED
    algorithm, never ``min(all_algorithm_metrics)``."""
    from gui.views import models_view

    info = {"model_name": "RandomForest", "mae": {"XGBoost": 1.0, "RandomForest": 3.5}}
    # Selected algorithm is RandomForest -> 3.5, not the min 1.0.
    assert models_view._fmt_selected_mae(info["mae"], "RandomForest") == "3.50"
    assert models_view._fmt_selected_mae(info["mae"], "XGBoost") == "1.00"


def test_leaderboard_selected_metric_not_cross_algorithm_min(monkeypatch):
    """Guard: the displayed metric must NOT be the cross-algorithm min."""
    from gui.views import models_view

    info = {"model_name": "CatBoost", "mae": {"XGBoost": 0.5, "CatBoost": 4.0}}
    displayed = models_view._fmt_selected_mae(info["mae"], "CatBoost")
    assert displayed == "4.00"
    assert displayed != "0.50"  # must not show another algorithm's min


def test_leaderboard_dict_fallback_lists_all(monkeypatch):
    from gui.views import models_view
    info = {"model_name": "X", "mae": {"A": 1.0, "B": 2.0}}
    out = models_view._fmt_selected_mae(info["mae"], "Z")  # unknown algorithm
    assert "A: 1.00" in out and "B: 2.00" in out


def test_family_detail_metric_uses_selected_algorithm(monkeypatch):
    """Regression (task 10): the Family-Details metric must be the SELECTED
    algorithm's, not the cross-algorithm minimum.

    Algorithm A MAE = 10, Algorithm B MAE = 20, selected = B -> display 20."""
    from gui.views import models_view

    # This is the exact scenario: selected model is B (the worse MAE), and the
    # displayed metric must be B's own 20, never min(10,20)=10.
    assert models_view._fmt_selected_mae({"A": 10.0, "B": 20.0}, "B") == "20.00"
    assert models_view._fmt_selected_mae({"A": 10.0, "B": 20.0}, "B") != "10.00"
    assert models_view._fmt_selected_mae({"A": 10.0, "B": 20.0}, "A") == "10.00"


# --------------------------------------------------------------------------- #
# Production/test family isolation via explicit role metadata
# --------------------------------------------------------------------------- #

def test_test_family_uses_manifest_role_metadata(tmp_path, monkeypatch):
    """Explicit role metadata (manifest) is authoritative over substring
    heuristics for production/test separation."""
    from gui import model_selection as ms
    from core.forecast_ai.prediction import production_registry as pr

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "production_OH.pkl").write_bytes(b"x")
    (models_dir / "production_NPS.pkl").write_bytes(b"y")
    (models_dir / pr.MANIFEST_NAME).write_text(json.dumps({
        "production_OH.pkl": {"role": "production", "sha256": "0" * 64},
        "production_NPS.pkl": {"role": "production", "sha256": "1" * 64},
    }))

    monkeypatch.setattr(pr, "MODELS_DIR", models_dir)

    # production family classified as non-test regardless of substrings.
    assert ms.is_test_family("production") is False
    # An explicitly-marked stress family is hidden even without the marker in
    # its name.
    (models_dir / "alpha_OH.pkl").write_bytes(b"a")
    (models_dir / "alpha_NPS.pkl").write_bytes(b"b")
    (models_dir / pr.MANIFEST_NAME).write_text(json.dumps({
        "alpha_OH.pkl": {"role": "test", "sha256": "2" * 64},
        "alpha_NPS.pkl": {"role": "test", "sha256": "3" * 64},
    }))
    assert ms.is_test_family("alpha") is True


def test_test_family_falls_back_to_markers_without_manifest(tmp_path, monkeypatch):
    from gui import model_selection as ms
    from core.forecast_ai.prediction import production_registry as pr

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(pr, "MODELS_DIR", models_dir)
    # No manifest -> substring heuristic still applied.
    assert ms.is_test_family("smoke_1") is True
    assert ms.is_test_family("1mil-10yr") is False


# --------------------------------------------------------------------------- #
# Legacy model isolation (Task F)
# --------------------------------------------------------------------------- #

def test_default_production_inference_cannot_resolve_to_legacy(tmp_path, monkeypatch):
    """Default production inference must target production_OH.pkl /
    production_NPS.pkl — never silently fall back to the legacy
    operation_health_predictor.joblib / nps_predictor_model.pkl."""
    from core.forecast_ai.prediction import predictor_config as pc

    # Only legacy filenames exist; no production artifacts.
    (tmp_path / pc.OH_LEGACY).write_bytes(b"legacy-oh")
    (tmp_path / pc.NPS_LEGACY).write_bytes(b"legacy-nps")
    monkeypatch.setattr(pc, "MODELS", tmp_path)

    with pytest.raises(FileNotFoundError):
        pc.create_oh_predictor()
    with pytest.raises(FileNotFoundError):
        pc.create_nps_predictor()

    # Once canonical production artifacts exist, they are used. A valid
    # integrity manifest is REQUIRED for production loading (fail closed).
    _write_production_bundles(tmp_path)
    _write_production_manifest(tmp_path)
    seen = {"oh": None, "nps": None}
    monkeypatch.setattr(
        pc, "OperationalHealthPredictor",
        lambda: types.SimpleNamespace(load_model=lambda p: seen.__setitem__("oh", p)),
    )
    monkeypatch.setattr(
        pc, "NPSPredictor",
        lambda: types.SimpleNamespace(load_model=lambda p: seen.__setitem__("nps", p)),
    )
    pc.create_oh_predictor()
    pc.create_nps_predictor()
    assert str(seen["oh"]).endswith("production_OH.pkl")
    assert str(seen["nps"]).endswith("production_NPS.pkl")
