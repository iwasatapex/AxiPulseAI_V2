"""
Focused tests for the AxiPulseAI V2 GUI service layer.

These validate that the GUI's thin service layer correctly DELEGATES to the
canonical V2 services and that the presentation serialisation (e.g. the
production prediction envelope) extracts values correctly. No model artifact
is trained or modified here; backends are stubbed/mocked.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest


# =====================================================================
# _detect_type
# =====================================================================

@pytest.mark.parametrize("name,head,expected", [
    ("data.csv", b"", "csv"),
    ("data.tsv", b"", "tsv"),
    ("data.json", b'{"a":1}', "json"),
    ("data.xlsx", b"", "excel"),
    ("model.pkl", b"", "model"),
    ("readme.txt", b"", "text"),
    ("mystery.bin", b"\x00\x01", "unknown"),
])
def test_detect_type(name, head, expected):
    from gui.services import _detect_type
    assert _detect_type(name, head) == expected


# =====================================================================
# _coerce
# =====================================================================

def test_coerce_basic():
    from gui.services import _coerce
    assert _coerce(1) == 1
    assert _coerce("x") == "x"
    assert _coerce(True) is True
    assert _coerce(1.5) == 1.5


def test_coerce_numpy_scalar():
    import numpy as np
    from gui.services import _coerce
    assert _coerce(np.float64(3.14)) == 3.14


# =====================================================================
# _safe_metrics
# =====================================================================

def test_safe_metrics_extracts_performance_and_history():
    from gui.services import _safe_metrics
    class P:
        algorithm_performance = {"RandomForest": {"mae": 0.5}}
        history_days = 42
    m = _safe_metrics(P())
    assert m["history_days"] == 42
    assert m["RandomForest"]["mae"] == 0.5


# =====================================================================
# _prediction_to_dict (the production envelope extraction bug fix)
# =====================================================================

def _make_prediction_result():
    def env(value, confidence, lo, hi):
        return SimpleNamespace(
            prediction=value,
            probabilistic=SimpleNamespace(
                confidence=confidence,
                likely_range_lower=lo,
                likely_range_upper=hi,
            ),
        )

    raw = SimpleNamespace(
        operations_health=82.0,
        nps=81.0,
        quality=87.0,
        competency=93.0,
        bayesian_score_distribution={5: 0.1, 6: 0.2},
        score_counts={"5": 10, "6": 20},
    )
    return SimpleNamespace(
        prediction=SimpleNamespace(
            raw=raw,
            operations_health=env(82.0, 0.91, 79.0, 85.0),
            nps=env(81.0, 0.87, 78.0, 84.0),
            bayesian_score_distribution={5: 0.1, 6: 0.2},
            score_counts={"5": 10, "6": 20},
        )
    )


def test_prediction_to_dict_extracts_envelope_values():
    from gui.services import _prediction_to_dict
    out = _prediction_to_dict(_make_prediction_result())
    assert out["operational_health"] == 82.0
    assert out["nps"] == 81.0
    assert out["oh_confidence"] == 0.91
    assert out["nps_confidence"] == 0.87
    assert out["oh_lower"] == 79.0
    assert out["oh_upper"] == 85.0
    assert out["bayesian_score_distribution"] == {5: 0.1, 6: 0.2}


def test_prediction_to_dict_handles_missing_envelope():
    from gui.services import _prediction_to_dict
    result = SimpleNamespace(
        prediction=SimpleNamespace(
            raw=SimpleNamespace(operations_health=None, nps=None),
            operations_health=None,
            nps=None,
            bayesian_score_distribution=None,
            score_counts=None,
        )
    )
    out = _prediction_to_dict(result)
    assert out["operational_health"] is None
    assert out["nps"] is None


# =====================================================================
# list_datasets (delegates to canonical list_training_files)
# =====================================================================

def test_list_datasets_reports_file_metadata(tmp_path, monkeypatch):
    from gui import services as svc

    f = tmp_path / "january_2026.csv"
    f.write_text("date,operational_health\n2026-01-01,90\n")
    monkeypatch.setattr(svc, "list_training_files", lambda: [f])

    ds = svc.list_datasets()
    assert len(ds) == 1
    row = ds[0]
    assert row["name"] == "january_2026.csv"
    assert row["stem"] == "january_2026"
    assert row["type"] == "csv"
    assert row["size_bytes"] == f.stat().st_size
    assert "modified" in row


def test_list_datasets_empty(tmp_path, monkeypatch):
    from gui import services as svc
    monkeypatch.setattr(svc, "list_training_files", lambda: [])
    assert svc.list_datasets() == []


# =====================================================================
# select_model_family (explicit; never silent)
# =====================================================================

def test_select_model_family_sets_active(monkeypatch):
    from gui import services as svc

    monkeypatch.setattr(svc, "validate_model_pair", lambda fam: (None, None))
    svc.STATE.set_active_family(None)
    res = svc.select_model_family("alpha")
    assert res["active_family"] == "alpha"
    assert svc.STATE.get_active_family() == "alpha"
    svc.STATE.set_active_family(None)


def test_select_model_family_rejects_missing(monkeypatch):
    from gui import services as svc
    from core.forecast_ai.prediction.model_selector import ModelPairError

    def _boom(fam):
        raise ModelPairError("not found")
    monkeypatch.setattr(svc, "validate_model_pair", _boom)
    svc.STATE.set_active_family(None)
    with pytest.raises(ModelPairError):
        svc.select_model_family("ghost")
    svc.STATE.set_active_family(None)


# =====================================================================
# list_scenarios
# =====================================================================

def test_list_scenarios_contains_baseline():
    from gui import services as svc
    scenarios = svc.list_scenarios()
    ids = [s["id"] for s in scenarios]
    assert "baseline" in ids


def test_list_scenarios_has_unique_ids_and_single_baseline():
    from gui import services as svc
    scenarios = svc.list_scenarios()
    ids = [s["id"] for s in scenarios]
    assert len(ids) == len(set(ids)), "scenario ids must be unique"
    assert ids.count("baseline") == 1, "exactly one baseline scenario expected"


def test_list_scenarios_excludes_disabled():
    from gui import services as svc
    from core.forecast_ai.scenarios.registry import ScenarioRegistry
    from core.forecast_ai.scenarios.models import Scenario

    ScenarioRegistry.register(Scenario(
        id="__gui_test_disabled__",
        name="Disabled Test",
        description="temporary",
        modifiers=[],
        enabled=False,
        priority=0,
    ))
    try:
        ids = [s["id"] for s in svc.list_scenarios()]
        assert "__gui_test_disabled__" not in ids
        exposed = {s["id"]: s for s in svc.list_scenarios(include_disabled=True)}
        assert exposed["__gui_test_disabled__"]["enabled"] is False
    finally:
        ScenarioRegistry.reset()


def test_scenario_registry_resets_cleanly():
    from core.forecast_ai.scenarios.registry import ScenarioRegistry
    ScenarioRegistry.reset()
    ids = [s.id for s in ScenarioRegistry.list()]
    assert "baseline" not in ids  # baseline is never registered


# =====================================================================
# KPI service-boundary validation
# =====================================================================

def test_find_target_state_rejects_out_of_range_release():
    from gui import services as svc
    with pytest.raises(ValueError, match="within"):
        svc.find_target_state({"release": 40.0})


def test_find_target_state_rejects_out_of_range_transfer():
    from gui import services as svc
    with pytest.raises(ValueError, match="within"):
        svc.find_target_state({"transfer": 30.0})


# =====================================================================
# system_health
# =====================================================================

def test_system_health_shape():
    from gui import services as svc
    health = svc.system_health()
    assert health["status"] in ("Ready", "Degraded", "Unavailable")
    assert "checks" in health
    assert "models" in health["checks"]
    assert "scenarios" in health["checks"]


# =====================================================================
# GUI view modules import cleanly
# =====================================================================

def test_gui_views_import():
    import gui.app  # noqa: F401
    import gui.components  # noqa: F401
    import gui.charts  # noqa: F401
    from gui.views import (  # noqa: F401
        adie_view, dashboard_view, forecast_view,
        models_view, predict_view, reverse_view, settings_view,
        target_state_view, train_view,
    )


def test_app_has_navigation():
    import gui.app as app
    assert set(app.NAV.keys()) == {
        "Dashboard", "Train", "Models", "Predict", "Forecast",
        "Target State", "Reverse Optimizer",
        "ADIE Decision", "Settings",
    }


# =====================================================================
# _inspect_model — engine_version fallback (NPS top-level vs OH metadata)
# =====================================================================

def test_inspect_model_engine_version_top_level(tmp_path):
    import joblib
    from gui.services import _inspect_model
    p = tmp_path / "model.pkl"
    joblib.dump({"model_name": "X", "trained": True, "engine_version": "2.1.0"}, p)
    info = _inspect_model(p)
    assert info["engine_version"] == "2.1.0"


def test_inspect_model_engine_version_falls_back_to_metadata(tmp_path):
    import joblib
    from gui.services import _inspect_model
    p = tmp_path / "model.pkl"
    # OH files store the version only inside metadata.
    joblib.dump({"model_name": "CatBoost", "trained": True,
                 "metadata": {"engine_version": "10.10"}}, p)
    info = _inspect_model(p)
    assert info["engine_version"] == "10.10"


# =====================================================================
# model path resolution (cwd-independent)
# =====================================================================

def test_train_models_uses_absolute_models_dir(tmp_path, monkeypatch):
    import core.nps_predictor.predictor as nps_mod
    import core.operation_health_predictor.predictor as oh_mod
    from gui import services as svc

    training_file = tmp_path / "alpha.csv"
    training_file.write_text(
        "actual_quality,actual_competency,actual_attendance,"
        "actual_release_rate,actual_transfer_rate,"
        "promoters,passives,detractors\n"
        "80,70,85,55,5,10,20,30\n"
    )

    saved = {}

    class StubOH:
        model_name = "CatBoost"
        feature_names = ["a", "b"]
        algorithm_performance = {"CatBoost": {"mae": 0.5}}
        history_days = 7
        def train(self, path):
            pass
        def save_model(self, path):
            saved["oh"] = str(path)

    class StubNPS:
        model_name = "RandomForest"
        feature_names = ["x", "y"]
        algorithm_performance = {"RandomForest": {"mae": 0.4}}
        history_days = 5
        def train(self, path):
            pass
        def save_model(self, path):
            saved["nps"] = str(path)

    monkeypatch.setattr(oh_mod, "OperationalHealthPredictor", StubOH)
    monkeypatch.setattr(nps_mod, "NPSPredictor", StubNPS)
    monkeypatch.setattr(svc, "list_training_files", lambda: [training_file])
    models_dir = tmp_path / "models"
    monkeypatch.setattr(svc, "MODELS_DIR", models_dir)

    out = svc.train_models("alpha.csv")

    assert out["oh_path"] == str(models_dir / "alpha_OH.pkl")
    assert out["nps_path"] == str(models_dir / "alpha_NPS.pkl")
    assert Path(out["oh_path"]).is_absolute()
    assert saved["oh"] == str(models_dir / "alpha_OH.pkl")
    assert saved["nps"] == str(models_dir / "alpha_NPS.pkl")
    # No accidental models/ dir is created in the arbitrary cwd: the save
    # path must be under the configured MODELS_DIR, never cwd-relative.
    assert str(models_dir) in out["oh_path"]


def test_train_models_releases_oh_before_nps_training(tmp_path, monkeypatch):
    """OH training memory must be released before NPS training starts.

    Regression guard for the 1M-row workflow: after OH training + save the
    OH predictor object must be garbage-collected before NPS training begins,
    so the two huge training states are never alive simultaneously.
    """
    import gc
    import weakref

    import core.nps_predictor.predictor as nps_mod
    import core.operation_health_predictor.predictor as oh_mod
    from gui import services as svc

    training_file = tmp_path / "alpha.csv"
    training_file.write_text(
        "actual_quality,actual_competency,actual_attendance,"
        "actual_release_rate,actual_transfer_rate,"
        "promoters,passives,detractors\n"
        "80,70,85,55,5,10,20,30\n"
    )

    state = {"oh_ref": None}

    class StubOH:
        model_name = "CatBoost"
        feature_names = ["a", "b"]
        algorithm_performance = {"CatBoost": {"mae": 0.5}}
        history_days = 7

        def train(self, path):
            pass

        def save_model(self, path):
            pass

    def _make_oh(*args, **kwargs):
        inst = StubOH()
        state["oh_ref"] = weakref.ref(inst)
        return inst

    class StubNPS:
        model_name = "RandomForest"
        feature_names = ["x", "y"]
        algorithm_performance = {"RandomForest": {"mae": 0.4}}
        history_days = 5

        def train(self, path):
            # OH predictor must already be del'd + collected before NPS
            # loads the same dataset for training.
            assert state["oh_ref"] is not None
            assert state["oh_ref"]() is None, (
                "OH predictor still alive when NPS training began"
            )

        def save_model(self, path):
            pass

    monkeypatch.setattr(oh_mod, "OperationalHealthPredictor", _make_oh)
    monkeypatch.setattr(nps_mod, "NPSPredictor", StubNPS)
    monkeypatch.setattr(svc, "list_training_files", lambda: [training_file])
    models_dir = tmp_path / "models"
    monkeypatch.setattr(svc, "MODELS_DIR", models_dir)

    out = svc.train_models("alpha.csv")

    assert out["oh_algorithm"] == "CatBoost"
    assert out["nps_algorithm"] == "RandomForest"
    # Nothing retains the OH predictor after train_models returns either.
    gc.collect()
    assert state["oh_ref"]() is None


def test_preview_dataset_uses_bounded_sample_not_full_load(tmp_path, monkeypatch):
    """Preview must never load the full training dataset (1M-row file)."""
    from gui import contracts as ct
    from gui import services as svc

    training_file = tmp_path / "alpha.csv"
    training_file.write_text(
        "actual_quality,actual_competency,actual_attendance,"
        "actual_release_rate,actual_transfer_rate,"
        "promoters,passives,detractors\n"
        "80,70,85,55,5,10,20,30\n"
    )

    calls = {"sample_rows": [], "full_calls": 0}
    orig_sample = ct.load_dataset_sample

    def _sample(path, n_rows=50_000):
        calls["sample_rows"].append(n_rows)
        return orig_sample(path, n_rows=n_rows)

    def _full(path):
        calls["full_calls"] += 1
        return orig_sample(path)

    monkeypatch.setattr(ct, "load_dataset_sample", _sample)
    monkeypatch.setattr(ct, "load_dataset", _full)
    monkeypatch.setattr(svc, "list_training_files", lambda: [training_file])

    out = svc.preview_dataset("alpha.csv")

    assert set(out["columns"]) == {
        "actual_quality", "actual_competency", "actual_attendance",
        "actual_release_rate", "actual_transfer_rate",
        "promoters", "passives", "detractors",
    }
    assert calls["full_calls"] == 0
    # The default preview must be a small bounded sample, never the full file.
    assert calls["sample_rows"] and max(calls["sample_rows"]) <= 5


def test_train_models_rejects_unsupported_format(tmp_path, monkeypatch):
    from gui import services as svc
    training_file = tmp_path / "alpha.xyz"
    training_file.write_text("junk")
    monkeypatch.setattr(svc, "list_training_files", lambda: [training_file])
    with pytest.raises(ValueError, match="unsupported format"):
        svc.train_models("alpha.xyz")


# =====================================================================
# Session isolation abstraction
# =====================================================================

def test_predict_activates_explicit_family_under_lock(monkeypatch):
    """The session's explicit family is what reaches the global provider.

    ``predict`` must set the provider to the *session-selected* family (not
    a stale global value) and run the pipeline while holding the lock.
    """
    from gui import services as svc

    activated = []
    calls = {"pipeline_run": 0}

    def _set_family(fam):
        activated.append(fam)

    class StubPipeline:
        def run(self, state):
            calls["pipeline_run"] += 1
            assert svc._PROVIDER_LOCK._is_owned()
            return _dummy_pipeline_result()

    monkeypatch.setattr(svc.PredictorProvider, "set_model_family", _set_family)
    monkeypatch.setattr(
        "core.forecast_ai.prediction.pipeline.ProductionPredictionPipeline",
        StubPipeline,
    )
    monkeypatch.setattr(svc, "validate_model_pair", lambda fam: (None, None))

    svc.STATE.reset()
    state = {
        "quality": 87.0, "competency": 93.0, "attendance": 90.0,
        "release": 60.0, "transfer": 9.0, "operations_health": 95.0,
        "nps": 82.0, "total_calls_received": 2000.0,
    }
    svc.predict(state, family="alpha")
    assert activated == ["alpha"]
    assert calls["pipeline_run"] == 1
    assert svc.STATE.get_active_family() == "alpha"
    svc.STATE.reset()


def _dummy_pipeline_result():
    from types import SimpleNamespace
    raw = SimpleNamespace(operations_health=None, nps=None)
    return SimpleNamespace(
        prediction=SimpleNamespace(
            raw=raw, operations_health=None, nps=None,
            bayesian_score_distribution=None, score_counts=None,
        )
    )


def test_state_is_session_scoped_and_reset():
    from gui import services as svc
    svc.STATE.reset()
    assert svc.STATE.get_active_family() is None
    svc.STATE.set_active_family("zeta")
    assert svc.STATE.get_active_family() == "zeta"
    svc.STATE.reset()
    assert svc.STATE.get_active_family() is None

