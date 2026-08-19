"""
Focused tests for the central GUI model-selection component.

Covers the required model-selection behaviors:
  * dynamic discovery (complete / valid / loadable pairs only)
  * incompatible pairs excluded (error entries, unreadable bundles, untrained)
  * smoke/test/staging excluded by default, included only when enabled
  * zero / one / multiple model states
  * explicit selection passed into the service layer (no silent fallback)
  * per-feature session persistence
"""
import joblib
import pytest

from core.forecast_ai.prediction.model_selector import ModelPairError
from gui import model_selection as ms
from gui import services as svc
from gui.state import STATE


# ============================================================
# Fixtures / helpers
# ============================================================

@pytest.fixture(autouse=True)
def _clean_gui_state():
    STATE.reset()
    ms.reset_feature_selections()
    yield
    STATE.reset()
    ms.reset_feature_selections()


def _entry(family, trained=True, oh_name="CatBoost", nps_name="LightGBM",
           oh_error=None, nps_error=None, error=None):
    """Build one raw ``svc.list_models()``-shaped entry."""
    oh = {"path": f"models/{family}_OH.pkl", "model_name": oh_name,
          "feature_count": 5, "feature_names_sample": [], "trained": trained,
          "engine_version": "3.0"}
    nps = {"path": f"models/{family}_NPS.pkl", "model_name": nps_name,
           "feature_count": 6, "feature_names_sample": [], "trained": trained,
           "engine_version": "3.0"}
    if oh_error:
        oh["error"] = oh_error
    if nps_error:
        nps["error"] = nps_error
    entry = {
        "family": family,
        "oh_path": f"models/{family}_OH.pkl",
        "nps_path": f"models/{family}_NPS.pkl",
        "oh": oh,
        "nps": nps,
        "saved_at": "2026-08-16T00:00:00",
        "active": False,
    }
    if error:
        entry["error"] = error
    return entry


@pytest.fixture
def models_dir(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    return d


def _bundle(path, kind, family):
    joblib.dump(
        {
            "model_name": f"{kind}_{family}",
            "trained": True,
            "feature_names": ["f1", "f2"],
            "algorithm_performance": {"mae": 0.5},
            "metadata": {"engine_version": "1.0", "family": family},
            "all_models": {},
        },
        path,
    )


def _make_family_pair(models_dir, family):
    _bundle(models_dir / f"{family}_OH.pkl", "oh", family)
    _bundle(models_dir / f"{family}_NPS.pkl", "nps", family)


# ============================================================
# Test/staging classification
# ============================================================

def test_is_test_family_detects_smoke_test_staging_and_hidden():
    assert ms.is_test_family("smoke test") is True
    assert ms.is_test_family(".training") is True
    assert ms.is_test_family("staging_prod") is True
    assert ms.is_test_family("tmp_models") is True
    assert ms.is_test_family("january_2026") is False
    assert ms.is_test_family(None) is False


# ============================================================
# Discovery: compatible / valid / loadable only
# ============================================================

def test_discover_returns_complete_valid_pairs():
    options = ms.discover_models(models=[
        _entry("alpha"), _entry("beta", oh_name="XGBoost", nps_name="CatBoost"),
    ])
    assert [o.family for o in options] == ["alpha", "beta"]
    assert options[0].oh_algorithm == "CatBoost"
    assert options[1].nps_algorithm == "CatBoost"


def test_discover_excludes_test_by_default():
    options = ms.discover_models(models=[
        _entry("prod_a"), _entry("smoke test"),
    ])
    assert [o.family for o in options] == ["prod_a"]


def test_discover_include_test_reveals_smoke():
    options = ms.discover_models(
        models=[_entry("prod_a"), _entry("smoke test")], include_test=True
    )
    assert [o.family for o in options] == ["prod_a", "smoke test"]
    assert any(o.is_test for o in options)


def test_discover_skips_error_entries():
    options = ms.discover_models(models=[
        _entry("bad_pair", error="missing NPS file"),
        _entry("bad_oh", oh_error="cannot inspect"),
        _entry("bad_nps", nps_error="cannot inspect"),
        _entry("untrained", trained=False),
        _entry("good"),
    ])
    assert [o.family for o in options] == ["good"]


def test_discover_zero_models():
    assert ms.discover_models(models=[]) == []


def test_discover_single_model():
    options = ms.discover_models(models=[_entry("solo")])
    assert [o.family for o in options] == ["solo"]


def test_discover_multiple_sorted_by_family():
    options = ms.discover_models(models=[
        _entry("zeta"), _entry("alpha"), _entry("mike"),
    ])
    assert [o.family for o in options] == ["alpha", "mike", "zeta"]


def test_discover_uses_live_listing(monkeypatch):
    monkeypatch.setattr(
        "gui.services.list_models", lambda: [_entry("live_prod")]
    )
    options = ms.discover_models()
    assert [o.family for o in options] == ["live_prod"]


# ============================================================
# Performance metrics (display only)
# ============================================================

def test_performance_metrics_flat_and_nested():
    assert ms.performance_metrics({"algorithm_performance": {"CatBoost": 0.12}}) == [
        ("CatBoost", 0.12)
    ]
    assert ms.performance_metrics(
        {"algorithm_performance": {"CatBoost": {"mae": 0.5}}}
    ) == [("CatBoost.mae", 0.5)]
    assert ms.performance_metrics({}) == []
    assert ms.performance_metrics({"algorithm_performance": "n/a"}) == []


def test_model_option_summary_and_label():
    option = ms.discover_models(models=[
        _entry("prod", oh_name="CatBoost", nps_name="LightGBM")
    ])[0]
    assert "CatBoost" in option.summary and "OH" in option.summary
    assert "LightGBM" in option.summary and "NPS" in option.summary
    assert option.short_label.startswith("prod")


# ============================================================
# Per-feature session persistence
# ============================================================

def test_feature_selection_persists_per_feature():
    ms.set_feature_selection("predict", "alpha")
    ms.set_feature_selection("forecast", "beta")
    assert ms.get_feature_selection("predict") == "alpha"
    assert ms.get_feature_selection("forecast") == "beta"
    ms.set_feature_selection("predict", None)
    assert ms.get_feature_selection("predict") is None
    assert ms.get_feature_selection("forecast") == "beta"


def test_include_test_toggle_persists():
    assert ms.get_include_test("predict") is False
    ms.set_include_test("predict", True)
    assert ms.get_include_test("predict") is True
    ms.reset_feature_selections()
    assert ms.get_include_test("predict") is False
    assert ms.get_feature_selection("predict") is None


# ============================================================
# Explicit selection passed into the service layer
# ============================================================

def test_find_target_state_passes_explicit_family_bundles(models_dir, monkeypatch):
    _make_family_pair(models_dir, "alpha")
    monkeypatch.setattr(
        "core.forecast_ai.prediction.model_selector.MODELS_DIR", models_dir
    )
    monkeypatch.setattr("gui.services.MODELS_DIR", models_dir)
    monkeypatch.setattr(svc, "_ram_guard", lambda *a, **k: None)

    captured = {}

    class _FakeEngine:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs

        def find_target_state(self, targets, **kwargs):
            return {
                "targets": dict(targets),
                "recommended_state": {"quality": 90.0},
                "consensus": {"oh": 88.0},
                "distance": 0.1,
                "leaderboards": {"OH": [], "NPS": []},
            }

    monkeypatch.setattr(
        "core.target_state_engine.engine.TargetStateEngine", _FakeEngine
    )

    result = svc.find_target_state({"oh": 80.0}, family="alpha")
    kwargs = captured["kwargs"]
    assert kwargs.get("oh_bundle", {}).get("model_name") == "oh_alpha"
    assert kwargs.get("nps_bundle", {}).get("model_name") == "nps_alpha"
    assert result["active_family"] == "alpha"


def test_find_target_state_without_family_uses_legacy_path(models_dir, monkeypatch):
    monkeypatch.setattr(svc, "_ram_guard", lambda *a, **k: None)

    captured = {}

    class _FakeEngine:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs

        def find_target_state(self, targets, **kwargs):
            return {"recommended_state": {}, "consensus": {}, "distance": None,
                    "leaderboards": {}}

    monkeypatch.setattr(
        "core.target_state_engine.engine.TargetStateEngine", _FakeEngine
    )

    result = svc.find_target_state({"oh": 80.0})
    assert captured["kwargs"] == {}
    assert result["active_family"] is None


def test_find_target_state_invalid_family_raises_no_fallback(models_dir, monkeypatch):
    monkeypatch.setattr(
        "core.forecast_ai.prediction.model_selector.MODELS_DIR", models_dir
    )
    monkeypatch.setattr("gui.services.MODELS_DIR", models_dir)
    monkeypatch.setattr(svc, "_ram_guard", lambda *a, **k: None)
    _make_family_pair(models_dir, "alpha")
    STATE.set_active_family("alpha")

    with pytest.raises(ModelPairError):
        svc.find_target_state({"oh": 80.0}, family="missing_family")
    # The engine must not have been reached and the session selection intact.
    assert STATE.get_active_family() == "alpha"


# ============================================================
# Never silently fall back from an explicit selection
# ============================================================

def test_predict_invalid_explicit_family_never_falls_back():
    STATE.set_active_family("prod_a")
    state = {
        "quality": 90.0, "competency": 90.0, "attendance": 90.0,
        "release": 60.0, "transfer": 9.0, "operations_health": 90.0,
        "nps": 70.0,
    }
    with pytest.raises(ModelPairError):
        svc.predict(state, family="does_not_exist")
    assert STATE.get_active_family() == "prod_a"


def test_forecast_invalid_explicit_family_never_falls_back():
    STATE.set_active_family("prod_a")
    state = {
        "quality": 90.0, "competency": 90.0, "attendance": 90.0,
        "release": 60.0, "transfer": 9.0, "operations_health": 90.0,
        "nps": 70.0,
    }
    with pytest.raises(ModelPairError):
        svc.forecast(state, 3, family="does_not_exist")
    assert STATE.get_active_family() == "prod_a"
