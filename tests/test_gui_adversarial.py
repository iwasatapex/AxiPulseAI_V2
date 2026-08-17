"""Adversarial tests: contract bypass, scenario security, dashboard health,
and cwd-independence.

These deliberately call the service layer directly (bypassing Streamlit
widgets) to prove validation/enforcement holds at the service boundary, and
exercise disabled-scenario rejection and dynamic health without a browser.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gui import contracts as ct
from gui import services as svc


def _valid_state():
    return {
        "quality": 87.0, "competency": 93.0, "attendance": 90.0,
        "release": 60.0, "transfer": 9.0, "operations_health": 95.0,
        "nps": 82.0, "total_calls_received": 2000.0,
    }


def _stub_engine(monkeypatch, pipeline_run=None):
    """Stub model loading + pipeline so validation is the only gate."""
    monkeypatch.setattr(svc, "validate_model_pair", lambda fam: (None, None))

    class StubPipeline:
        def run(self, state):
            if pipeline_run is not None:
                pipeline_run()
            raw = SimpleNamespace(operations_health=None, nps=None)
            return SimpleNamespace(
                prediction=SimpleNamespace(
                    raw=raw, operations_health=None, nps=None,
                    bayesian_score_distribution=None, score_counts=None,
                )
            )
    monkeypatch.setattr(
        "core.forecast_ai.prediction.pipeline.ProductionPredictionPipeline",
        StubPipeline,
    )


# =====================================================================
# 7. Contract bypass (service-level, no widget validation)
# =====================================================================

@pytest.mark.parametrize("key,value", [
    ("release", 0.0), ("release", 49.99), ("release", 101.0),
    ("transfer", -1.0), ("transfer", 20.01), ("transfer", 100.0),
    ("quality", 59.99), ("quality", 100.01),
    ("competency", 54.99), ("competency", 100.01),
    ("attendance", 64.99), ("attendance", 100.01),
    ("operations_health", -1.0), ("operations_health", 100.01),
    ("nps", -100.01), ("nps", 100.01),
])
def test_service_rejects_out_of_range_kpi(monkeypatch, key, value):
    _stub_engine(monkeypatch)
    state = _valid_state()
    state[key] = value
    svc.STATE.reset()
    with pytest.raises(ValueError, match="within"):
        svc.predict(state, family="alpha")
    svc.STATE.reset()


@pytest.mark.parametrize("key,value", [
    ("release", 50.0), ("release", 100.0),
    ("transfer", 0.0), ("transfer", 20.0),
    ("quality", 60.0), ("quality", 100.0),
    ("competency", 55.0), ("competency", 100.0),
    ("attendance", 65.0), ("attendance", 100.0),
    ("operations_health", 0.0), ("operations_health", 100.0),
    ("nps", -100.0), ("nps", 100.0),
])
def test_service_accepts_boundary_values(monkeypatch, key, value):
    reached = []
    _stub_engine(monkeypatch, pipeline_run=lambda: reached.append(1))
    state = _valid_state()
    state[key] = value
    svc.STATE.reset()
    out = svc.predict(state, family="alpha")
    assert reached, "pipeline should have run for a valid state"
    assert out["active_family"] == "alpha"
    svc.STATE.reset()


def test_validate_state_accepts_valid_and_rejects_invalid():
    ct.validate_state(_valid_state())
    bad = _valid_state()
    bad["release"] = 49.0
    with pytest.raises(ValueError):
        ct.validate_state(bad)


# =====================================================================
# 10. Scenario security / validity (service layer, not dropdown)
# =====================================================================

def test_enabled_scenario_allowed():
    from core.forecast_ai.scenarios.registry import ScenarioRegistry
    ScenarioRegistry.reset()
    svc._ensure_enabled_scenario("training")  # registered + enabled


def test_baseline_scenario_allowed():
    svc._ensure_enabled_scenario("baseline")  # default no-op


def test_nonexistent_scenario_rejected():
    with pytest.raises(ValueError, match="Unknown scenario"):
        svc._ensure_enabled_scenario("__no_such_scenario__")


def test_disabled_scenario_rejected():
    from core.forecast_ai.scenarios.registry import ScenarioRegistry
    from core.forecast_ai.scenarios.models import Scenario

    ScenarioRegistry.register(Scenario(
        id="__disabled__", name="x", description="x",
        modifiers=[], enabled=False,
    ))
    try:
        with pytest.raises(ValueError, match="disabled"):
            svc._ensure_enabled_scenario("__disabled__")
    finally:
        ScenarioRegistry.reset()


def test_forecast_rejects_disabled_scenario_before_engine(monkeypatch):
    from core.forecast_ai.scenarios.registry import ScenarioRegistry
    from core.forecast_ai.scenarios.models import Scenario

    reached = []
    ScenarioRegistry.register(Scenario(
        id="__disabled__", name="x", description="x", modifiers=[], enabled=False,
    ))

    def _boom(*a, **k):
        reached.append("orchestrator")

    class StubOrchestrator:
        def __init__(self, *a, **k): pass
        def execute(self, req): _boom()

    monkeypatch.setattr("core.forecast_ai.engines.forecast_orchestrator.ForecastOrchestrator", StubOrchestrator)
    monkeypatch.setattr(svc, "validate_model_pair", lambda fam: (None, None))
    svc.STATE.reset()
    try:
        with pytest.raises(ValueError, match="disabled"):
            svc.forecast(_valid_state(), horizon=1, scenario="__disabled__", family="alpha")
        assert not reached, "orchestrator must not run for a disabled scenario"
    finally:
        ScenarioRegistry.reset()
        svc.STATE.reset()


def test_duplicate_baseline_registration_exposes_one_baseline():
    from core.forecast_ai.scenarios.registry import ScenarioRegistry
    from core.forecast_ai.scenarios.models import Scenario

    # Registry is keyed by id, so re-registering "baseline" overwrites, never
    # duplicates. Guard list_scenarios() exposes exactly one baseline.
    ScenarioRegistry.register(Scenario(
        id="baseline", name="Baseline2", description="dup", modifiers=[],
    ))
    try:
        ids = [s["id"] for s in svc.list_scenarios()]
        assert ids.count("baseline") == 1
    finally:
        ScenarioRegistry.reset()


# =====================================================================
# 12. Dashboard health
# =====================================================================

def test_health_ready_when_all_dependencies_ok(monkeypatch):
    monkeypatch.setattr(svc, "list_model_families", lambda: ["alpha"])
    monkeypatch.setattr(svc, "validate_model_pair", lambda fam: (None, None))
    svc.STATE.reset()
    svc.STATE.set_active_family("alpha")
    health = svc.system_health()
    assert health["status"] == "Ready"
    assert health["checks"]["models"]["status"] == "Ready"
    assert health["checks"]["active_model"]["status"] == "Ready"
    assert health["checks"]["scenarios"]["status"] == "Ready"
    svc.STATE.reset()


def test_health_unavailable_when_no_models(monkeypatch):
    monkeypatch.setattr(svc, "list_model_families", lambda: [])
    svc.STATE.reset()
    health = svc.system_health()
    assert health["status"] == "Unavailable"
    assert health["checks"]["models"]["status"] == "Unavailable"
    svc.STATE.reset()


def test_health_missing_models_dir_is_unavailable(monkeypatch):
    monkeypatch.setattr(svc, "list_model_families", lambda: [])
    svc.STATE.reset()
    health = svc.system_health()
    assert health["status"] == "Unavailable"
    svc.STATE.reset()


def test_health_degraded_when_baseline_missing(monkeypatch):
    monkeypatch.setattr(svc, "list_model_families", lambda: ["alpha"])
    monkeypatch.setattr(svc, "validate_model_pair", lambda fam: (None, None))
    monkeypatch.setattr(svc, "list_scenarios", lambda *a, **k: [])
    svc.STATE.reset()
    svc.STATE.set_active_family("alpha")
    health = svc.system_health()
    assert health["status"] == "Degraded"
    assert health["checks"]["scenarios"]["status"] == "Degraded"
    svc.STATE.reset()


def test_health_degraded_when_active_family_invalid(monkeypatch):
    from core.forecast_ai.prediction.model_selector import ModelPairError
    monkeypatch.setattr(svc, "list_model_families", lambda: ["alpha"])
    monkeypatch.setattr(
        svc, "validate_model_pair",
        lambda fam: (_ for _ in ()).throw(ModelPairError("bad pair")),
    )
    svc.STATE.reset()
    svc.STATE.set_active_family("alpha")
    health = svc.system_health()
    assert health["status"] == "Degraded"
    assert health["checks"]["active_model"]["status"] == "Degraded"
    svc.STATE.reset()


# =====================================================================
# 5. CWD-independence
# =====================================================================

def test_paths_and_discovery_are_cwd_independent(monkeypatch, tmp_path):
    from core.forecast_ai.prediction.model_selector import (
        MODELS_DIR,
        list_model_families,
        list_training_files,
    )

    monkeypatch.chdir(tmp_path)  # unrelated working directory

    assert MODELS_DIR.is_absolute()
    # Discovery looks in the canonical repo dirs, not the cwd.
    assert isinstance(list_model_families(), list)
    assert isinstance(list_training_files(), list)
    # No accidental ./models or ./training dirs in the unrelated cwd.
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "training").exists()


def test_no_relative_models_dir_in_gui():
    """GUI must not construct relative 'models/...' paths."""
    import gui
    root = Path(gui.__file__).resolve().parent
    hits = []
    for p in root.rglob("*.py"):
        if "__pycache__" in str(p):
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if ("\"models/" in line or "'models/" in line
                    or "models/{" in line):
                hits.append(f"{p}:{i}: {line.strip()}")
    assert not hits, "relative model paths found:\n" + "\n".join(hits)
