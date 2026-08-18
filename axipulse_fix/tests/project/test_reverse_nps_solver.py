import importlib

import pytest

import reverse_nps_solver as solver


def test_reverse_nps_solver_surface():
    module = importlib.import_module("reverse_nps_solver")
    assert hasattr(module, "load_latest_state")
    assert hasattr(module, "predict")
    assert hasattr(module, "solve")


# A single KPI combination (deterministic and fast).
TINY_GRID = {
    "QUALITY_RANGE": range(87, 88),
    "COMPETENCY_RANGE": range(93, 94),
    "RELEASE_RANGE": range(60, 61),
    "TRANSFER_RANGE": range(9, 10),
    "ATTENDANCE_RANGE": range(90, 91),
}


@pytest.fixture(autouse=True)
def base_state(monkeypatch):
    """Provide a deterministic base state instead of reading the generated
    training/training.csv artifact, which is not present in the repo."""
    monkeypatch.setattr(
        solver,
        "load_latest_state",
        lambda: {
            "operational_health": 95.0,
            "quality": 87.0,
            "competency": 93.0,
            "attendance": 90.0,
            "release": 60.0,
            "transfer": 9.0,
            "total_calls_received": 2000,
        },
    )


@pytest.fixture()
def tiny_grid(monkeypatch):
    for name, value in TINY_GRID.items():
        monkeypatch.setattr(solver, name, value)


def _combo_state():
    base = solver.load_latest_state()
    # _build_state(base, quality, competency, release, transfer, attendance)
    return solver._build_state(base, 87, 93, 60, 9, 90)


def test_current_model_loads():
    predictor = solver._load_predictor()
    assert predictor is not None
    assert predictor.trained is True
    assert predictor.model is not None


def test_predict_returns_float_from_canonical_engine():
    predictor = solver._load_predictor()
    nps = solver.predict(predictor, _combo_state())
    assert isinstance(nps, float)
    assert -100.0 <= nps <= 100.0


def test_solve_for_no_bundle_type_mismatch(tiny_grid):
    # Regression: the old solver called .predict() on the raw joblib bundle
    # dict and always returned found=False. This must now execute against the
    # canonical NPSPredictor and return a structured result.
    result = solver.solve_for(50.0)
    assert isinstance(result, dict)
    assert "found" in result
    assert result["found"] is False
    assert "reason" in result


def test_solve_for_feasible_target(tiny_grid):
    predictor = solver._load_predictor()
    achievable = solver.predict(predictor, _combo_state())

    result = solver.solve_for(achievable)
    assert result["found"] is True
    assert result["distance"] <= 1.0
    assert result["predicted_nps"] == pytest.approx(achievable)


def test_solve_for_unreachable_target(tiny_grid):
    result = solver.solve_for(0.0)
    assert result["found"] is False
    assert "reason" in result
    assert result["distance"] is not None


def test_solve_for_malformed_input():
    for bad in ("not-a-number", float("nan"), float("inf"), 250.0, -250.0):
        result = solver.solve_for(bad)
        assert result["found"] is False
        assert "reason" in result


def test_solution_stays_within_bounds(tiny_grid):
    predictor = solver._load_predictor()
    achievable = solver.predict(predictor, _combo_state())

    result = solver.solve_for(achievable)
    assert result["found"] is True
    assert 0.0 <= result["quality"] <= 100.0
    assert 0.0 <= result["competency"] <= 100.0
    assert 0.0 <= result["attendance"] <= 100.0
    assert 0.0 <= result["release"] <= 100.0
    assert 0.0 <= result["transfer"] <= 100.0
    assert 0.0 <= result["operational_health"] <= 120.0

