import importlib

import pytest


def test_transition_surface():
    module = importlib.import_module("core.forecast_ai.state.transition")
    # The momentum API lives on the class (not a module-level `apply`).
    assert hasattr(module, "KPITransition")
    assert hasattr(module.KPITransition, "apply")


def test_transition_momentum_coefficient_is_0_6():
    """V2.3 rule 9: momentum coefficient is 0.6 (0.6*previous + 0.4*base)."""
    from core.forecast_ai.state.transition import KPITransition

    t = KPITransition(autocorrelation=0.6)
    state = {"quality": 80.0, "competency": 80.0, "attendance": 80.0,
             "release": 50.0, "transfer": 12.0}
    out = t.apply(state)

    assert out["quality"] == pytest.approx(0.6 * 80.0 + 0.4 * 87.0, abs=1e-6)
    assert out["competency"] == pytest.approx(0.6 * 80.0 + 0.4 * 93.0, abs=1e-6)
    assert out["attendance"] == pytest.approx(0.6 * 80.0 + 0.4 * 90.0, abs=1e-6)
    assert out["release"] == pytest.approx(0.6 * 50.0 + 0.4 * 60.0, abs=1e-6)
    assert out["transfer"] == pytest.approx(0.6 * 12.0 + 0.4 * 9.0, abs=1e-6)


def test_transition_hard_bounds_match_v23():
    """V2.3 rule 2: KPI hard bounds in the state transition."""
    from core.forecast_ai.state.transition import KPITransition

    t = KPITransition(autocorrelation=0.6)
    # Extreme current values force clamping to the hard bounds.
    low = {"quality": -10.0, "competency": -10.0, "attendance": -10.0,
           "release": -10.0, "transfer": -10.0}
    high = {"quality": 200.0, "competency": 200.0, "attendance": 200.0,
            "release": 200.0, "transfer": 200.0}

    lo = t.apply(low)
    hi = t.apply(high)

    # Lower hard bounds: quality 60, competency 55, attendance 65, release 50, transfer 0.
    assert lo["quality"] == pytest.approx(60.0)
    assert lo["competency"] == pytest.approx(55.0)
    assert lo["attendance"] == pytest.approx(65.0)
    assert lo["release"] == pytest.approx(50.0)
    assert lo["transfer"] == pytest.approx(0.0)

    # Upper hard bounds: all 100 except transfer 20.
    assert hi["quality"] == pytest.approx(100.0)
    assert hi["competency"] == pytest.approx(100.0)
    assert hi["attendance"] == pytest.approx(100.0)
    assert hi["release"] == pytest.approx(100.0)
    assert hi["transfer"] == pytest.approx(20.0)

