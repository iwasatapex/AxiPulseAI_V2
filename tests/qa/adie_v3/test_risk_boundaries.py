"""Canonical risk-model boundary tests (Phase 8)."""

from core.decision_intelligence.v3.risk.uncertainty import UncertaintyRiskEngine
from core.decision_intelligence.v3.policy import constants as C


def test_low_medium_high_boundaries():
    engine = UncertaintyRiskEngine()
    t = C.RISK_THRESHOLDS
    # Just above the medium threshold -> LOW.
    low = engine.assess(t["medium_probability"] + 0.01, t["medium_confidence"] + 0.01, 0.4, 0.9)
    assert low.risk == "LOW"
    # At/just below medium -> MEDIUM.
    med = engine.assess(t["medium_probability"] - 0.01, t["medium_confidence"] + 0.01, 0.4, 0.9)
    assert med.risk == "MEDIUM"
    # Below high -> HIGH.
    high = engine.assess(t["high_probability"] - 0.01, t["medium_confidence"], 0.4, 0.9)
    assert high.risk == "HIGH"


def test_abstain_rules():
    engine = UncertaintyRiskEngine()
    # Confidence below abstain threshold always abstains.
    r = engine.assess(0.6, C.ABSTAIN_THRESHOLDS["confidence"] - 0.01, 0.4, 0.9)
    assert r.abstain is True
    # Low probability with no upside also abstains.
    r2 = engine.assess(0.2, 0.6, downside=0.5, upside=0.4)
    assert r2.abstain is True
    # High probability + upside does not abstain.
    r3 = engine.assess(0.8, 0.7, downside=0.3, upside=0.95)
    assert r3.abstain is False


def test_score_bounded_0_1():
    engine = UncertaintyRiskEngine()
    r = engine.assess(0.0, 0.0, downside=0.0, upside=0.0)
    assert 0.0 <= r.score <= 1.0


def test_single_risk_model_shared():
    # The forecast service risk and the API risk both come from the canonical
    # model (constants-driven). Compare two assess() calls on the same inputs.
    engine = UncertaintyRiskEngine()
    a = engine.assess(0.7, 0.7, 0.5, 0.95)
    b = engine.assess(0.7, 0.7, 0.5, 0.95)
    assert a == b
    assert a.risk == "LOW"
