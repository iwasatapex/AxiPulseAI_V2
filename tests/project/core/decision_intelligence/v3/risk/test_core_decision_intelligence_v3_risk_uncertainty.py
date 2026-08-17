import importlib

import pytest

from core.decision_intelligence.v3.policy import constants as C
from core.decision_intelligence.v3.risk.uncertainty import UncertaintyRiskEngine


def test_uncertainty_surface():
    module = importlib.import_module("core.decision_intelligence.v3.risk.uncertainty")
    assert hasattr(module, "assess")
    assert hasattr(module, "RiskAssessment")
    assert hasattr(module, "UncertaintyRiskEngine")


def test_classify_level_matches_assess():
    """classify_level is the canonical level source; assess() must agree."""
    engine = UncertaintyRiskEngine()
    for probability in (0.1, 0.3, 0.34, 0.35, 0.36, 0.59, 0.60, 0.61, 0.9):
        for confidence in (0.1, 0.34, 0.35, 0.36, 0.59, 0.60, 0.61, 0.9):
            expected = engine.assess(
                probability, confidence, downside=0.0, upside=1.0
            ).risk
            assert engine.classify_level(probability, confidence) == expected


def test_classify_level_boundaries():
    """Values immediately below/on/above the canonical thresholds."""
    hi_conf = C.RISK_THRESHOLDS["high_confidence"]   # 0.35
    hi_prob = C.RISK_THRESHOLDS["high_probability"]   # 0.35
    med_conf = C.RISK_THRESHOLDS["medium_confidence"]  # 0.60
    med_prob = C.RISK_THRESHOLDS["medium_probability"]  # 0.60

    # HIGH below the high threshold; MEDIUM at/above it.
    assert UncertaintyRiskEngine.classify_level(hi_prob - 0.01, 1.0) == "HIGH"
    assert UncertaintyRiskEngine.classify_level(hi_prob, 1.0) == "MEDIUM"
    assert UncertaintyRiskEngine.classify_level(hi_prob + 0.01, 1.0) == "MEDIUM"
    assert UncertaintyRiskEngine.classify_level(1.0, hi_conf - 0.01) == "HIGH"
    assert UncertaintyRiskEngine.classify_level(1.0, hi_conf) == "MEDIUM"

    # MEDIUM below the medium threshold; LOW at/above it.
    assert UncertaintyRiskEngine.classify_level(med_prob - 0.01, 1.0) == "MEDIUM"
    assert UncertaintyRiskEngine.classify_level(med_prob, 1.0) == "LOW"
    assert UncertaintyRiskEngine.classify_level(med_prob + 0.01, 1.0) == "LOW"
    assert UncertaintyRiskEngine.classify_level(1.0, med_conf - 0.01) == "MEDIUM"
    assert UncertaintyRiskEngine.classify_level(1.0, med_conf) == "LOW"


def test_classify_level_clamps_inputs():
    """Out-of-domain inputs are clamped to [0,1], never raising."""
    assert UncertaintyRiskEngine.classify_level(5.0, 0.9) in {"HIGH", "MEDIUM", "LOW"}
    assert UncertaintyRiskEngine.classify_level(-1.0, 0.9) in {"HIGH", "MEDIUM", "LOW"}
    assert UncertaintyRiskEngine.classify_level(0.9, 5.0) in {"HIGH", "MEDIUM", "LOW"}
    assert UncertaintyRiskEngine.classify_level(0.9, 0.9) == "LOW"

