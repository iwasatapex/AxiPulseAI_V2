"""
AxiPulseAI Decision Intelligence Engine (ADIE) — canonical V3.

ADIE is the decision/reasoning layer. It consumes Forecast AI outputs
(forecast, scenarios, risk, confidence, sensitivity, simulation) and
produces a single decision package via the V3 probabilistic pipeline
(Bayesian + Monte Carlo -> risk/uncertainty -> policy/decision synthesis).

V3 is advisor-only: it never executes business actions, never trains or
replaces the predictive models, and never mutates observed state.
"""

from __future__ import annotations

from .v3.intelligence import ADIEProbabilisticEngine, ProbabilisticDecision, analyze
from .v3.scenario.engine import (
    ADIEScenarioEngine,
    Scenario,
    ScenarioResult,
    run,
    compare,
)
from .v3.synthesis.decision_synthesizer import (
    DecisionSynthesizer,
    SynthesizedDecision,
    synthesize,
)
from .v3.synthesis.decision_detail import build_adie_detail
from .v3.synthesis.explanation import build_explanation  # noqa: F401
from .v3.policy.decision_policy import DecisionPolicyEngine, PolicyDecision, select  # noqa: F401
from .v3.policy.constants import (  # noqa: F401
    RISK_SCORE_WEIGHTS,
    RISK_THRESHOLDS,
    ABSTAIN_THRESHOLDS,
    DECISION_POLICY,
    SCENARIO_RANKING_WEIGHTS,
)
from .v3.risk.uncertainty import (  # noqa: F401
    UncertaintyRiskEngine,
    RiskAssessment,
    assess,
)
from .v3.scenario.scoring import (  # noqa: F401
    compute_scenario_score,
    rank_scenarios,
)
from .v3.integration.probabilistic_decision import (
    ProbabilisticDecisionPackage,
    ProbabilisticDecisionService,
)
from .v3.integration.production_boundary import (
    ProductionDecisionBoundary,
    ProductionDecisionInput,
)

__all__ = [
    "ADIEProbabilisticEngine",
    "ProbabilisticDecision",
    "analyze",
    "ADIEScenarioEngine",
    "Scenario",
    "ScenarioResult",
    "run",
    "compare",
    "DecisionSynthesizer",
    "SynthesizedDecision",
    "synthesize",
    "DecisionPolicyEngine",
    "PolicyDecision",
    "select",
    "RiskAssessment",
    "assess",
    "UncertaintyRiskEngine",
    "ProbabilisticDecisionPackage",
    "ProbabilisticDecisionService",
    "ProductionDecisionBoundary",
    "ProductionDecisionInput",
    "compute_scenario_score",
    "rank_scenarios",
    "build_explanation",
    "build_adie_detail",
]
