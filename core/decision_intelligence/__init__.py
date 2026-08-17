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

# =====================================================================
# CANONICAL (production) — ADIE V3
# =====================================================================
# These are the production decision surfaces. The single risk semantic is
# ``UncertaintyRiskEngine``; scenario ranking is the deterministic policy.
from .v3.intelligence import ADIEProbabilisticEngine, ProbabilisticDecision, analyze
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
from .v3.synthesis.decision_detail import build_adie_detail
from .v3.synthesis.explanation import build_explanation  # noqa: F401
from .v3.integration.probabilistic_decision import (
    ProbabilisticDecisionPackage,
    ProbabilisticDecisionService,
)
from .v3.integration.production_boundary import (
    ProductionDecisionBoundary,
    ProductionDecisionInput,
)

# =====================================================================
# LEGACY (ADIE 2.0) — DEPRECATED, retained for backward compatibility
# =====================================================================
# ``DecisionSynthesizer`` embeds its own non-canonical risk thresholds and
# ``ADIEScenarioEngine`` runs its own Bayesian + Monte Carlo per scenario.
# Neither is part of the production ADIE V3 pipeline (which uses
# ``UncertaintyRiskEngine`` and deterministic ``rank_scenarios`` with a
# single Monte Carlo execution). Do not use them for new code.
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
