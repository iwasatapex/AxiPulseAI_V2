from dataclasses import dataclass, field
from typing import Any, Mapping

from core.decision_intelligence.v3.policy import constants as C


@dataclass
class PolicyDecision:
    recommendation: str
    action: str
    priority: str
    risk: str
    confidence: float
    abstain: bool
    reason: str = ""
    affected_kpi: str | None = None
    direction: str | None = None
    evidence: list[str] = field(default_factory=list)
    uncertainty: dict[str, Any] = field(default_factory=dict)


def _as_dict(item: Any) -> Mapping[str, Any]:
    """Read a scenario as a mapping whether it is a dict or an object."""
    if isinstance(item, Mapping):
        return item
    return {k: getattr(item, k) for k in (
        "name", "probability", "confidence", "operations_health", "nps",
        "risk_severity", "risk", "delta_oh", "score", "rank", "tie", "date",
    ) if hasattr(item, k)}


def _kpi_recommendation_name(kpi: str) -> str:
    kpi = (kpi or "").strip().lower().replace(" ", "_")
    if kpi == "transfer":
        return "prioritize_transfer_reduction"
    return f"prioritize_{kpi}_improvement"


def _best_kpi_gap(
    best: Mapping[str, Any],
    targets: Mapping[str, Any] | None,
    sensitivity_output: Mapping[str, Any] | None,
) -> str | None:
    """Return the KPI most worth prioritizing (target gap, else the top
    sensitivity/leverage KPI), or None when no evidence supports a
    KPI-specific decision."""
    targets = dict(targets or {})

    candidates: list[tuple[str, float]] = []
    target_nps = targets.get("target_nps")
    if target_nps is not None:
        value = best.get("nps")
        if value is not None:
            target = float(target_nps)
            value = float(value)
            if value < target:
                candidates.append(
                    ("nps", (target - value) / max(abs(target), 1e-9))
                )
    target_oh = targets.get("target_oh")
    if target_oh is not None:
        value = best.get("operations_health")
        if value is not None:
            target = float(target_oh)
            value = float(value)
            if value < target:
                candidates.append(
                    ("operations_health", (target - value) / max(abs(target), 1e-9))
                )
    if candidates:
        candidates.sort(key=lambda pair: -pair[1])
        return candidates[0][0]

    if sensitivity_output:
        ranking = sensitivity_output.get("ranking") or []
        if ranking:
            scored = []
            for entry in ranking:
                metric = entry.get("metric") if isinstance(entry, Mapping) else getattr(entry, "metric", None)
                score = entry.get("sensitivity_score_oh") if isinstance(entry, Mapping) else getattr(entry, "sensitivity_score_oh", 0.0)
                if metric:
                    scored.append((abs(float(score or 0.0)), str(metric)))
            if scored:
                scored.sort(key=lambda pair: -pair[0])
                return scored[0][1].strip().lower().replace(" ", "_")

    return None


class DecisionPolicyEngine:
    def select(
        self,
        scenarios,
        risk,
    ) -> PolicyDecision:
        """Compatibility surface: legacy scenario/risk decision selection.

        ``scenarios`` may be objects (with ``name``/``probability``/
        ``confidence``) or dicts. Behavior is preserved from the original
        implementation; meaningful recommendation semantics live in
        ``select_evidence``.
        """
        if not scenarios:
            return PolicyDecision(
                recommendation="insufficient_evidence",
                action=C.DECISION_POLICY["abstain_action"],
                priority=C.DECISION_POLICY["abstain_priority"],
                risk=risk.risk,
                confidence=risk.confidence,
                abstain=True,
                reason="no scenarios supplied; insufficient evidence to decide",
                direction="defer",
            )

        def _prob(item):
            if isinstance(item, Mapping):
                return float(item.get("probability", 0.0) or 0.0)
            return float(getattr(item, "probability", 0.0) or 0.0)

        def _conf(item):
            if isinstance(item, Mapping):
                return float(item.get("confidence", 0.0) or 0.0)
            return float(getattr(item, "confidence", 0.0) or 0.0)

        ranked = sorted(
            scenarios,
            key=lambda x: (_prob(x), _conf(x)),
            reverse=True,
        )

        best = ranked[0]
        recommendation = (
            best.get("name", "improved_operations")
            if isinstance(best, Mapping)
            else getattr(best, "name", "improved_operations")
        )
        probability = _prob(best)

        if risk.abstain:
            return PolicyDecision(
                recommendation=recommendation,
                action=C.DECISION_POLICY["abstain_action"],
                priority=C.DECISION_POLICY["abstain_priority"],
                risk=risk.risk,
                confidence=risk.confidence,
                abstain=True,
                reason="risk model flagged abstain (insufficient confidence or no upside)",
                direction="defer",
            )

        if risk.risk == "HIGH":
            action = C.DECISION_POLICY["high_risk_action"]
            priority = str(C.DECISION_POLICY["high_risk_priority"])
        elif risk.risk == "MEDIUM":
            action = C.DECISION_POLICY["medium_risk_action"]
            priority = str(C.DECISION_POLICY["medium_risk_priority"])
        else:
            action = C.DECISION_POLICY["low_risk_action"]
            priority = (
                str(C.DECISION_POLICY["execute_high_priority_label"])
                if probability >= float(C.DECISION_POLICY["execute_high_priority_probability"])
                else str(C.DECISION_POLICY["execute_medium_priority_label"])
            )

        return PolicyDecision(
            recommendation=recommendation,
            action=action,
            priority=priority,
            risk=risk.risk,
            confidence=risk.confidence,
            abstain=False,
            reason=f"scenario '{recommendation}' ranked best; {risk.risk} risk -> {action}",
        )

    def select_evidence(
        self,
        scenarios,
        risk,
        *,
        targets: Any = None,
        sensitivity_output: Any = None,
        aggregate: Mapping[str, Any] | None = None,
        observed: Any = None,
    ) -> PolicyDecision:
        """
        Evidence-based decision selection (Phases 2, 3, 6, 9).

        ``scenarios`` should be the already-ranked list (dicts; rank 1 =
        preferred). Produces a meaningful, explainable recommendation using
        the canonical risk model plus the real forecast evidence carried on
        each scenario (target gaps, sensitivity leverage).

        This never invents quantitative gains and never hard-codes a single
        recommendation for every input.
        """
        scenarios = list(scenarios or [])
        if not scenarios or risk.abstain:
            return PolicyDecision(
                recommendation="defer_action_due_to_uncertainty",
                action=C.DECISION_POLICY["abstain_action"],
                priority=C.DECISION_POLICY["abstain_priority"],
                risk=risk.risk,
                confidence=risk.confidence,
                abstain=True,
                reason=(
                    "insufficient evidence to make a decision"
                    if not scenarios
                    else "risk model flagged abstain (insufficient confidence or no upside)"
                ),
                direction="defer",
                evidence=(
                    ["no scenario evidence", "canonical risk model returned abstain"]
                    if risk.abstain
                    else ["no scenario evidence"]
                ),
            )

        best = _as_dict(scenarios[0])
        best_name = str(best.get("name") or "current_state")
        best_probability = best.get("probability")

        if risk.risk == "HIGH":
            action = C.DECISION_POLICY["high_risk_action"]
            priority = str(C.DECISION_POLICY["high_risk_priority"])
        elif risk.risk == "MEDIUM":
            action = C.DECISION_POLICY["medium_risk_action"]
            priority = str(C.DECISION_POLICY["medium_risk_priority"])
        else:
            action = C.DECISION_POLICY["low_risk_action"]
            try:
                threshold = float(C.DECISION_POLICY["execute_high_priority_probability"])
            except (TypeError, ValueError):
                threshold = 0.80
            priority = (
                str(C.DECISION_POLICY["execute_high_priority_label"])
                if best_probability is not None and float(best_probability) >= threshold
                else str(C.DECISION_POLICY["execute_medium_priority_label"])
            )

        is_generated_day = best_name.startswith("forecast_day_")
        recommendation = best_name
        affected_kpi = None
        direction = "improve"
        reason = (
            f"scenario '{best_name}' ranked best under the deterministic "
            "scenario policy"
        )

        gap = _best_kpi_gap(best, targets, sensitivity_output)
        if gap is not None:
            affected_kpi = gap
            recommendation = _kpi_recommendation_name(gap)
            direction = "reduce" if gap == "transfer" else "improve"
            reason = (
                f"{gap} identified as the priority from target gap / "
                "sensitivity leverage on the best forecast scenario"
            )
        elif risk.risk == "HIGH":
            recommendation = "monitor_high_risk_forecast"
            direction = "observe"
            action = C.DECISION_POLICY["high_risk_action"]
            reason = (
                "canonical risk is HIGH; keep the best forecast under "
                "monitoring before executing"
            )
        elif is_generated_day and observed is not None:
            try:
                observed_score = float(observed)
            except (TypeError, ValueError):
                observed_score = None
            best_performance = best.get("operations_health")
            if best_performance is None:
                best_performance = best.get("nps")
            improved = (
                best_performance is not None
                and observed_score is not None
                and float(best_performance) > observed_score
            ) or float(best.get("delta_oh") or 0.0) > 0.0
            if improved:
                recommendation = "pursue_" + best_name
                reason = (
                    "best scenario '%s' improves on the observed state under "
                    "the deterministic ranking policy" % best_name
                )
            else:
                recommendation = "maintain_current_plan"
                direction = "hold"
                reason = (
                    "best forecast scenario does not clearly improve on the "
                    "observed state; recommend maintaining the current plan"
                )

        return PolicyDecision(
            recommendation=recommendation,
            action=action,
            priority=priority,
            risk=risk.risk,
            confidence=risk.confidence,
            abstain=False,
            reason=reason,
            affected_kpi=affected_kpi,
            direction=direction,
            evidence=[
                "best scenario: " + best_name,
                "canonical risk level: %s (score %.3f)" % (risk.risk, risk.score),
            ],
            uncertainty={
                "downside": risk.downside,
                "upside": risk.upside,
                "confidence": risk.confidence,
            },
        )


__all__ = ["PolicyDecision", "DecisionPolicyEngine"]



_default_policy_engine = DecisionPolicyEngine()


def select(
    scenarios,
    risk,
    *,
    targets=None,
    sensitivity_output=None,
    aggregate=None,
    observed=None,
) -> PolicyDecision:
    """Module-level convenience: select a policy decision from ranked scenarios."""
    return _default_policy_engine.select_evidence(
        scenarios,
        risk,
        targets=targets,
        sensitivity_output=sensitivity_output,
        aggregate=aggregate,
        observed=observed,
    )


__all__ = ["PolicyDecision", "DecisionPolicyEngine", "select"]
