"""
RecommendationEngine – interprets optimization results into actionable advice.
"""
import uuid
from typing import List, Optional
from .models import Recommendation, RecommendationResult, Category, Priority, Difficulty
from .templates import RecommendationTemplates
from .ranking import RecommendationRanker
from .formatter import RecommendationFormatter
from .conflicts import ConflictDetector
from ..optimization import OptimizationResult, OptimizationSolution

class RecommendationEngine:
    def __init__(self):
        self.templates = RecommendationTemplates()
        self.ranker = RecommendationRanker()
        self.formatter = RecommendationFormatter()
        self.conflict_detector = ConflictDetector()

    def generate(self, optimization_result: OptimizationResult) -> RecommendationResult:
        """
        Convert OptimizationResult into Recommendations.

        - Target reached (``optimization_result.success``): normal success.
        - Best-effort (target NOT reached but a genuine improving candidate
          exists): advisory recommendations are produced but explicitly marked
          ``goal_achieved=False`` / ``best_effort=True`` — the target is never
          claimed achieved and estimated gains stay assumption-based (None).
        - Infeasible / timeout: no recommendations (skipped).
        """
        achieved = bool(optimization_result.success)
        best_effort = bool(
            (optimization_result.metadata or {}).get("best_effort")
        )
        if not (achieved or best_effort):
            return RecommendationResult(
                success=False,
                recommendations=[],
                errors=["Optimization failed, cannot generate recommendations."]
            )

        solutions = optimization_result.solutions
        if not solutions:
            return RecommendationResult(
                success=False,
                recommendations=[],
                errors=["No optimization solutions provided."]
            )

        best = optimization_result.best_solution
        if best is None:
            best = solutions[0]

        recommendations = self._deduplicate(self._extract_recommendations(best))
        ranked = self.ranker.rank(recommendations)
        conflicts = self.conflict_detector.detect_conflicts(ranked)
        warnings = [f"Conflict detected: {c[2]}" for c in conflicts] if conflicts else []
        if best_effort:
            warnings.append(
                "Target not reached; showing best-effort candidate. The target is "
                "NOT claimed achieved, and estimated gains are assumptions."
            )

        # Explicitly never claim achievement for best-effort: tag every rec.
        for rec in ranked:
            rec.metadata = dict(rec.metadata or {})
            rec.metadata["goal_achieved"] = achieved
            rec.metadata["best_effort"] = best_effort
            if best_effort:
                rec.metadata["gain_basis"] = "assumption_target_not_reached"

        return RecommendationResult(
            success=True,
            recommendations=ranked,
            warnings=warnings,
            errors=[],
            metadata={
                "total_solutions": len(solutions),
                "conflicts_detected": len(conflicts),
                "goal_achieved": achieved,
                "best_effort": best_effort,
            }
        )

    def _extract_recommendations(self, solution: OptimizationSolution) -> List[Recommendation]:
        recommendations = []
        # Extract from state_changes
        for field, change in solution.state_changes.items():
            if abs(change) < 0.01:
                continue  # negligible change
            # Map change to recommendation
            title, description, category = self.templates.get_template(field)
            # Determine priority based on change magnitude and target effect
            priority = self._estimate_priority(change, solution.distance_to_target)
            difficulty = self.templates.get_difficulty(change)
            # Confidence is DERIVED from the optimizer's distance to target
            # (real output): closer to target => higher confidence. Never a
            # hard-coded 0.7.
            distance = float(getattr(solution, "distance_to_target", 0.0) or 0.0)
            confidence = max(0.0, min(1.0, 1.0 - (distance / 10.0)))
            # Estimated gains are NOT fabricated: a baseline prediction for the
            # original state is unavailable here, so gains are explicitly
            # marked as assumptions (None) rather than presented as derived.
            # The solution's predicted absolute OH/NPS ARE real and preserved.
            predicted_oh = getattr(solution, "predicted_operations_health", None)
            predicted_nps = getattr(solution, "predicted_nps", None)
            rec = Recommendation(
                id=f"rec-{uuid.uuid4().hex[:8]}",
                title=title,
                description=description,
                category=category,
                priority=priority,
                difficulty=difficulty,
                estimated_operations_health_gain=None,
                estimated_nps_gain=None,
                estimated_disruption=0.0,
                confidence=confidence,
                actions=self.templates.get_actions(field),
                reasoning=(
                    f"Optimizer adjusts {field} by {change:.2f} units toward the "
                    "target (confidence derived from distance-to-target)."
                ),
                optimization_score=solution.optimization_score,
                target_kpi=field,
                direction="increase" if change > 0 else "decrease",
                magnitude=abs(change),
                metadata={
                    "field": field,
                    "gain_basis": "assumption_baseline_oh_unavailable",
                    "predicted_operations_health": predicted_oh,
                    "predicted_nps": predicted_nps,
                    "optimization_distance": distance,
                },
            )
            recommendations.append(rec)
        return recommendations

    def _estimate_priority(self, change: float, distance: float) -> Priority:
        # Critical if change is large and distance is large
        if abs(change) > 5 and distance > 10:
            return Priority.CRITICAL
        elif abs(change) > 3 and distance > 5:
            return Priority.HIGH
        elif abs(change) > 1:
            return Priority.MEDIUM
        else:
            return Priority.LOW

    def _deduplicate(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Merge recommendations with same category."""
        merged = {}
        for rec in recommendations:
            key = rec.category
            if key not in merged:
                merged[key] = rec
            else:
                # Keep the one with higher impact or priority
                existing = merged[key]
                if rec.priority.value < existing.priority.value:  # higher priority (lower value)
                    merged[key] = rec
                elif rec.estimated_operations_health_gain and existing.estimated_operations_health_gain:
                    if rec.estimated_operations_health_gain > existing.estimated_operations_health_gain:
                        merged[key] = rec
        return list(merged.values())
generate = RecommendationEngine.generate
