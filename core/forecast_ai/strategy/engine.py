"""
StrategyEngine – transforms RecommendationResult into operational strategies.
Pure transformation – no prediction, no optimization.
"""
import hashlib
from typing import List, Dict, Any
from .models import StrategyPlan, StrategyResult, StrategyCategory, Milestone
from .planner import StrategyPlanner
from .templates import StrategyTemplates
from .timeline import TimelineGenerator
from .scoring import StrategyScorer
from .formatter import StrategyFormatter
from ..recommendations import RecommendationResult

class StrategyEngine:
    def __init__(self):
        self.planner = StrategyPlanner()
        self.templates = StrategyTemplates()
        self.timeline = TimelineGenerator()
        self.scorer = StrategyScorer()
        self.formatter = StrategyFormatter()

    def generate(self, recommendation_result: RecommendationResult) -> StrategyResult:
        if not recommendation_result.success:
            return StrategyResult(
                success=False,
                strategies=[],
                best_strategy=None,
                warnings=[],
                errors=["Recommendation generation failed, cannot create strategies."]
            )

        recommendations = recommendation_result.recommendations
        if not recommendations:
            return StrategyResult(
                success=False,
                strategies=[],
                best_strategy=None,
                warnings=[],
                errors=["No recommendations available to build strategies."]
            )

        # Group recommendations by strategy category
        grouped = self.planner.group_recommendations(recommendations)

        # Generate strategy plan for each non-empty group
        recommendations_by_id = {r.id: r for r in recommendations}
        strategies = []
        for category, rec_ids in grouped.items():
            if not rec_ids:
                continue
            # Get template for this category
            template = self.templates.get_template(category)
            # Determine priority from average recommendation priority/score
            # For simplicity, we'll use the template's default or compute from recs
            priority = self.templates.get_priority(0.5)  # placeholder; could be improved

            # Build strategy plan
            # Deterministic ID
            id_seed = f"{category.value}_{sorted(rec_ids)}".encode('utf-8')
            strategy_id = hashlib.md5(id_seed).hexdigest()[:8]

            # Generate milestones
            milestones = self.timeline.generate(template.get('default_milestones', []))

            # Phase 11: strategy estimates are DERIVED from the real optimized
            # recommendations when available; otherwise explicitly left as
            # assumptions (None) — never hard-coded 80.0 / 70.0 / 0.7.
            group_recs = [
                recommendations_by_id[rid] for rid in rec_ids
                if rid in recommendations_by_id
            ]
            predicted_oh_values = [
                (r.metadata or {}).get("predicted_operations_health")
                for r in group_recs
                if (r.metadata or {}).get("predicted_operations_health") is not None
            ]
            predicted_nps_values = [
                (r.metadata or {}).get("predicted_nps")
                for r in group_recs
                if (r.metadata or {}).get("predicted_nps") is not None
            ]
            estimated_oh = (
                float(max(predicted_oh_values))
                if predicted_oh_values
                else None
            )
            estimated_nps = (
                float(max(predicted_nps_values))
                if predicted_nps_values
                else None
            )
            rec_confidences = [float(r.confidence) for r in group_recs if r.confidence is not None]
            if rec_confidences:
                confidence = sum(rec_confidences) / len(rec_confidences)
            else:
                # No real recommendation confidence; marked as an assumption.
                confidence = 0.5

            assumption_notes = []
            if estimated_oh is None:
                assumption_notes.append("estimated_operations_health: not derived (no baseline prediction)")
            if estimated_nps is None:
                assumption_notes.append("estimated_nps: not derived (no baseline prediction)")
            if not rec_confidences:
                assumption_notes.append("confidence: default assumption (no recommendation confidence available)")

            strategy = StrategyPlan(
                id=f"strat-{strategy_id}",
                name=template.get('name', f"{category.value.title()} Strategy"),
                description=template.get('description', ''),
                objective=template.get('objective', 'Improve operational performance.'),
                category=category,
                priority=priority,
                estimated_operations_health=estimated_oh,
                estimated_nps=estimated_nps,
                estimated_duration_weeks=template.get('estimated_duration_weeks', 4),
                estimated_complexity=template.get('estimated_complexity', 0.5),
                estimated_disruption=template.get('estimated_disruption', 0.3),
                confidence=confidence,
                recommendations=rec_ids,
                timeline=milestones,
                milestones=milestones,
                risks=template.get('default_risks', []),
                metadata={
                    "template": category.value,
                    "assumptions": assumption_notes,
                    "derived": {
                        "estimated_operations_health": estimated_oh,
                        "estimated_nps": estimated_nps,
                        "confidence": confidence,
                    },
                }
            )
            strategies.append(strategy)

        if not strategies:
            return StrategyResult(
                success=False,
                strategies=[],
                best_strategy=None,
                warnings=["No strategies could be generated from recommendations."],
                errors=[]
            )

        # Rank strategies
        ranked = self.scorer.rank(strategies)
        best = ranked[0] if ranked else None

        return StrategyResult(
            success=True,
            strategies=ranked,
            best_strategy=best,
            warnings=[],
            errors=[],
            metadata={"total_strategies": len(ranked)}
        )
