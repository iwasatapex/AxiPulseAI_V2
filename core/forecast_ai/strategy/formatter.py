"""
Strategy formatter – text, markdown, dict, JSON.
"""
from typing import List, Dict, Any
import json
from .models import StrategyPlan

class StrategyFormatter:
    @staticmethod
    def to_text(strategies: List[StrategyPlan]) -> str:
        lines = ["STRATEGIES"]
        for idx, s in enumerate(strategies, 1):
            lines.append(f"{idx}. {s.name} (Priority: {s.priority})")
            lines.append(f"   {s.description}")
            lines.append(f"   Category: {s.category.value}")
            lines.append(f"   Expected OH: {s.estimated_operations_health}")
            lines.append(f"   Expected NPS: {s.estimated_nps}")
            lines.append(f"   Duration: {s.estimated_duration_weeks} weeks")
            lines.append(f"   Risks: {', '.join(s.risks) if s.risks else 'None'}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def to_markdown(strategies: List[StrategyPlan]) -> str:
        lines = ["# Strategies\n"]
        for s in strategies:
            lines.append(f"## {s.name}")
            lines.append(f"*{s.description}*")
            lines.append(f"- **Category:** {s.category.value}")
            lines.append(f"- **Priority:** {s.priority}")
            lines.append(f"- **Expected OH:** {s.estimated_operations_health}")
            lines.append(f"- **Expected NPS:** {s.estimated_nps}")
            lines.append(f"- **Duration:** {s.estimated_duration_weeks} weeks")
            lines.append(f"- **Risks:** {', '.join(s.risks) if s.risks else 'None'}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def to_dict(strategies: List[StrategyPlan]) -> List[Dict[str, Any]]:
        result = []
        for s in strategies:
            result.append({
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "objective": s.objective,
                "category": s.category.value,
                "priority": s.priority,
                "expected_oh": s.estimated_operations_health,
                "estimated_nps": s.estimated_nps,
                "duration_weeks": s.estimated_duration_weeks,
                "complexity": s.estimated_complexity,
                "disruption": s.estimated_disruption,
                "confidence": s.confidence,
                "recommendations": s.recommendations,
                "milestones": [{"week": m.week, "title": m.title, "progress": m.expected_progress} for m in s.milestones],
                "risks": s.risks
            })
        return result

    @staticmethod
    def to_json(strategies: List[StrategyPlan]) -> str:
        return json.dumps(StrategyFormatter.to_dict(strategies), indent=2, default=str)
