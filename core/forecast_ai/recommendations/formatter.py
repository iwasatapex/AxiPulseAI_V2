"""
Format recommendations as plain text, markdown, or dictionary.
"""
from typing import List, Union, Dict, Any
from .models import Recommendation

class RecommendationFormatter:
    @staticmethod
    def to_text(recommendations: List[Recommendation]) -> str:
        lines = ["Recommendations:"]
        for idx, rec in enumerate(recommendations, 1):
            lines.append(f"{idx}. {rec.title}")
            lines.append(f"   Description: {rec.description}")
            lines.append(f"   Category: {rec.category.value}")
            lines.append(f"   Priority: {rec.priority.value}")
            lines.append(f"   Difficulty: {rec.difficulty.value}")
            if rec.estimated_operations_health_gain is not None:
                lines.append(f"   Expected OH gain: {rec.estimated_operations_health_gain:.2f}")
            if rec.estimated_nps_gain is not None:
                lines.append(f"   Expected NPS gain: {rec.estimated_nps_gain:.2f}")
            lines.append(f"   Reasoning: {rec.reasoning}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def to_markdown(recommendations: List[Recommendation]) -> str:
        lines = ["# Recommendations\n"]
        for rec in recommendations:
            lines.append(f"## {rec.title}")
            lines.append(f"*{rec.description}*")
            lines.append(f"- **Category:** {rec.category.value}")
            lines.append(f"- **Priority:** {rec.priority.value}")
            lines.append(f"- **Difficulty:** {rec.difficulty.value}")
            if rec.estimated_operations_health_gain is not None:
                lines.append(f"- **Expected OH gain:** {rec.estimated_operations_health_gain:.2f}")
            if rec.estimated_nps_gain is not None:
                lines.append(f"- **Expected NPS gain:** {rec.estimated_nps_gain:.2f}")
            lines.append(f"- **Reasoning:** {rec.reasoning}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def to_dict(recommendations: List[Recommendation]) -> List[Dict[str, Any]]:
        result = []
        for rec in recommendations:
            result.append({
                "id": rec.id,
                "title": rec.title,
                "description": rec.description,
                "category": rec.category.value,
                "priority": rec.priority.value,
                "difficulty": rec.difficulty.value,
                "estimated_oh_gain": rec.estimated_operations_health_gain,
                "estimated_nps_gain": rec.estimated_nps_gain,
                "estimated_disruption": rec.estimated_disruption,
                "confidence": rec.confidence,
                "actions": rec.actions,
                "reasoning": rec.reasoning,
                "optimization_score": rec.optimization_score,
                "metadata": rec.metadata
            })
        return result

to_text = RecommendationFormatter.to_text

to_markdown = RecommendationFormatter.to_markdown

to_dict = RecommendationFormatter.to_dict
