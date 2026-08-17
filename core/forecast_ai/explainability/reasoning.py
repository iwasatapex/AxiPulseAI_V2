"""
Reasoning builder – composes deterministic narrative reasoning.
"""
from typing import List, Dict, Any
from .models import Evidence

class ReasoningBuilder:
    @staticmethod
    def build_reasoning(component: str, evidence: List[Evidence],
                        template: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """
        Build a narrative reasoning string from evidence and template.
        """
        # Extract key evidence descriptions
        evidence_text = ". ".join([f"{e.field}: {e.description}" for e in evidence if e.importance in ['High', 'Medium']])
        if not evidence_text:
            evidence_text = "No strong evidence available."

        # Format the reasoning using the template's reasoning_template
        reasoning_template = template.get('reasoning_template', "Reasoning: {evidence}")
        # Fill placeholders
        reasoning = reasoning_template.format(
            evidence=evidence_text,
            direction=metadata.get('direction', 'unknown'),
            horizon=metadata.get('horizon', 'N/A'),
            metric=metadata.get('metric', ''),
            strength=metadata.get('strength', ''),
            volatility=metadata.get('volatility', ''),
            pattern=metadata.get('pattern', ''),
            classification=metadata.get('classification', ''),
            score=metadata.get('score', 0.0),
            priority=metadata.get('priority', ''),
            areas=metadata.get('areas', ''),
            count=metadata.get('count', 0),
            best=metadata.get('best', ''),
            reasons=metadata.get('reasons', ''),
            drivers=metadata.get('drivers', ''),
            details=metadata.get('details', ''),
            top_risks=metadata.get('top_risks', ''),
            critical_risks=metadata.get('critical_risks', ''),
            top_actions=metadata.get('top_actions', ''),
            overload=metadata.get('overload', ''),
            low_components=metadata.get('low_components', ''),
            additional=metadata.get('additional', '')
        )
        return reasoning

# Module-level compatibility surface.
# Delegates to the existing implementation; no logic changed.
build_reasoning = ReasoningBuilder.build_reasoning
