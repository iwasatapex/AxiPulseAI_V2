"""
Formatter for Sensitivity results – text, markdown, dict, JSON.
"""
from typing import List, Dict, Any
import json
from .models import SensitivityAnalysis

class SensitivityFormatter:
    @staticmethod
    def to_text(analyses: List[SensitivityAnalysis]) -> str:
        lines = ["Sensitivity Analysis (OH Impact)"]
        for a in analyses:
            lines.append(f"\n{a.metric}:")
            lines.append(f"  OH Change: {a.operations_health_change:+.2f}")
            lines.append(f"  Sensitivity (OH): {a.sensitivity_score_oh:+.3f}")
            lines.append(f"  Elasticity (OH): {a.elasticity_oh:+.3f}")
            lines.append(f"  Classification: {a.classification}")
            lines.append(f"  NPS Change: {a.nps_change:+.2f}")
            lines.append(f"  Sensitivity (NPS): {a.sensitivity_score_nps:+.3f}")
            lines.append(f"  Elasticity (NPS): {a.elasticity_nps:+.3f}")
        return "\n".join(lines)

    @staticmethod
    def to_markdown(analyses: List[SensitivityAnalysis]) -> str:
        lines = ["# Sensitivity Analysis\n"]
        for a in analyses:
            lines.append(f"## {a.metric}")
            lines.append(f"- **OH Change:** {a.operations_health_change:+.2f}")
            lines.append(f"- **Sensitivity (OH):** {a.sensitivity_score_oh:+.3f}")
            lines.append(f"- **Elasticity (OH):** {a.elasticity_oh:+.3f}")
            lines.append(f"- **Classification:** {a.classification}")
            lines.append(f"- **NPS Change:** {a.nps_change:+.2f}")
            lines.append(f"- **Sensitivity (NPS):** {a.sensitivity_score_nps:+.3f}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def to_dict(analyses: List[SensitivityAnalysis]) -> List[Dict[str, Any]]:
        result = []
        for a in analyses:
            result.append({
                "metric": a.metric,
                "oh_change": a.operations_health_change,
                "nps_change": a.nps_change,
                "sensitivity_oh": a.sensitivity_score_oh,
                "sensitivity_nps": a.sensitivity_score_nps,
                "elasticity_oh": a.elasticity_oh,
                "elasticity_nps": a.elasticity_nps,
                "classification": a.classification,
                "rank": a.rank
            })
        return result

    @staticmethod
    def to_json(analyses: List[SensitivityAnalysis]) -> str:
        return json.dumps(SensitivityFormatter.to_dict(analyses), indent=2, default=str)
