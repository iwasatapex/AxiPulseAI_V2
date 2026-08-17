import json
from typing import List, Dict, Any
from .models import RiskResult

class RiskFormatter:
    @staticmethod
    def to_text(result: RiskResult) -> str:
        lines = [f"Risk Report\nOverall Risk: {result.overall_risk:.2%}\n"]
        for a in result.analyses:
            lines.append(f"{a.component.title()} Risk: {a.overall_risk:.2%} ({a.classification})")
            lines.append(f"  {a.summary}")
            for rf in a.risk_factors[:3]:
                lines.append(f"    - {rf.name}: {rf.risk_score:.2f}")
            lines.append("")
        if result.warnings:
            lines.append("Warnings:")
            for w in result.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)

    @staticmethod
    def to_markdown(result: RiskResult) -> str:
        lines = ["# Risk Report", f"**Overall Risk:** {result.overall_risk:.2%}\n"]
        for a in result.analyses:
            lines.append(f"## {a.component.title()} Risk")
            lines.append(f"- **Score:** {a.overall_risk:.2%}")
            lines.append(f"- **Classification:** {a.classification}")
            lines.append(f"- **Summary:** {a.summary}")
            if a.risk_factors:
                lines.append("- **Risk Factors:**")
                for rf in a.risk_factors:
                    lines.append(f"  - **{rf.name}**: {rf.risk_score:.2f} – {rf.mitigation}")
            lines.append("")
        if result.warnings:
            lines.append("## Warnings")
            for w in result.warnings:
                lines.append(f"- {w}")
        return "\n".join(lines)

    @staticmethod
    def to_dict(result: RiskResult) -> Dict[str, Any]:
        return {
            "success": result.success,
            "overall_risk": result.overall_risk,
            "analyses": [{
                "component": a.component,
                "overall_risk": a.overall_risk,
                "classification": a.classification,
                "summary": a.summary,
                "risk_factors": [{
                    "name": rf.name,
                    "risk_score": rf.risk_score,
                    "reason": rf.reason,
                    "mitigation": rf.mitigation
                } for rf in a.risk_factors]
            } for a in result.analyses],
            "warnings": result.warnings,
            "errors": result.errors,
            "metadata": result.metadata
        }

    @staticmethod
    def to_json(result: RiskResult) -> str:
        return json.dumps(RiskFormatter.to_dict(result), indent=2, default=str)


# ---------------------------------------------------------------------------
# Module-level compatibility surface
# ---------------------------------------------------------------------------
def to_text(*args, **kwargs):
    return RiskFormatter.to_text(*args, **kwargs)

def to_markdown(*args, **kwargs):
    return RiskFormatter.to_markdown(*args, **kwargs)

def to_dict(*args, **kwargs):
    return RiskFormatter.to_dict(*args, **kwargs)

def to_json(*args, **kwargs):
    return RiskFormatter.to_json(*args, **kwargs)
