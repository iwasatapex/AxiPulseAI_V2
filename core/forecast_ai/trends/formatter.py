"""
Trend formatter – text, markdown, dict, JSON.
"""
from typing import List, Dict, Any
import json
from .models import TrendAnalysis

class TrendFormatter:
    @staticmethod
    def to_text(analyses: List[TrendAnalysis]) -> str:
        lines = ["KPI Trend Analysis"]
        for a in analyses:
            lines.append(f"\n{a.metric}:")
            lines.append(f"  Direction: {a.trend_direction}")
            lines.append(f"  Strength: {a.trend_strength}")
            lines.append(f"  Pattern: {a.pattern}")
            lines.append(f"  Volatility: {a.volatility}")
            lines.append(f"  Mean: {a.mean:.2f}")
            lines.append(f"  Median: {a.median:.2f}")
            lines.append(f"  Min: {a.minimum:.2f}")
            lines.append(f"  Max: {a.maximum:.2f}")
            lines.append(f"  Change: {a.absolute_change:.2f} ({a.percent_change:.1f}%)")
        return "\n".join(lines)

    @staticmethod
    def to_markdown(analyses: List[TrendAnalysis]) -> str:
        lines = ["# KPI Trend Analysis\n"]
        for a in analyses:
            lines.append(f"## {a.metric}")
            lines.append(f"- **Direction:** {a.trend_direction}")
            lines.append(f"- **Strength:** {a.trend_strength}")
            lines.append(f"- **Pattern:** {a.pattern}")
            lines.append(f"- **Volatility:** {a.volatility}")
            lines.append(f"- **Mean:** {a.mean:.2f}")
            lines.append(f"- **Median:** {a.median:.2f}")
            lines.append(f"- **Min / Max:** {a.minimum:.2f} / {a.maximum:.2f}")
            lines.append(f"- **Change:** {a.absolute_change:.2f} ({a.percent_change:.1f}%)")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def to_dict(analyses: List[TrendAnalysis]) -> List[Dict[str, Any]]:
        result = []
        for a in analyses:
            result.append({
                "metric": a.metric,
                "direction": a.trend_direction,
                "strength": a.trend_strength,
                "pattern": a.pattern,
                "volatility": a.volatility,
                "mean": a.mean,
                "median": a.median,
                "min": a.minimum,
                "max": a.maximum,
                "std_dev": a.standard_deviation,
                "variance": a.variance,
                "absolute_change": a.absolute_change,
                "percent_change": a.percent_change,
                "confidence": a.confidence,
                "moving_average": a.moving_average
            })
        return result

    @staticmethod
    def to_json(analyses: List[TrendAnalysis]) -> str:
        return json.dumps(TrendFormatter.to_dict(analyses), indent=2, default=str)
