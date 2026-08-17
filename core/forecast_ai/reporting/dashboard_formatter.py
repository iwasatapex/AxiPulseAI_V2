from typing import Dict, Any, List


class DashboardFormatter:
    """
    Phase 8:
    Analytics dashboard presentation layer.

    Does not modify forecast logic.
    Converts ForecastResult payload into human-readable views.
    """

    def __init__(self, ansi: bool = False):
        self.ansi = ansi

    def _color(self, text: str, code: str) -> str:
        if not self.ansi:
            return text
        return f"\033[{code}m{text}\033[0m"

    def _status(self, value, green, yellow):
        if value is None:
            return "N/A"

        if value >= green:
            return self._color("GOOD", "32")
        if value >= yellow:
            return self._color("WATCH", "33")

        return self._color("RISK", "31")

    def _bar(self, value, maximum=100, width=20):
        if value is None:
            return ""

        filled = int(
            max(0, min(value, maximum))
            / maximum
            * width
        )

        return (
            "█" * filled +
            "░" * (width - filled)
        )

    def _kpi_status(self, metric, value):
        if value is None:
            return "N/A"

        targets = {
            "quality": 87,
            "competency": 93,
            "attendance": 90,
            "release": 60,
            "transfer": 9,
        }

        target = targets.get(metric)

        if target is None:
            return ""

        if metric == "transfer":
            return "✓ Target" if value <= target * 1.05 else "⚠ High"

        return (
            "✓ Target"
            if value >= target * 0.95
            else "⚠ Below"
        )

    def format_plain(
        self,
        timeline: List[Dict[str, Any]]
    ) -> str:

        lines = []

        lines.append("=" * 58)
        lines.append("              AxiPulseAI FORECAST DASHBOARD")
        lines.append("=" * 58)
        lines.append("")

        lines.append(
            f"{'DATE':<12}"
            f"{'OH':<10}"
            f"{'NPS':<10}"
            f"{'CONF':<10}"
        )
        lines.append("-" * 58)

        for day in timeline:
            confidence = day.get("confidence", {})

            conf = None

            if isinstance(confidence, dict):
                conf = confidence.get(
                    "overall_confidence"
                )

            conf_text = (
                f"{conf:.0%}"
                if isinstance(conf, (int, float))
                else "N/A"
            )

            lines.append(
                f"{day.get('date',''):<12}"
                f"{day.get('operations_health',0):<10.2f}"
                f"{day.get('nps',0):<10.2f}"
                f"{conf_text:<10}"
            )

        lines.append("")
        lines.append("KPI PERFORMANCE")
        lines.append("-" * 58)

        if timeline:
            latest = timeline[-1]

            for key in [
                "quality",
                "competency",
                "release",
                "transfer"
            ]:
                value = latest.get(key)

                maximum = 20 if key == "transfer" else 100

                lines.append(
                    f"{key.upper():<14}"
                    f"{value:<8.2f}"
                    f"{self._bar(value, maximum)}  "
                    f"{self._kpi_status(key, value)}"
                )

        lines.append("")

        lines.append("TREND ANALYSIS")
        lines.append("-" * 58)

        if len(timeline) >= 2:
            first = timeline[0]
            last = timeline[-1]

            oh_change = (
                last.get("operations_health", 0)
                - first.get("operations_health", 0)
            )

            nps_change = (
                last.get("nps", 0)
                - first.get("nps", 0)
            )

            oh_arrow = "▲" if oh_change > 0 else "▼" if oh_change < 0 else "━"
            nps_arrow = "▲" if nps_change > 0 else "▼" if nps_change < 0 else "━"

            lines.append(
                f"OH TREND        {oh_arrow} "
                f"{oh_change:+.2f}"
            )

            lines.append(
                f"NPS TREND       {nps_arrow} "
                f"{nps_change:+.2f}"
            )

        lines.append("")
        lines.append("KPI GAP TO TARGET")
        lines.append("-" * 58)

        targets = {
            "quality": 87,
            "competency": 93,
            "release": 60,
            "transfer": 9,
        }

        if timeline:
            latest = timeline[-1]

            for metric, target in targets.items():
                value = latest.get(metric)

                if value is not None:
                    gap = value - target

                    lines.append(
                        f"{metric.upper():<14}"
                        f"{gap:+.2f}"
                    )

        lines.append("")
        lines.append("CONFIDENCE RANGE")
        lines.append("-" * 58)

        if timeline:
            confidences = []

            for day in timeline:
                c = day.get("confidence", {})

                if isinstance(c, dict):
                    score = c.get("overall_confidence")

                    if isinstance(score, (int, float)):
                        confidences.append(score)

            if confidences:
                lines.append(
                    f"START           {confidences[0]:.0%}"
                )
                lines.append(
                    f"END             {confidences[-1]:.0%}"
                )

        lines.append("")
        lines.append("RISK ANALYSIS")
        lines.append("-" * 58)

        risk_status = "LOW"

        if timeline:
            latest = timeline[-1]

            oh = latest.get("operations_health")
            release = latest.get("release")
            transfer = latest.get("transfer")

            if oh is not None and oh < 70:
                risk_status = "HIGH"
            elif release is not None and release < 55:
                risk_status = "MEDIUM"
            elif transfer is not None and transfer > 12:
                risk_status = "MEDIUM"

        lines.append(
            f"OVERALL RISK    {risk_status}"
        )

        lines.append("")
        lines.append("DRIVERS")
        lines.append("-" * 58)

        if timeline:
            latest = timeline[-1]

            if latest.get("operations_health", 0) >= 90:
                lines.append(
                    "✓ Operations Health improving"
                )

            if latest.get("competency", 0) >= 90:
                lines.append(
                    "✓ Competency stable"
                )

            if latest.get("release", 0) <= 60:
                lines.append(
                    "⚠ Release near threshold"
                )

            if latest.get("transfer", 0) >= 10:
                lines.append(
                    "⚠ Transfer pressure elevated"
                )

        lines.append("")
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 58)

        if timeline:
            latest = timeline[-1]

            if latest.get("transfer", 0) >= 10:
                lines.append(
                    "→ Monitor transfer trend"
                )
            else:
                lines.append(
                    "→ Maintain current operating state"
                )

            if latest.get("release", 0) <= 60:
                lines.append(
                    "→ Monitor release efficiency"
                )

        lines.append("=" * 58)

        return "\n".join(lines)

    def format_ansi(
        self,
        timeline: List[Dict[str, Any]]
    ) -> str:
        return self.format_plain(timeline)


# ---------------------------------------------------------------------------
# Module-level compatibility surface
# ---------------------------------------------------------------------------
def format_plain(*args, **kwargs):
    return DashboardFormatter().format_plain(*args, **kwargs)

def format_ansi(*args, **kwargs):
    return DashboardFormatter().format_ansi(*args, **kwargs)
