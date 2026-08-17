"""Reusable Plotly chart builders for the GUI.

These are pure presentation helpers: they only convert already-computed
values (from the canonical V2 services) into chart figures. No prediction,
forecast, or decision logic lives here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from . import contracts as ct

try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except Exception:  # pragma: no cover - defensive
    _PLOTLY_AVAILABLE = False


def _fmt(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    return None


def forecast_timeline_chart(
    timeline: List[Dict[str, Any]],
    horizon: int,
) -> Any:
    """Return a Plotly figure of OH and NPS across the forecast days.

    Every day after day 0 is a *predicted* day.  Day 0 (when present) is
    the observed starting point.  Values are never relabelled.
    """
    if not _PLOTLY_AVAILABLE or not timeline:
        return None

    days = [i for i in range(len(timeline))]
    labels = [f"Day {i}" for i in range(len(timeline))]
    oh = [_fmt(d.get("operations_health") or d.get("operational_health")) for d in timeline]
    nps = [_fmt(d.get("nps")) for d in timeline]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=labels,
        y=oh,
        mode="lines+markers",
        name="Operational Health",
        line=dict(color="#22c55e", width=2),
        hovertemplate="%{x}<br>OH: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels,
        y=nps,
        mode="lines+markers",
        name="NPS",
        yaxis="y2",
        line=dict(color="#6366f1", width=2),
        hovertemplate="%{x}<br>NPS: %{y:.1f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"Forecast Timeline (H{horizon})", font=dict(size=16)),
        xaxis=dict(title="Day", tickangle=0),
        yaxis=dict(title="Operational Health (%)", range=[0, 120]),
        # NPS is -100..+100; never clamp it to [0,100] so negative NPS
        # remains visible.
        yaxis2=dict(title="NPS", overlaying="y", side="right",
                    range=[ct.NPS_MIN, ct.NPS_MAX]),
        legend=dict(orientation="h", y=1.12),
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    # Reference line at NPS 0 for legibility (unlabelled, thin).
    fig.add_hline(y=0, line=dict(color="rgba(99,102,241,0.35)", width=1),
                  yref="y2", layer="below")
    return fig


def nps_distribution_chart(distribution: Dict[str, Any]) -> Any:
    """Bar chart of the real NPS 0..10 score distribution (if present).

    The distribution keys are normalised via
    :func:`gui.contracts.normalize_nps_distribution` so every supported
    engine schema (``5``, ``"5"``, ``"score_5"``) renders correctly.
    Malformed data raises a clear ``ValueError`` rather than an obscure
    ``int(k)`` failure.
    """
    if not _PLOTLY_AVAILABLE or not distribution:
        return None

    scores_probs = ct.normalize_nps_distribution(distribution)
    if not scores_probs:
        return None

    ordered = sorted(scores_probs.items())
    scores = [s for s, _ in ordered]
    probs = [p for _, p in ordered]

    fig = go.Figure(go.Bar(
        x=scores,
        y=probs,
        name="Score probability",
        marker=dict(color="#6366f1"),
    ))
    fig.update_layout(
        title=dict(text="NPS 0..10 Posterior Distribution", font=dict(size=14)),
        xaxis=dict(title="Score", dtick=1),
        yaxis=dict(title="Probability"),
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def gauge_figure(
    value: Optional[float],
    title: str,
    max_value: float = 100.0,
    color: str = "#22c55e",
) -> Any:
    """Simple progress/gauge indicator."""
    if not _PLOTLY_AVAILABLE or value is None:
        return None
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value),
        number=dict(suffix="%" if max_value == 100.0 else "", font=dict(size=28)),
        gauge=dict(
            axis=dict(range=[0, max_value], tickwidth=1),
            bar=dict(color=color),
            bgcolor="rgba(0,0,0,0)",
        ),
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=13)), height=200, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def oh_sensitivity_chart(sensitivity: Dict[str, Any]) -> Any:
    """Render sensitivity output (best-effort) as a horizontal bar chart."""
    if not _PLOTLY_AVAILABLE or not sensitivity:
        return None

    # Sensitivity payloads vary; render any flat {label: value} mapping found.
    rows = []
    if isinstance(sensitivity, dict):
        for k, v in sensitivity.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, (int, float)):
                        rows.append((f"{k}:{kk}", float(vv)))
            elif isinstance(v, (int, float)):
                rows.append((str(k), float(v)))
    if not rows:
        return None

    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    rows = rows[:12]
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color="#f59e0b"),
    ))
    fig.update_layout(
        title=dict(text="Sensitivity (top factors)", font=dict(size=13)),
        xaxis=dict(title="Impact"),
        height=max(220, 32 * len(rows)),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def forecast_metric_chart(metrics: Dict[str, Any]) -> Any:
    """Simple chart of the forecast summary metrics if available."""
    if not _PLOTLY_AVAILABLE or not metrics:
        return None
    keys = [k for k in metrics.keys() if isinstance(metrics[k], (int, float))]
    if not keys:
        return None
    fig = go.Figure(go.Bar(
        x=keys,
        y=[metrics[k] for k in keys],
        marker=dict(color="#3b82f6"),
    ))
    fig.update_layout(
        title=dict(text="Forecast Summary", font=dict(size=13)),
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig
