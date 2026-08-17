"""ADIE / Decision Intelligence analytics.

Pure functions consume the ``adie_decision`` payload and explain the
decision from actual engine output — never an LLM, never fabrication.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from gui.analytics import common as a


def _package(result: Dict[str, Any]) -> Dict[str, Any]:
    di = result.get("decision_intelligence") or {}
    if isinstance(di, dict) and isinstance(di.get("package"), dict):
        return di["package"]
    if isinstance(result.get("decision"), dict):
        return result["decision"]
    return {}


def _details(result: Dict[str, Any]) -> Dict[str, Any]:
    """Canonical enriched ADIE detail (recommendations, risk_detail, …)."""
    di = result.get("decision_intelligence") or {}
    if isinstance(di, dict) and isinstance(di.get("details"), dict):
        return di["details"]
    if isinstance(result.get("details"), dict):
        return result["details"]
    return {}


def _recommendations(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Top-ranked recommendations: prefer the enriched detail list, then the
    raw recommendation-output list, never a dict-shaped status block."""
    recs = _details(result).get("recommendations") or []
    if recs:
        return recs
    raw = _package(result).get("recommendations") or []
    if isinstance(raw, dict):
        raw = raw.get("recommendations") or []
    return raw if isinstance(raw, list) else []


def decision_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    recs = _recommendations(result)
    first = recs[0] if isinstance(recs, list) and recs else {}
    return {
        "action": a.safe_get(first, "action") or "No recommendation produced",
        "affected_kpi": a.safe_get(first, "affected_kpi"),
        "direction": a.safe_get(first, "direction"),
        "confidence": a.fnum(a.safe_get(first, "confidence")),
        "risk": a.safe_get(first, "risk"),
        "num_recommendations": len(recs) if isinstance(recs, list) else 0,
    }


def decision_drivers(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rank actual factors used by ADIE if the engine exposed contributions.

    Falls back to the real normalized ranking components of the top-ranked
    scenario (evidence.components) when no explicit contribution surface is
    exposed — both are genuine engine output, never fabricated.
    """
    details = _details(result)
    contributions = a.safe_get(details, "contributions") or a.safe_get(details, "factor_contributions")
    if isinstance(contributions, dict):
        rows = []
        for k, v in contributions.items():
            val = a.fnum(v)
            if val is not None:
                rows.append({"factor": k, "contribution": val,
                             "direction": "positive" if val > 0 else ("negative" if val < 0 else "neutral")})
        rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
        return rows
    # Canonical ranking components of the preferred scenario (performance,
    # confidence, safety, momentum — each normalized to [0,1]).
    best = details.get("best_scenario") or {}
    evidence = best.get("evidence") or {}
    comps = evidence.get("components")
    if isinstance(comps, dict):
        rows = []
        for k, v in comps.items():
            val = a.fnum(v)
            if val is not None:
                rows.append({"factor": k, "contribution": val,
                             "direction": "positive" if val > 0 else ("negative" if val < 0 else "neutral")})
        rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
        return rows
    return []


def risk_analysis(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Canonical risk surface: the enriched risk_detail first, then any
    package/result risk payload."""
    risk_detail = _details(result).get("risk_detail")
    if isinstance(risk_detail, dict) and risk_detail:
        return [{
            "level": risk_detail.get("level"),
            "score": a.fnum(risk_detail.get("score")),
            "confidence": a.fnum(risk_detail.get("confidence")),
            "downside": a.fnum(risk_detail.get("downside")),
            "upside": a.fnum(risk_detail.get("upside")),
            "abstain": risk_detail.get("abstain"),
        }]
    pkg = _package(result)
    risks = pkg.get("risks") or a.safe_get(result, "risk") or []
    if isinstance(risks, list):
        return risks
    if isinstance(risks, dict):
        return [risks]
    return []


def recommendation_quality(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    recs = _recommendations(result)
    out = []
    if not isinstance(recs, list):
        return out
    for r in recs[:5]:
        eff = a.safe_get(r, "expected_effect") or {}
        out.append({
            "action": a.safe_get(r, "action"),
            "confidence": a.fnum(a.safe_get(r, "confidence")),
            "affected_kpi": a.safe_get(r, "affected_kpi"),
            "expected_oh": a.safe_get(eff, "oh_gain") or a.safe_get(eff, "oh_lift"),
            "expected_nps": a.safe_get(eff, "nps_gain") or a.safe_get(eff, "nps_lift"),
            "evidence_count": len(a.safe_get(r, "evidence") or []) if isinstance(a.safe_get(r, "evidence"), list) else None,
        })
    return out


def explainability_text(result: Dict[str, Any]) -> List[str]:
    """Concise deterministic explanation from actual data."""
    lines: List[str] = []
    summary = decision_summary(result)
    if summary["action"] != "No recommendation produced":
        lines.append(
            f"ADIE recommends '{summary['action']}' targeting "
            f"{summary['affected_kpi'] or 'unknown KPI'} "
            f"(direction {summary['direction'] or 'unspecified'})."
        )
    drivers = decision_drivers(result)
    if drivers:
        top = drivers[0]
        lines.append(f"Top decision driver: {top['factor']} "
                     f"({top['direction']}, contribution {top['contribution']:.3f}).")
    risks = risk_analysis(result)
    if risks:
        lines.append(f"{len(risks)} risk(s) flagged by the decision package.")
    else:
        lines.append("No risks flagged in the decision output.")
    return lines


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------

def render_analytics(st, result: Dict[str, Any]) -> None:
    st.markdown("## Analytics")
    st.caption("Decision explanation derived from the canonical ADIE V3 output.")

    with st.expander("Decision Summary", expanded=True):
        s = decision_summary(result)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Action", str(s["action"])[:40])
        m2.metric("Affected KPI", s["affected_kpi"] or "—")
        m3.metric("Direction", s["direction"] or "—")
        m4.metric("Confidence", f"{s['confidence']:.2f}" if s["confidence"] is not None else "—")

    with st.expander("Decision Drivers", expanded=False):
        drivers = decision_drivers(result)
        if drivers:
            import pandas as pd
            st.dataframe(pd.DataFrame(drivers), width="stretch", hide_index=True)
        else:
            st.info("Factor contributions not exposed by the ADIE engine.")

    with st.expander("Risk Analysis", expanded=False):
        risks = risk_analysis(result)
        if risks:
            import pandas as pd
            st.dataframe(pd.DataFrame(risks), width="stretch", hide_index=True)
        else:
            st.info("No risk output exposed by the ADIE engine.")

    with st.expander("Recommendation Quality", expanded=False):
        recs = recommendation_quality(result)
        if recs:
            import pandas as pd
            st.dataframe(pd.DataFrame(recs), width="stretch", hide_index=True)
        else:
            st.info("No recommendation-quality details exposed by the ADIE engine.")

    with st.expander("Interpretation", expanded=False):
        for line in explainability_text(result):
            st.markdown(f"- {line}")
