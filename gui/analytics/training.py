"""Training analytics: data quality + model-fit diagnostics.

Pure analysis functions consume a loaded dataset (pandas) and the
``train_models`` result dict. Rendering is a thin Streamlit layer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from gui.analytics import common as a

# Documented, conservative health thresholds (0-100 scale).
_DATA_QUALITY_GOOD = 95.0
_DATA_QUALITY_WARN = 80.0
_MODEL_FIT_GOOD = 0.70
_MODEL_FIT_WARN = 0.40


def _extract_metric(metrics: Dict[str, Any], metric: str):
    """Best-effort extraction of a metric from algorithm_performance dicts."""
    if not isinstance(metrics, dict):
        return None
    values = []
    for k, v in metrics.items():
        if k == metric and isinstance(v, (int, float)):
            values.append(float(v))
        elif isinstance(v, dict) and metric in v and isinstance(v[metric], (int, float)):
            values.append(float(v[metric]))
    if not values:
        return None
    # For per-algorithm MAE/RMSE/MAPE report the best (lowest) across
    # algorithms; for R2 report the best (highest).
    return (min(values) if metric in ("mae", "rmse", "mape") else max(values))


def model_metrics_analytics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract available OH/NPS model metrics without inventing any."""
    def _extract(side: str) -> Dict[str, Any]:
        metrics = result.get(f"{side}_metrics") or {}
        return {
            "algorithm": result.get(f"{side}_algorithm"),
            "features": result.get(f"{side}_features"),
            "mae": _extract_metric(metrics, "mae"),
            "rmse": _extract_metric(metrics, "rmse"),
            "r2": _extract_metric(metrics, "r2"),
            "mape": _extract_metric(metrics, "mape"),
            "raw": metrics,
        }
    return {"oh": _extract("oh"), "nps": _extract("nps")}


def training_health(profile: Dict[str, Any],
                    metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Compute conservative training health summary.

    Data Quality is derived from the dataset profile (missing + duplicate
    rates); Model Fit uses the best available R2; Generalization is reported
    as unknown unless the training layer exposes validation/test metrics.
    Thresholds are conservative and documented in the module docstring.
    """
    rows = max(profile.get("rows", 0), 1)
    missing_pct = 100.0 * profile.get("missing_total", 0) / rows
    dup_pct = 100.0 * profile.get("duplicate_rows", 0) / rows
    dq_score = max(0.0, 100.0 - missing_pct - dup_pct)
    dq_level = a.health_level(dq_score, _DATA_QUALITY_GOOD, _DATA_QUALITY_WARN)

    r2 = max(
        [m["r2"] for m in metrics.values() if m.get("r2") is not None] or [None]
    )
    if r2 is None:
        fit_level = "Unknown"
        fit_detail = a.NA
    else:
        fit_level = a.health_level(r2 * 100.0, _MODEL_FIT_GOOD * 100.0, _MODEL_FIT_WARN * 100.0)
        fit_detail = f"Best available R² = {r2:.3f}"

    # Generalization: engine does not expose validation/test splits here.
    gen_level = "Unknown"
    gen_detail = "No validation/test split exposed by the training layer."

    levels = [dq_level, fit_level]
    if "Poor" in levels:
        overall = "Poor"
    elif "Warning" in levels:
        overall = "Warning"
    elif "Unknown" in levels:
        overall = "Warning" if dq_level == "Warning" else "Unknown"
    elif dq_level == "Good" and fit_level == "Good":
        overall = "Good"
    else:
        overall = "Warning"

    return {
        "data_quality": {"level": dq_level, "score": round(dq_score, 1),
                         "detail": f"Missing {missing_pct:.1f}% · dup {dup_pct:.1f}%"},
        "model_fit": {"level": fit_level, "detail": fit_detail},
        "generalization": {"level": gen_level, "detail": gen_detail},
        "overall": overall,
    }


# ---------------------------------------------------------------------
# Rendering (thin Streamlit layer)
# ---------------------------------------------------------------------

def render_analytics(st, df: pd.DataFrame, result: Dict[str, Any],
                     nps_cols: Optional[List[str]] = None) -> None:
    """Render the full Training Analytics section."""
    import plotly.express as px

    nps_cols = nps_cols or ["promoters", "passives", "detractors"]
    profile = a.dataset_profile(df)
    quality = a.data_quality_report(df)
    metrics = model_metrics_analytics(result)
    health = training_health(profile, metrics)

    st.markdown("## Analytics")
    st.caption("Derived from the trained dataset and model-training output. "
               "Correlations are associations, not causation.")

    with st.expander("Overview", expanded=False):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", profile["rows"])
        m2.metric("Columns", profile["columns"])
        m3.metric("Duplicate rows", profile["duplicate_rows"])
        m4.metric("Memory (MB)", profile["memory_mb"])
        st.write("Numeric columns:", profile["n_numeric"],
                 "· Categorical columns:", profile["n_categorical"],
                 "· Missing cells:", profile["missing_total"])

    with st.expander("Data Quality", expanded=False):
        dq = pd.DataFrame(quality)
        st.dataframe(dq, width="stretch", hide_index=True)
        if not dq.empty:
            const = dq[dq["constant"]]
            if not const.empty:
                st.warning("Constant/near-constant features: " + ", ".join(const["feature"]))
        inv = a.invalid_numeric(df)
        if inv:
            st.warning("Non-numeric cells in numeric fields: " + str(inv))
        st.info(f"Overall data quality: **{health['data_quality']['level']}** — {health['data_quality']['detail']}")

    with st.expander("Target Analysis", expanded=False):
        # Only ever pick an actual Operational Health column, by name — never
        # silently substitute an unrelated numeric column (e.g. quality) under
        # the "Operational Health" label, and never fall back to numeric_cols[0].
        _OH_COL_CANDIDATES = ["operational_health", "operations_health", "oh"]
        oh_col = next((c for c in _OH_COL_CANDIDATES if c in df.columns), None)
        st.markdown("**Operational Health target**")
        if oh_col:
            desc = a.describe_target(df, oh_col)
            st.json(desc) if desc else st.info(a.NA)
        else:
            st.info("No operational_health column present in this dataset — "
                     "OH target analysis unavailable (not substituting another KPI).")
        st.markdown("**NPS survey distribution** (promoters/passives/detractors counts)")
        nps_dist = {c: int(df[c].sum()) if c in df.columns else 0 for c in nps_cols}
        if any(v for v in nps_dist.values()):
            st.json(nps_dist)
        else:
            st.info(a.NA)

    with st.expander("Feature Relationships", expanded=False):
        corr = a.numeric_correlation_matrix(df)
        if corr is not None:
            fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                            color_continuous_scale="RdBu_r",
                            title="Numeric feature correlation matrix")
            fig.update_layout(height=520)
            st.plotly_chart(fig, width="stretch", key="analytics_training_corr_matrix")
            st.caption("Correlation (linear, Pearson) — association only, not causation.")
        else:
            st.info("Correlation matrix not available — fewer than 2 numeric columns.")
        if oh_col:
            top = a.top_correlations(df, oh_col)
            if top:
                st.markdown("**Top correlations with OH proxy target**")
                st.dataframe(pd.DataFrame(top), width="stretch", hide_index=True)

    with st.expander("Model Metrics", expanded=False):
        for side, label in (("oh", "Operational Health"), ("nps", "NPS")):
            m = metrics[side]
            st.markdown(f"**{label}** — algorithm: {m['algorithm'] or '—'}, features: {m['features'] or '—'}")
            present = {k: v for k, v in {"MAE": m["mae"], "RMSE": m["rmse"],
                                         "R²": m["r2"], "MAPE": m["mape"]}.items() if v is not None}
            if present:
                st.dataframe(pd.DataFrame([present]), width="stretch", hide_index=True)
            else:
                st.info("No model-fit metrics (MAE/RMSE/R²/MAPE) exposed by the training layer.")

    with st.expander("Training Health", expanded=True):
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Data Quality", health["data_quality"]["level"])
        h2.metric("Model Fit", health["model_fit"]["level"])
        h3.metric("Generalization", health["generalization"]["level"])
        h4.metric("Overall", health["overall"])
        st.caption(f"Data Quality: {health['data_quality']['detail']} · "
                   f"Model Fit: {health['model_fit']['detail']} · "
                   f"Generalization: {health['generalization']['detail']}")
        st.caption("Thresholds (documented, conservative): Data Quality Good ≥95 / Warning ≥80; "
                   "Model Fit Good ≥0.70 R² / Warning ≥0.40 R². Unknown = metric not exposed.")
