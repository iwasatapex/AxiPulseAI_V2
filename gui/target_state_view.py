"""Target State view: multi-target reverse optimization via the TargetStateEngine."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from gui import components as c
from gui import contracts as ct
from gui import services as svc

# Ordered field definitions: (label, target key, contract key, default, help)
# Hard bounds are pulled from the canonical contracts (not re-listed here).
TARGET_FIELDS = [
    ("Operational Health", "operational_health", "operations_health",
     "Target operational health index (0–100)."),
    ("NPS", "nps", "nps",
     "Target Net Promoter Score (−100 to +100)."),
    ("Release Rate", "release", "release",
     "Target release rate % (canonical hard range 50–100; never below 50)."),
    ("Transfer Rate", "transfer", "transfer",
     "Target transfer rate % (canonical hard range 0–20; never above 20)."),
    ("Quality", "quality", "quality",
     "Target quality % (canonical hard range 60–100)."),
    ("Competency", "competency", "competency",
     "Target competency % (canonical hard range 55–100)."),
    ("Attendance", "attendance", "attendance",
     "Target attendance % (canonical hard range 65–100)."),
]

STATE_LABELS = {
    "operational_health": "Operational Health",
    "nps": "NPS",
    "release": "Release Rate",
    "transfer": "Transfer Rate",
    "quality": "Quality",
    "competency": "Competency",
    "attendance": "Attendance",
    "total_calls_received": "Total Calls Received",
}


def render() -> None:
    c.page_title("Target State", eyebrow="Optimization", help_text="Multi-target reverse optimization")

    from gui import model_selection as ms

    c.section("Model", "\U0001f9ec")
    option = ms.render_model_selector(feature="target_state")
    family = option.family if option is not None else None

    st.caption(
        "Set one or more targets; the engine searches the model council for the "
        "operational state that best hits them. Runs the canonical **TargetStateEngine** "
        "and may take a minute or two. Missing targets are ignored."
    )

    c.section("Targets", "\U0001f6a9")
    with st.form("target_state_form"):
        st.caption("Leave a field blank to ignore it — the engine optimises only the fields you set.")
        values = {}
        cols = st.columns(2)
        for i, (label, key, ck, help_text) in enumerate(TARGET_FIELDS):
            col = cols[i % 2]
            default = ct.kpi_default(ck)
            values[key] = col.text_input(
                label,
                value="" if key == "nps" else str(default),
                placeholder=str(default),
                help=help_text,
            )
        submitted = st.form_submit_button("\u25b6  Run Target State Search", type="primary",
                                          disabled=option is None, width="stretch")

    if submitted and family:
        targets = {}
        for _, key, ck, _ in TARGET_FIELDS:
            raw = values[key].strip()
            if raw == "":
                continue  # blank = ignore that target
            try:
                val = float(raw)
            except ValueError:
                st.warning(f"Invalid target for {key!r}: {raw!r}. Ignored.")
                continue
            targets[key] = val
        if not targets:
            st.error("Provide at least one target to search.")
            return

        with st.spinner("Searching model council for the required operational state\u2026"):
            result = c.guarded(svc.find_target_state, targets, family=family)
        if result:
            st.session_state["target_state_result"] = result

    result = st.session_state.get("target_state_result")
    if not result:
        st.write("")
        c.empty_state("Set targets above and run a search to see the recommended state.",
                      icon="\U0001f6a9")
        return

    st.divider()
    c.section("Recommended operational state", "\u2705")
    ms.render_result_model(result.get("active_family"), option)
    rec = result.get("recommended_state") or {}
    consensus = result.get("consensus") or {}
    if rec:
        _render_recommended(rec, consensus, result.get("distance"))
    else:
        st.error("No recommended state found.")

    boards = result.get("leaderboards") or {}
    if boards:
        with st.expander("Model council leaderboards", expanded=False):
            col_oh, col_nps = st.columns(2)
            _render_leaderboard(col_oh, "Operational Health", boards.get("OH", []))
            _render_leaderboard(col_nps, "NPS", boards.get("NPS", []))

    st.caption(f"Targets: {result.get('targets')} \u00b7 {result.get('_timestamp', '')[:19]}")
    c.raw_json_expander(result)

    # ---- Analytics ----
    st.divider()
    c.section("Supporting analytics", "\U0001f4d1")
    from gui.analytics import target_state as _ta
    _ta.render_analytics(st, result)


def _render_recommended(rec: dict, consensus: dict, distance) -> None:
    m1, m2, m3 = st.columns(3)
    with m1:
        oh_v = consensus.get("oh")
        c.kpi_tile("Achieved OH", f"{oh_v:.2f}" if oh_v is not None else "\u2014", status="ready" if oh_v is not None else "none")
    with m2:
        nps_v = consensus.get("nps")
        c.kpi_tile("Achieved NPS", f"{nps_v:.2f}" if nps_v is not None else "\u2014", status="ready" if nps_v is not None else "none")
    with m3:
        c.kpi_tile("Distance", f"{distance:.3f}" if distance is not None else "\u2014", status="none")

    rows = []
    for key, label in STATE_LABELS.items():
        if key in rec:
            rows.append({"KPI": label, "Value": rec[key]})
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)

    c.raw_json_expander(rec, label="Recommended state JSON")


def _render_leaderboard(col, title: str, rows) -> None:
    col.markdown(f"**{title}**")
    if not rows:
        col.write("Unavailable")
        return
    data = pd.DataFrame(rows)
    keep = [k for k in ("model", "prediction", "confidence", "status") if k in data.columns]
    if not keep:
        data = data.astype(str)
        col.json(data.to_dict(orient="records"))
        return
    col.dataframe(data[keep], width="stretch", hide_index=True)
