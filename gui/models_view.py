"""Models view: list complete model families, select the active family, and
show a leaderboard-style comparison of OH/NPS performance per family."""
from __future__ import annotations

import streamlit as st

from gui import components as c
from gui import services as svc


def render() -> None:
    c.page_title("Models", eyebrow="Registry", help_text="Complete OH+NPS model families")

    models = svc.list_models()
    active = svc.STATE.get_active_family()

    if not models:
        c.empty_state(
            "No complete model pairs found. "
            "A family needs both `{name}_OH.pkl` and `{name}_NPS.pkl`.",
            icon="\U0001f5c2\ufe0f",
        )
        return

    # ---- Active selection ----------------------------------------------
    c.section("Active model family", "\U0001f3af")
    st.caption("Only families with **both** OH and NPS files are shown. "
               "Selection is explicit — never silent.")
    from gui import model_selection as ms
    include_test = st.checkbox(
        "Show test/staging models",
        value=bool(st.session_state.get("apgui_models_include_test", False)),
        key="legacy_apgui_models_include_test",
        help="Smoke/test/staging model pairs are hidden from selection unless "
             "this is enabled.",
    )
    options = [
        m["family"] for m in models
        if "error" not in m and (include_test or not ms.is_test_family(m["family"]))
    ]
    if not options:
        c.empty_state("No selectable model families. Enable test/staging models "
                      "above or train one on the Train page.", icon="\u26a0\ufe0f")
        return
    idx = options.index(active) if active in options else 0
    sel_col, badge_col = st.columns([2, 1])
    with sel_col:
        chosen = st.selectbox("Model family", options=options, index=idx)
    if chosen != active:
        try:
            svc.select_model_family(chosen)
            st.success(f"Active model family set to **{chosen}**")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not activate {chosen}: {exc}")
    else:
        chosen_info = next((m for m in models if m.get("family") == chosen), None)
        if chosen_info and "error" not in chosen_info:
            with badge_col:
                c.model_badge(
                    chosen,
                    (chosen_info.get("oh") or {}).get("model_name"),
                    (chosen_info.get("nps") or {}).get("model_name"),
                    status="active",
                )

    st.write("")

    # ---- Leaderboard ------------------------------------------------------
    c.section("Leaderboard", "\U0001f3c6")
    oh_rows, nps_rows = [], []
    for m in models:
        fam = m.get("family")
        is_active = "\u2605 active" if m.get("active") else ""
        if "error" in m:
            oh_rows.append({"Family": fam, "Error": m["error"]})
            continue
        oh = m.get("oh") or {}
        nps = m.get("nps") or {}
        oh_perf = ms.performance_metrics(oh)
        nps_perf = ms.performance_metrics(nps)
        oh_rows.append({
            "Family": fam,
            "Status": is_active,
            "Algorithm": oh.get("model_name"),
            "Error metric": oh_perf[0][1] if oh_perf else None,
            "Trained": "yes" if oh.get("trained") else "no",
        })
        nps_mae = next((v for k, v in nps_perf if "bucket" not in k.lower()), None)
        bucket_mae = next((v for k, v in nps_perf if "bucket" in k.lower()), None)
        nps_rows.append({
            "Family": fam,
            "Status": is_active,
            "Algorithm": nps.get("model_name"),
            "NPS MAE": nps_mae,
            "Bucket MAE": bucket_mae,
            "Trained": "yes" if nps.get("trained") else "no",
        })

    tab_oh, tab_nps = st.tabs(["Operations Health", "NPS"])
    with tab_oh:
        st.caption("Model | Error | Status — lower error is better.")
        st.dataframe(oh_rows, width="stretch", hide_index=True)
    with tab_nps:
        st.caption("NPS MAE and Bucket MAE are distinct metrics — a low bucket MAE "
                   "is never reported as NPS accuracy.")
        st.dataframe(nps_rows, width="stretch", hide_index=True)

    st.write("")

    # ---- Full table ----
    c.section("All families", "\U0001f4cb")
    rows = []
    for m in models:
        if "error" in m:
            rows.append({"Family": m["family"], "Status": f"Error: {m['error']}"})
            continue
        oh = m.get("oh") or {}
        nps = m.get("nps") or {}
        rows.append({
            "Family": m["family"],
            "Active": "\u2605" if m.get("active") else "",
            "OH model": oh.get("model_name"),
            "NPS model": nps.get("model_name"),
            "OH feats": oh.get("feature_count"),
            "NPS feats": nps.get("feature_count"),
            "OH hist": oh.get("history_days") if "history_days" in oh else (m.get("oh_trained_days")),
            "Saved": (m.get("saved_at") or "")[:19],
        })
    st.dataframe(rows, width="stretch", hide_index=True)

    # ---- Detail cards ----
    c.section("Family details", "\U0001f50e")
    for m in models:
        if "error" in m:
            continue
        with st.expander(f"{m['family']}  {'(active)' if m.get('active') else ''}", expanded=m.get("active")):
            _render_family(m)


def _render_family(m: dict) -> None:
    oh = m.get("oh") or {}
    nps = m.get("nps") or {}
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Operational Health**")
        c.kpi_tile("Algorithm", str(oh.get("model_name") or "\u2014"),
                    status="ready" if oh.get("trained") else "none")
        st.write(f"Features: {oh.get('feature_count')}")
        st.write(f"Engine: {oh.get('engine_version')}")
        perf = oh.get("algorithm_performance")
        if perf:
            st.caption("Algorithm performance:")
            st.json(perf)

    with col2:
        st.markdown("**NPS**")
        c.kpi_tile("Algorithm", str(nps.get("model_name") or "\u2014"),
                    status="ready" if nps.get("trained") else "none")
        st.write(f"Features: {nps.get('feature_count')}")
        st.write(f"Engine: {nps.get('engine_version')}")
        perf = nps.get("algorithm_performance")
        if perf:
            st.caption("Algorithm performance:")
            st.json(perf)

    st.caption(f"Saved: {m.get('saved_at', '')[:19]}")
    c.raw_json_expander({"family": m["family"], "oh": oh, "nps": nps})
