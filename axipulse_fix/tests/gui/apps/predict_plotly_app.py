"""AppTest fixture app: renders the Predict view with a pre-seeded result.

Reproduces the ``StreamlitDuplicateElementId`` scenario where the Predict view
and its Analytics panel both render the NPS distribution plotly chart in the
same script run. Uses the real ``gui.views.predict_view`` /
``gui.analytics.prediction`` production code paths with a fake model family so
no real model artifacts are loaded.
"""
import streamlit as st

from gui import services as svc

# Minimal fake model family so the view does not load real model artifacts.
svc.list_models = lambda: [{"family": "fake_family"}]

st.session_state["apgui_active_family"] = "fake_family"

_DIST = {
    "score_0": 0.0049, "score_1": 0.0049, "score_2": 0.0049,
    "score_3": 0.0049, "score_4": 0.0049, "score_5": 0.0034,
    "score_6": 0.0031, "score_7": 0.0364, "score_8": 0.0369,
    "score_9": 0.4014, "score_10": 0.4939,
}

st.session_state["predict_result"] = {
    "operational_health": 95.0,
    "nps": 85.48,
    "bayesian_score_distribution": _DIST,
    "promoters": 111,
    "passives": 8,
    "detractors": 5,
    "active_family": "fake_family",
    "_timestamp": "2026-08-15T00:00:00",
}
st.session_state["predict_state"] = {
    "quality": 87.0, "competency": 93.0, "attendance": 90.0,
    "release": 60.0, "transfer": 9.0,
    "operations_health": 95.0, "nps": 82.0,
    "total_calls_received": 2000.0,
}

from gui.views import predict_view  # noqa: E402

predict_view.render()
