"""Headless Streamlit harness for the Reverse Optimizer view (AppTest).

This script is driven by streamlit.testing.v1.AppTest and is NOT collected by
pytest (it does not match ``test_*.py``). It stubs the model selector and the
canonical reverse optimizer service so the real ``reverse_view.render()`` UI
lifecycle can be exercised deterministically (checkbox/target/family changes,
stale-result invalidation, no-op when no objective is selected).

The service is a faithful stand-in for ``svc.reverse_optimize_canonical``: it
records the exact keyword targets it received and returns a canonical-shaped
payload whose targets echo the active objectives.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.views import reverse_view as rv  # noqa: E402
from gui import model_selection as ms  # noqa: E402


# ---- deterministic model selector: always returns a known family ---------
def _selector(feature="reverse", **k):
    family = st.session_state.get("_rev_family", "FAM")
    return SimpleNamespace(
        family=family,
        oh_algorithm="RandomForest",
        nps_algorithm="CatBoost",
        short_label=f"{family} (RF/CB)",
    )


ms.render_model_selector = _selector


def _payload(targets):
    return {
        "success": True,
        "status": "Target reached within tolerance.",
        "target_oh": targets.get("target_oh"),
        "target_nps": targets.get("target_nps"),
        "predicted_oh": targets.get("target_oh", 95.0),
        "predicted_nps": targets.get("target_nps", 82.0),
        "distance": 0.01,
        "recommended_state": {"quality": 92.0, "attendance": 90.0},
        "state_changes": {"quality": 5.0},
        "candidates": [],
        "errors": [],
        "active_family": "FAM",
    }


def _fake_reverse(**kw):
    st.session_state["_rev_calls"] = st.session_state.get("_rev_calls", []) + [dict(kw)]
    return _payload(kw)


# Replace reverse_view's own ``svc`` binding with a minimal stub so the SHARED
# ``gui.services`` module is never mutated (no cross-test leakage). The view
# only ever calls ``svc.reverse_optimize_canonical`` from its service import.
rv.svc = SimpleNamespace(reverse_optimize_canonical=_fake_reverse)

rv.render()
