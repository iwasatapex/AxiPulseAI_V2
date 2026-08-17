"""Session-scoped GUI state.

Holds the currently selected model family and recent results for the
*dashboard* of a single Streamlit session.  State is stored in
``st.session_state`` (namespaced keys) so one browser session's model
selection never leaks into another session.

This is NOT a model artifact.  The active family is only a *session*
preference.  The canonical ``core.forecast_ai.prediction.PredictorProvider``
is intentionally NOT mutated here: activating a provider family happens
inside the GUI service layer, serialised per request (see
``gui.services``) so concurrent sessions cannot race on the shared
process-global provider.

When Streamlit is unavailable (unit tests / headless contexts) a module
level fallback store is used.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from core.forecast_ai.prediction.model_selector import list_model_families

# Fallback store used when Streamlit is not available (tests, headless).
_FALLBACK: Dict[str, Any] = {}


def _store() -> Dict[str, Any]:
    """Return the Streamlit session-state store or the fallback store."""
    try:
        import streamlit as st
        return st.session_state
    except Exception:  # pragma: no cover - non-streamlit context
        return _FALLBACK


def _key(name: str) -> str:
    return f"apgui_{name}"


class GUIState:
    """Session-scoped holder for the active model family + recent results."""

    # ---------------- model family ----------------
    def set_active_family(self, family: Optional[str]) -> None:
        previous = self.get_active_family()
        _store()[_key("active_family")] = family
        if previous != family:
            # A family change invalidates any stored result computed under a
            # DIFFERENT family.  Displaying a prior-family result under a new
            # selection would misattribute model output.  (The new result is
            # produced and stored on the next predict/forecast.)
            for key in ("last_prediction", "last_forecast", "last_adie"):
                payload = _store().get(_key(key))
                if isinstance(payload, dict):
                    produced_family = payload.get("active_family") or payload.get("family")
                    if produced_family is not None and produced_family != family:
                        _store()[_key(key)] = None

    def get_active_family(self) -> Optional[str]:
        return _store().get(_key("active_family"))

    def get_provider_family(self) -> Optional[str]:
        # The process-global provider is only meaningful at request time;
        # for display purposes the session active family is authoritative.
        return self.get_active_family()

    # ---------------- recent results ----------------
    def set_last_prediction(self, payload: Dict[str, Any]) -> None:
        _store()[_key("last_prediction")] = payload

    def get_last_prediction(self) -> Optional[Dict[str, Any]]:
        return _store().get(_key("last_prediction"))

    def set_last_forecast(self, payload: Dict[str, Any]) -> None:
        _store()[_key("last_forecast")] = payload

    def get_last_forecast(self) -> Optional[Dict[str, Any]]:
        return _store().get(_key("last_forecast"))

    def set_last_adie(self, payload: Dict[str, Any]) -> None:
        _store()[_key("last_adie")] = payload

    def get_last_adie(self) -> Optional[Dict[str, Any]]:
        return _store().get(_key("last_adie"))

    # ---------------- helpers ----------------
    def status(self) -> Dict[str, Any]:
        active = self.get_active_family()
        return {
            "active_family": active,
            "provider_family": active,
            "available_families": list_model_families(),
            "last_prediction_at": _stamp(self.get_last_prediction()),
            "last_forecast_at": _stamp(self.get_last_forecast()),
            "last_adie_at": _stamp(self.get_last_adie()),
        }

    def reset(self) -> None:
        """Clear this session's state (used by tests)."""
        s = _store()
        for k in (
            "active_family",
            "last_prediction",
            "last_forecast",
            "last_adie",
        ):
            s.pop(_key(k), None)


def _stamp(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    if not payload:
        return None
    return payload.get("_timestamp")


STATE = GUIState()

