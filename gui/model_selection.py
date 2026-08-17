"""Central GUI model-selection service.

One reusable picker drives EVERY GUI feature that runs a trained model
(Predict, Forecast, ADIE Decision, Target State, Reverse Optimizer).

Rules enforced here:

  * Only **complete, valid, loadable** OH+NPS model pairs are offered.
  * Smoke/test/staging pairs are excluded unless an explicit
    "Show test/staging models" option is enabled.
  * The chosen model is persisted **per feature** for the session.
  * A previously chosen model that disappeared is never silently replaced —
    the user is told and must pick again.
  * The selected family is passed **explicitly** into the GUI service layer
    (never discovered behind the scenes), and the session's active family is
    kept in sync only after a validated, visible choice.

The pure helpers (discovery / classification / formatting / persistence) are
Streamlit-free so they are unit-testable in headless contexts.  Only
``render_*`` touches ``streamlit``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gui import services as svc
from gui.state import STATE, _store

# ---------------------------------------------------------------------------
# Test/staging classification
# ---------------------------------------------------------------------------

# Markers that classify a model family as a smoke/test/staging artifact.
# Hidden files (leading dot, e.g. ``.training``) are always treated as
# staging/test too.
TEST_MODEL_MARKERS = ("smoke", "test", "staging", "tmp")

_SELECTION_PREFIX = "apgui_modelsel_"
_TEST_PREFIX = "apgui_modelsel_include_test_"


def is_test_family(family: Optional[str]) -> bool:
    """True for smoke/test/staging families (hidden unless explicitly enabled).

    Uses explicit artifact ROLE metadata from the manifest when available
    (``candidate`` / ``test`` / ``stress`` are treated as non-production), and
    only falls back to substring markers when no role metadata exists.
    """
    low = str(family or "").strip().lower()
    if low.startswith("."):
        return True

    role = _manifest_role_for_family(family)
    if role is not None:
        # Explicit role metadata is authoritative over substring heuristics.
        return role in {"test", "stress", "candidate"}

    return any(marker in low for marker in TEST_MODEL_MARKERS)


def _manifest_role_for_family(family: Optional[str]) -> Optional[str]:
    """Return the manifest role for a family, or None if not determinable.

    Reads ``manifest.json`` and looks up the family's OH/NPS artifact role.
    A plain family is reported ``production`` only when the manifest
    explicitly records its artifacts with role ``production``.
    """
    try:
        import json

        from core.forecast_ai.prediction.production_registry import (
            MANIFEST_NAME,
            MODELS_DIR,
            _artifact_role,
        )
        manifest_path = MODELS_DIR / MANIFEST_NAME
        if not manifest_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    f = str(family or "")
    for name in (f"{f}_OH.pkl", f"{f}_NPS.pkl"):
        entry = manifest.get(name)
        if isinstance(entry, dict) and entry.get("role"):
            return str(entry["role"]).lower()
    return None


# ---------------------------------------------------------------------------
# Model options
# ---------------------------------------------------------------------------


@dataclass
class ModelOption:
    """One selectable, validated OH+NPS model pair."""

    family: str
    oh: Dict[str, Any]
    nps: Dict[str, Any]
    oh_path: str
    nps_path: str
    is_test: bool
    saved_at: Optional[str] = None

    @property
    def oh_algorithm(self) -> str:
        return str(self.oh.get("model_name") or "OH")

    @property
    def nps_algorithm(self) -> str:
        return str(self.nps.get("model_name") or "NPS")

    @property
    def summary(self) -> str:
        return (
            f"Model: {self.oh_algorithm} — OH · {self.nps_algorithm} — NPS "
            f"· family `{self.family}`"
        )

    @property
    def short_label(self) -> str:
        return f"{self.family} — OH: {self.oh_algorithm} · NPS: {self.nps_algorithm}"


def discover_models(
    models: Optional[List[Dict[str, Any]]] = None,
    include_test: bool = False,
) -> List[ModelOption]:
    """Dynamically discover complete, valid, loadable model pairs.

    ``models`` accepts the raw output of ``svc.list_models()`` (tests inject
    synthetic listings this way); when omitted it is fetched live.  Only
    families whose OH and NPS bundles both inspected cleanly and are marked
    ``trained`` are offered.  Test/staging families are excluded unless
    ``include_test`` is True.
    """
    if models is None:
        models = svc.list_models()
    options: List[ModelOption] = []
    for m in models:
        if "error" in m:
            continue  # incomplete pair / not inspectable
        oh = m.get("oh") or {}
        nps = m.get("nps") or {}
        if "error" in oh or "error" in nps:
            continue  # unreadable bundle
        if not (oh.get("trained") and nps.get("trained")):
            continue  # not a usable trained model
        family = str(m.get("family") or "")
        is_test = is_test_family(family)
        if is_test and not include_test:
            continue
        options.append(
            ModelOption(
                family=family,
                oh=oh,
                nps=nps,
                oh_path=str(m.get("oh_path") or ""),
                nps_path=str(m.get("nps_path") or ""),
                is_test=is_test,
                saved_at=m.get("saved_at"),
            )
        )
    return sorted(options, key=lambda o: o.family)


def option_by_family(
    family: Optional[str], options: List[ModelOption]
) -> Optional[ModelOption]:
    for o in options:
        if o.family == family:
            return o
    return None


# ---------------------------------------------------------------------------
# Performance metrics (display only — never fabricated)
# ---------------------------------------------------------------------------


def performance_metrics(info: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Flatten ``algorithm_performance`` into (label, scalar) pairs.

    Handles both flat ``{algorithm: mae}`` and nested
    ``{algorithm: {metric: value}}`` shapes without inventing numbers.
    """
    perf = info.get("algorithm_performance") or {}
    if not isinstance(perf, dict):
        return []
    out: List[Tuple[str, float]] = []
    for key, value in perf.items():
        if isinstance(value, dict):
            for metric, v in value.items():
                if isinstance(v, (int, float)):
                    out.append((f"{key}.{metric}", float(v)))
        elif isinstance(value, (int, float)):
            out.append((str(key), float(value)))
    return out


def _metric_str(info: Dict[str, Any]) -> str:
    metrics = performance_metrics(info)
    if not metrics:
        return "—"
    label, value = metrics[0]
    return f"{label}={value:.4g}"


# ---------------------------------------------------------------------------
# Per-feature session persistence
# ---------------------------------------------------------------------------


def get_feature_selection(feature: str) -> Optional[str]:
    return _store().get(f"{_SELECTION_PREFIX}{feature}")


def set_feature_selection(feature: str, family: Optional[str]) -> None:
    if family is None:
        _store().pop(f"{_SELECTION_PREFIX}{feature}", None)
    else:
        _store()[f"{_SELECTION_PREFIX}{feature}"] = family


def get_include_test(feature: str) -> bool:
    return bool(_store().get(f"{_TEST_PREFIX}{feature}", False))


def set_include_test(feature: str, enabled: bool) -> None:
    _store()[f"{_TEST_PREFIX}{feature}"] = bool(enabled)


def reset_feature_selections() -> None:
    """Clear every per-feature model selection (used by tests)."""
    s = _store()
    for k in [
        k
        for k in list(s)
        if k.startswith(_SELECTION_PREFIX) or k.startswith(_TEST_PREFIX)
    ]:
        s.pop(k, None)


# ---------------------------------------------------------------------------
# Streamlit component
# ---------------------------------------------------------------------------


def render_model_selector(
    feature: str,
    label: str = "Trained model (explicit)",
    models: Optional[List[Dict[str, Any]]] = None,
) -> Optional[ModelOption]:
    """Render the reusable model picker; return the selected option or None.

    Returns ``None`` (and never runs) when nothing compatible is available —
    the zero-state explains the expected location/name and offers a training
    navigation action.
    """
    import streamlit as st

    include_test = st.checkbox(
        "Show test/staging models",
        value=get_include_test(feature),
        key=f"{_TEST_PREFIX}{feature}",
        help="Smoke/test/staging model pairs are hidden by default and are "
             "only offered when this is enabled.",
    )

    options = discover_models(models=models, include_test=include_test)

    if not options:
        _render_zero_state(feature)
        return None

    families = [o.family for o in options]
    persisted = get_feature_selection(feature)
    selection_invalidated = persisted is not None and persisted not in families

    if persisted in families:
        # Existing valid per-feature choice wins.
        default_idx = families.index(persisted)
    elif selection_invalidated:
        # A previously selected family disappeared. Do NOT silently replace it
        # with the global active family or the first available family. Force an
        # explicit new choice by starting with no selection.
        default_idx = None
        st.warning(
            f"Previously selected model `{persisted}` is no longer available. "
            f"Choose a new model explicitly below before running this feature."
        )
    elif STATE.get_active_family() in families:
        # No per-feature selection exists yet: the session-wide active family
        # is a valid initial choice. This is not a replacement because there is
        # no prior feature-specific choice being overwritten.
        default_idx = families.index(STATE.get_active_family())
    else:
        default_idx = 0

    chosen = st.selectbox(
        label,
        options=families,
        index=default_idx,
        format_func=lambda fam: _label(fam, options),
        placeholder=(
            "Choose a model explicitly…"
            if selection_invalidated
            else "Select a trained model…"
        ),
        help="Only complete, valid, loadable OH+NPS model pairs are offered. "
             "Selection is explicit — never silently replaced.",
    )

    if chosen is None:
        # Do not persist or activate anything until the user explicitly chooses
        # a replacement after a previously selected family disappears.
        return None

    if chosen != get_feature_selection(feature):
        set_feature_selection(feature, chosen)
        try:
            svc.select_model_family(chosen)
        except Exception as exc:  # noqa: BLE001 - GUI boundary
            st.error(f"Could not activate model {chosen}: {exc}")
            return None

    selected = option_by_family(chosen, options)
    if selected is not None:
        _render_details(selected)
    return selected


def render_result_model(family: Optional[str], option: Optional[ModelOption] = None) -> None:
    """Show which model produced an output, near the feature's output.

    Shows the full algorithm detail when the current selection matches the
    family that actually ran; otherwise falls back to the family name — it
    never claims a model was used that was not.
    """
    from gui import components as c

    if option is not None and option.family == family:
        c.model_badge(option.family, option.oh_algorithm, option.nps_algorithm, status="ready")
    else:
        import streamlit as st
        st.caption(f"Model: {family or '—'} (OH+NPS)")


def _render_zero_state(feature: str) -> None:
    import streamlit as st
    from gui import components as c

    c.empty_state(
        "No compatible trained model available. This feature cannot run "
        f"until a model is chosen. Expected a complete OH+NPS pair — "
        f"`{{family}}_OH.pkl` + `{{family}}_NPS.pkl` — in `{svc.MODELS_DIR}`. "
        f"Test/staging models are hidden unless 'Show test/staging models' is enabled.",
        icon="\U0001f9e9",
    )
    if st.button("Go to Train page \u2192", key=f"apgui_go_train_{feature}", type="primary"):
        st.session_state["apgui_go"] = "Train"
        st.rerun()


def _label(family: str, options: List[ModelOption]) -> str:
    option = option_by_family(family, options)
    return option.short_label if option else family


def _render_details(option: ModelOption) -> None:
    import streamlit as st
    from gui import components as c

    c.model_badge(option.family, option.oh_algorithm, option.nps_algorithm, status="ready")

    with st.expander("Selected model details", expanded=False):
        rows = [
            {
                "Predictor type": "OH (operational_health)",
                "Algorithm": option.oh_algorithm,
                "Features": option.oh.get("feature_count"),
                "Engine version": option.oh.get("engine_version") or "—",
                "Trained": "yes" if option.oh.get("trained") else "no",
                "Metric": _metric_str(option.oh),
                "Model file": Path(option.oh_path).name if option.oh_path else "—",
            },
            {
                "Predictor type": "NPS (nps)",
                "Algorithm": option.nps_algorithm,
                "Features": option.nps.get("feature_count"),
                "Engine version": option.nps.get("engine_version") or "—",
                "Trained": "yes" if option.nps.get("trained") else "no",
                "Metric": _metric_str(option.nps),
                "Model file": Path(option.nps_path).name if option.nps_path else "—",
            },
        ]
        st.dataframe(rows, width="stretch", hide_index=True)
