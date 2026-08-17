"""Reusable rendering of training status for the Streamlit GUI.

These functions are deliberately framework-lite: they accept a duck-typed
``container`` so they can be unit-tested without an active Streamlit runtime.
The real GUI passes a ``st.empty()`` placeholder (or ``st.status``) container.

No percentage value is invented here. ``progress_bar_value`` returns ``None``
(rendered as an indeterminate spinner) whenever the status object reports an
indeterminate stage such as the final model fit.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def progress_bar_value(snapshot: Dict[str, Any]) -> Optional[float]:
    """Return a 0..1 progress fraction, or None for indeterminate stages.

    Only a real, non-None ``percent`` from the status object is used. A None
    percent (e.g. during final model fit) yields None so the GUI shows a
    spinner rather than a fake percentage.
    """
    pct = snapshot.get("percent")
    if pct is None:
        return None
    try:
        value = max(0.0, min(100.0, float(pct))) / 100.0
    except (TypeError, ValueError):
        return None
    return value


def format_elapsed(seconds: Optional[float]) -> str:
    """Format elapsed seconds as ``MM:SS`` (or ``H:MM:SS`` past an hour)."""
    if seconds is None:
        return "--:--"
    try:
        secs = int(float(seconds))
    except (TypeError, ValueError):
        return "--:--"
    if secs < 0:
        secs = 0
    hours, rem = divmod(secs, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_status_lines(snapshot: Dict[str, Any]) -> List[str]:
    """Return human-readable status lines for the current snapshot."""
    lines: List[str] = []
    stage_label = snapshot.get("stage_label") or snapshot.get("stage") or ""
    lines.append(f"**Stage:** {stage_label}")

    model = snapshot.get("current_model") or snapshot.get("model_name")
    if model:
        lines.append(f"**Model:** {model}")

    fold = snapshot.get("current_fold")
    total_folds = snapshot.get("total_folds")
    if fold is not None and total_folds:
        lines.append(f"**Fold:** {fold} / {total_folds}")

    completed = snapshot.get("completed_models")
    total_models = snapshot.get("total_models")
    if total_models:
        lines.append(f"**Models completed:** {completed or 0} / {total_models}")

    if snapshot.get("device"):
        lines.append(f"**Device:** {snapshot.get('device')}")

    rows = snapshot.get("rows")
    if rows is not None:
        lines.append(f"**Rows:** {int(rows):,}")

    hist = snapshot.get("history_days")
    if hist is not None:
        lines.append(f"**History days:** {int(hist)}")

    elapsed = format_elapsed(snapshot.get("elapsed_seconds"))
    lines.append(f"**Elapsed:** {elapsed}")

    message = snapshot.get("message")
    if message and message not in ("", "Complete", "Failed"):
        lines.append(message)

    error = snapshot.get("error")
    if error:
        lines.append(f"**Error:** {error}")

    return lines


def render_status(container: Any, snapshot: Dict[str, Any]) -> None:
    """Render a snapshot into a duck-typed Streamlit container.

    ``container`` must support:
      - ``progress(value, text=...)`` (value None => indeterminate),
      - ``markdown(text)``.
    """
    value = progress_bar_value(snapshot)
    stage = snapshot.get("stage") or ""

    if value is None:
        # Indeterminate stage (e.g. final model fit): show a spinner-like bar
        # plus the message rather than a fake percentage.
        message = snapshot.get("message") or (
            f"Working ({format_elapsed(snapshot.get('elapsed_seconds'))})..."
        )
        container.progress(0.0, text=message)
    else:
        if stage == "complete":
            text = "Complete"
        elif stage == "failed":
            text = "Training failed"
        else:
            text = f"{snapshot.get('stage_label') or stage}: {int(value * 100)}%"
        container.progress(value, text=text)

    for line in format_status_lines(snapshot):
        container.markdown(line)


def render_live(bar: Any, detail: Any, snapshot: Dict[str, Any]) -> None:
    """Update an already-created progress bar + text placeholder in place.

    This is the preferred API for the live Streamlit loop: it mutates a single
    ``bar`` object (no accumulating elements) and a ``detail`` placeholder.
    ``bar`` must expose ``progress(value, text=...)``; ``detail`` must expose
    ``markdown(text)``.
    """
    value = progress_bar_value(snapshot)
    stage = snapshot.get("stage") or ""
    label = snapshot.get("stage_label") or stage or ""

    if value is None:
        message = snapshot.get("message") or (
            f"Working ({format_elapsed(snapshot.get('elapsed_seconds'))})..."
        )
        bar.progress(0.0, text=message)
    elif stage == "complete":
        bar.progress(1.0, text="Complete")
    elif stage == "failed":
        bar.progress(1.0, text="Training failed")
    else:
        bar.progress(value, text=f"{label}: {int(value * 100)}%")

    detail.markdown("\n\n".join(format_status_lines(snapshot)))

