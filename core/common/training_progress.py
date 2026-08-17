"""
AxiPulseAI – Central training progress/status mechanism.

A single lightweight, thread-safe status object shared by the GUI and the
trainers. The background training thread mutates it via the mutator methods;
the Streamlit main thread reads immutable snapshots via :meth:`snapshot` and
renders them. No widget is ever touched from the background thread.

Progress semantics:
- ``percent`` is computed ONLY from real completed model / fold counts
  (never invented). It is ``None`` while a stage has no measurable units
  (e.g. indeterminate final model fit), so the GUI shows a spinner instead of
  a fake percentage.
- ``elapsed_seconds`` is derived from a monotonic clock, cached on snapshot.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# Canonical stage identifiers (see also STAGE_LABELS).
STAGE_LOADING = "loading"
STAGE_FEATURE_ENGINEERING = "feature_engineering"
STAGE_PREPARING_TARGETS = "preparing_targets"
STAGE_PREPARING_FEATURES = "preparing_features"
STAGE_MODEL_SELECTION = "model_selection"
STAGE_VALIDATION = "validation"
STAGE_FINAL_REFIT = "final_refit"
STAGE_SAVING = "saving"
STAGE_COMPLETE = "complete"
STAGE_FAILED = "failed"

# Human-friendly labels for each stage.
STAGE_LABELS: Dict[str, str] = {
    STAGE_LOADING: "Loading dataset",
    STAGE_FEATURE_ENGINEERING: "Feature engineering",
    STAGE_PREPARING_TARGETS: "Preparing targets",
    STAGE_PREPARING_FEATURES: "Preparing features",
    STAGE_MODEL_SELECTION: "Model validation",
    STAGE_VALIDATION: "Model validation",
    STAGE_FINAL_REFIT: "Final model fit",
    STAGE_SAVING: "Saving model",
    STAGE_COMPLETE: "Complete",
    STAGE_FAILED: "Failed",
}


@dataclass
class TrainingProgress:
    """Thread-safe training status object."""

    kind: str = ""                      # e.g. "OH" or "NPS"
    stage: str = STAGE_LOADING
    percent: Optional[float] = None     # None => indeterminate
    current_model: Optional[str] = None
    current_fold: Optional[int] = None
    total_folds: Optional[int] = None
    completed_models: int = 0
    total_models: Optional[int] = None
    message: str = ""
    device: str = "cpu"
    rows: Optional[int] = None
    history_days: Optional[int] = None
    model_name: Optional[str] = None
    error: Optional[str] = None

    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False
    )
    _started: float = field(
        default_factory=time.monotonic, repr=False
    )

    # -- mutators (called from the training thread) ---------------------------

    def set_stage(
        self,
        stage: str,
        message: Optional[str] = None,
        percent: Optional[float] = None,
    ) -> None:
        with self._lock:
            self.stage = stage
            if message is not None:
                self.message = message
            if percent is not None:
                self.percent = percent
            elif stage in (STAGE_COMPLETE, STAGE_FAILED):
                self.percent = 100.0 if stage == STAGE_COMPLETE else self.percent

    def set_kind(self, kind: str) -> None:
        with self._lock:
            self.kind = kind

    def set_models(self, total: int) -> None:
        with self._lock:
            self.total_models = int(total)
            self.completed_models = 0

    def start_candidate(
        self,
        name: str,
        total_models: Optional[int] = None,
    ) -> None:
        with self._lock:
            self.current_model = name
            if total_models is not None:
                self.total_models = int(total_models)
            self.current_fold = None

    def start_fold(
        self,
        fold: int,
        total_folds: int,
    ) -> None:
        with self._lock:
            self.current_fold = int(fold)
            self.total_folds = int(total_folds)
            self.percent = self._selection_percent()

    def complete_candidate(self) -> None:
        with self._lock:
            if self.total_models:
                self.completed_models = min(
                    self.completed_models + 1, self.total_models
                )
            self.current_model = None
            self.current_fold = None
            self.percent = self._selection_percent()

    def set_final_fit(
        self,
        model_name: str,
        device: str = "cpu",
        rows: Optional[int] = None,
    ) -> None:
        with self._lock:
            self.stage = STAGE_FINAL_REFIT
            self.model_name = model_name
            self.device = device or "cpu"
            if rows is not None:
                self.rows = int(rows)
            # Indeterminate: we never fake a precise percentage for a fit that
            # does not expose reliable internal iteration progress.
            self.percent = None
            self.current_model = None
            self.current_fold = None
            device_label = (device or "cpu").upper()
            self.message = (
                f"Final training: {model_name} — {device_label}"
                + (f" on {rows:,} rows..." if rows else "...")
            )

    def complete(
        self,
        model_name: Optional[str] = None,
        rows: Optional[int] = None,
        history_days: Optional[int] = None,
    ) -> None:
        with self._lock:
            self.stage = STAGE_COMPLETE
            self.percent = 100.0
            if model_name is not None:
                self.model_name = model_name
            if rows is not None:
                self.rows = int(rows)
            if history_days is not None:
                self.history_days = int(history_days)
            self.current_model = None
            self.current_fold = None
            self.error = None
            self.message = "Complete"

    def fail(self, message: str) -> None:
        with self._lock:
            self.stage = STAGE_FAILED
            self.error = message
            self.message = message
            self.current_model = None
            self.current_fold = None

    # -- readers --------------------------------------------------------------

    def elapsed_seconds(self) -> float:
        with self._lock:
            return time.monotonic() - self._started

    def _selection_percent(self) -> Optional[float]:
        """Real percentage from completed model / fold counts."""
        if not self.total_models:
            return None
        if self.total_folds:
            denom = max(1, self.total_models * self.total_folds)
            num = self.completed_models * self.total_folds
            if self.current_fold:
                num += max(0, self.current_fold - 1)
            return 100.0 * min(num, denom) / denom
        return 100.0 * self.completed_models / self.total_models

    def snapshot(self) -> Dict[str, Any]:
        """Immutable dict snapshot safe to read from the GUI thread."""
        with self._lock:
            return {
                "kind": self.kind,
                "stage": self.stage,
                "stage_label": STAGE_LABELS.get(self.stage, self.stage),
                "percent": self.percent,
                "current_model": self.current_model,
                "current_fold": self.current_fold,
                "total_folds": self.total_folds,
                "completed_models": self.completed_models,
                "total_models": self.total_models,
                "message": self.message,
                "device": self.device,
                "rows": self.rows,
                "history_days": self.history_days,
                "model_name": self.model_name,
                "error": self.error,
                "elapsed_seconds": time.monotonic() - self._started,
            }
