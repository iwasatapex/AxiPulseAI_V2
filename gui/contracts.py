"""Canonical GUI contracts for AxiPulseAI V2.

This module is the single place the GUI keeps the canonical V2.3 KPI
targets / hard bounds, the NPS / OH ranges, scenario and model-family
metadata, plus a small number of shared helpers (state validation and
NPS-distribution normalisation).

The numeric source of truth lives in the canonical engines:

* ``core.forecast_ai.prediction.service.PredictionService`` targets
  (quality=87, competency=93, attendance=90, release=60, transfer=9).
* ``core.forecast_ai.state.transition.KPITransition.apply`` hard bounds
  (quality 60-100, competency 55-100, attendance 65-100, release 50-100,
  transfer 0-20).

These numbers are mirrored here (not duplicated per-view) and must never
contradict the engine.  Nothing in this module performs any model /
simulator mathematics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------
# NPS / OH ranges
# ---------------------------------------------------------------------
NPS_MIN = -100.0
NPS_MAX = 100.0
OH_MIN = 0.0
OH_MAX = 100.0

# ---------------------------------------------------------------------
# Canonical V2.3 KPI hard bounds + targets (mirror the engines above).
# ---------------------------------------------------------------------
# Each entry: label, unit, target, hard min, hard max, default input.
KPI: Dict[str, Dict[str, Any]] = {
    "quality": {
        "label": "Quality",
        "unit": "%",
        "target": 87.0,
        "min": 60.0,
        "max": 100.0,
        "default": 87.0,
    },
    "competency": {
        "label": "Competency",
        "unit": "%",
        "target": 93.0,
        "min": 55.0,
        "max": 100.0,
        "default": 93.0,
    },
    "attendance": {
        "label": "Attendance",
        "unit": "%",
        "target": 90.0,
        "min": 65.0,
        "max": 100.0,
        "default": 90.0,
    },
    "release": {
        "label": "Release Rate",
        "unit": "%",
        "target": 60.0,
        "min": 50.0,
        "max": 100.0,
        "default": 60.0,
    },
    "transfer": {
        "label": "Transfer Rate",
        "unit": "%",
        "target": 9.0,
        "min": 0.0,
        "max": 20.0,
        "default": 9.0,
    },
    "operations_health": {
        "label": "Operational Health",
        "unit": "%",
        "target": None,
        "min": OH_MIN,
        "max": OH_MAX,
        "default": 95.0,
    },
    "nps": {
        "label": "NPS",
        "unit": "",
        "target": None,
        "min": NPS_MIN,
        "max": NPS_MAX,
        "default": 82.0,
    },
    "total_calls_received": {
        "label": "Total Calls Received",
        "unit": "",
        "target": None,
        "min": 1.0,
        "max": 100000.0,
        "default": 2000.0,
    },
}

# NPS is observed prior-period history, never a contemporaneous driver.
NPS_INPUT_LABEL = "Observed Prior-Period NPS"
NPS_INPUT_HELP = (
    "Observed prior-period NPS (historical state). This is NOT a "
    "contemporaneous KPI the engine generates from the current inputs — "
    "NPS is an output of the operational pipeline, not an input driver."
)

# ---------------------------------------------------------------------
# Model-family metadata
# ---------------------------------------------------------------------
MODEL_SUFFIX_OH = "_OH.pkl"
MODEL_SUFFIX_NPS = "_NPS.pkl"

# Direction each KPI moves to be "better" (used only for analytic
# met/not-met interpretation against the canonical target).
KPI_DIRECTION = {
    "quality": "high",
    "competency": "high",
    "attendance": "high",
    "release": "high",
    "transfer": "low",  # lower transfer is better
    "operations_health": "high",
    "nps": "high",
}

# Canonical V2.3 KPI-met thresholds (rule 2):
#   quality / competency / release are "met" at >= 95% of target.
#   transfer is "met" at <= 105% of target (lower is better).
# Attendance has no canonical KPI-met definition, so it uses the raw target.
KPI_MET_FACTOR = {
    "quality": 0.95,
    "competency": 0.95,
    "release": 0.95,
    "transfer": 1.05,
}


def kpi_target(key: str):
    """Return the canonical target for a KPI (or None if no target)."""
    cfg = KPI.get(key)
    return cfg.get("target") if cfg else None


def kpi_met(key: str, value) -> bool | None:
    """Return True/False if ``value`` meets the canonical KPI-met threshold.

    Per the V2.3 contract, quality/competency/release meet at >= 95% of
    target and transfer meets at <= 105% of target (inverse semantics:
    lower transfer is better). Returns ``None`` when the KPI has no
    canonical target or the value is not numeric.
    """
    cfg = KPI.get(key)
    if not cfg or cfg.get("target") is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    target = float(cfg["target"])
    threshold = target * KPI_MET_FACTOR.get(key, 1.0)
    if KPI_DIRECTION.get(key) == "low":
        return val <= threshold
    return val >= threshold

# ---------------------------------------------------------------------
# Scenario metadata
# ---------------------------------------------------------------------
BASELINE_SCENARIO_ID = "baseline"


def kpi_bounds(key: str) -> tuple[float, float]:
    """Return (min, max) hard bounds for a KPI key."""
    if key not in KPI:
        raise KeyError(f"Unknown KPI key: {key}")
    return float(KPI[key]["min"]), float(KPI[key]["max"])


def kpi_default(key: str) -> float:
    """Return the canonical default input value for a KPI key."""
    if key not in KPI:
        raise KeyError(f"Unknown KPI key: {key}")
    return float(KPI[key]["default"])


# ---------------------------------------------------------------------
# KPI state validation (service boundary — must not be bypassable)
# ---------------------------------------------------------------------

def validate_state(state: Dict[str, Any]) -> None:
    """Validate every KPI in ``state`` against its canonical hard bounds.

    Raises ``ValueError`` listing every offending field so a caller can
    never feed an out-of-range KPI (e.g. release < 50 or transfer > 20)
    into any engine through the GUI service layer.

    Only the canonical KPI keys present in ``state`` are checked; unknown
    keys are left untouched so the state can still carry engine-supported
    extras (e.g. ``history_buffer``).
    """
    errors: list[str] = []
    for key, cfg in KPI.items():
        if key not in state:
            continue
        raw = state[key]
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{cfg['label']} must be numeric, got {raw!r}")
            continue
        if not (cfg["min"] <= value <= cfg["max"]):
            errors.append(
                f"{cfg['label']} must be within "
                f"[{cfg['min']:g}, {cfg['max']:g}], got {value:g}"
            )
    if errors:
        raise ValueError("Invalid KPI state: " + "; ".join(errors))


# ---------------------------------------------------------------------
# NPS distribution normalisation
# ---------------------------------------------------------------------
def normalize_nps_distribution(distribution: Dict[str, Any]) -> Dict[int, float]:
    """Normalise an NPS 0..10 score distribution to ``{score: probability}``.

    Supported key forms produced by the canonical engines:
      * numeric keys:  ``5``
      * string ints:   ``"5"``
      * prefixed:      ``"score_5"``

    Raises ``ValueError`` (a clear UI/service error) for malformed data
    instead of an obscure ``ValueError`` from ``int(k)``.
    """
    if not isinstance(distribution, dict):
        raise ValueError(
            f"NPS distribution must be a mapping, got {type(distribution).__name__}"
        )
    out: Dict[int, float] = {}
    for key, prob in distribution.items():
        parsed = _parse_dist_key(key)
        if parsed is None:
            raise ValueError(
                f"Malformed NPS distribution key {key!r}: expected a score "
                f"0..10 (e.g. 5, '5', or 'score_5')."
            )
        try:
            out[parsed] = float(prob)
        except (TypeError, ValueError):
            raise ValueError(
                f"NPS distribution probability for score {parsed} is not "
                f"numeric: {prob!r}"
            )
    return out


def _parse_dist_key(key: Any):
    """Return an int score 0..10 from a supported distribution key, else None."""
    if isinstance(key, bool):  # bool is an int subclass — reject explicitly
        return None
    if isinstance(key, int):
        return key if 0 <= key <= 10 else None
    text = str(key).strip().lower()
    if not text:
        return None
    if text.startswith("score_"):
        text = text[len("score_"):]
    if text.isdigit():
        score = int(text)
        return score if 0 <= score <= 10 else None
    return None


# ---------------------------------------------------------------------
# Dataset-loading helper (canonical, used by preview + training)
# ---------------------------------------------------------------------

# Formats the canonical loader can actually read — computed from the
# libraries actually installed, so trainability metadata always agrees
# with what load_dataset() can process.  Anything outside this set is
# *never advertised* as a trainable/previewable dataset.
def _loader_dependency_available(module: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(module) is not None


SUPPORTED_DATASET_FORMATS = {".csv", ".tsv", ".json"}
if _loader_dependency_available("pyarrow") or _loader_dependency_available("fastparquet"):
    SUPPORTED_DATASET_FORMATS.add(".parquet")
if _loader_dependency_available("openpyxl"):
    SUPPORTED_DATASET_FORMATS.add(".xlsx")
if _loader_dependency_available("xlrd"):
    SUPPORTED_DATASET_FORMATS.add(".xls")
SUPPORTED_DATASET_FORMATS = frozenset(SUPPORTED_DATASET_FORMATS)


def load_dataset_sample(path, n_rows: int = 50_000) -> "Any":
    """Load only a bounded sample for GUI preview/analytics.

    Never use this for model training. Training must use load_dataset().
    """
    import pandas as pd

    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".csv":
        return pd.read_csv(p, nrows=n_rows)
    if ext == ".tsv":
        return pd.read_csv(p, sep="\\t", nrows=n_rows)
    if ext == ".xlsx":
        return pd.read_excel(p, nrows=n_rows)
    if ext == ".xls":
        return pd.read_excel(p, nrows=n_rows)

    # JSON / Parquet fallback: read normally only where a bounded reader
    # is not available through the existing dependency layer.
    # These should normally be converted to CSV/Parquet for large datasets.
    if ext == ".json":
        return pd.read_json(p)
    if ext == ".parquet":
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(p)
            batches = []
            remaining = n_rows
            for batch in pf.iter_batches(batch_size=min(remaining, 10_000)):
                batches.append(batch.to_pandas())
                remaining -= len(batch)
                if remaining <= 0:
                    break
            return pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()
        except Exception:
            return pd.read_parquet(p).head(n_rows)

    return load_dataset(p).head(n_rows)


def load_dataset(path) -> "Any":
    """Load a training/preview dataset into a pandas DataFrame.

    Supports CSV, TSV, JSON, Parquet and Excel.  Raises ``ValueError`` with
    an explicit message for any unsupported format so callers can tell the
    user exactly why a file cannot be used (including the missing optional
    dependency for a format that is not installed here).
    """
    import pandas as pd

    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(p)
    if ext == ".tsv":
        return pd.read_csv(p, sep="\t")
    if ext == ".json":
        return pd.read_json(p)
    if ext == ".parquet":
        if not (_loader_dependency_available("pyarrow") or _loader_dependency_available("fastparquet")):
            raise ValueError(
                f"Parquet dataset {p.name!r} requires the 'pyarrow' or "
                f"'fastparquet' package, which is not installed."
            )
        return pd.read_parquet(p)
    if ext == ".xlsx":
        if not _loader_dependency_available("openpyxl"):
            raise ValueError(
                f"Excel dataset {p.name!r} requires the 'openpyxl' package, "
                f"which is not installed."
            )
        return pd.read_excel(p)
    if ext == ".xls":
        if not _loader_dependency_available("xlrd"):
            raise ValueError(
                f"Excel .xls dataset {p.name!r} requires the 'xlrd' package, "
                f"which is not installed."
            )
        return pd.read_excel(p)
    raise ValueError(
        f"Unsupported dataset format {ext!r} for {p.name}. Supported "
        f"training/preview formats: "
        + ", ".join(sorted(SUPPORTED_DATASET_FORMATS))
    )

