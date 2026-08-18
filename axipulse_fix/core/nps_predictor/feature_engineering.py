"""
AxiPulseAI – Feature Engineering
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
from .config import DEFAULT_CONFIG

# External causal factors whose cutoff-time provenance must be proven before
# they are treated as "known at T". These are consumed (if present) only when a
# companion ``{factor}_known_at`` timestamp proves they were available at or
# before the prediction cutoff. Without that proof they are rejected rather
# than silently treated as known-at-T.
EXTERNAL_FACTOR_COLUMNS = (
    "seasonal_factor",
    "weekday_factor",
    "flu_factor",
    "enrollment_factor",
    "holiday_factor",
    "random_factor",
)

def prepare_features(df: pd.DataFrame, config=None, *, copy: bool = True) -> pd.DataFrame:
    """
    Create engineered features from raw data.
    Returns a DataFrame with 'date' and all engineered features.
    If target columns (score_0..score_10) exist, they are kept as targets.

    ``copy`` defaults to True so callers that reuse their frame (e.g. the
    prediction alignment path) keep it untouched. The NPS trainer passes
    ``copy=False``: it exclusively owns the freshly loaded 1M-row frame and
    never uses it again, so copying it would needlessly double peak RAM.
    """
    config = config or DEFAULT_CONFIG
    if copy:
        df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # ==========================================================
    # AxiPulseAI canonical schema adapter
    # ==========================================================

    aliases = {
        "quality": "actual_quality",
        "competency": "actual_competency",
        "attendance": "actual_attendance",
        "transfer": "actual_transfer_rate",
    }

    for old_name, new_name in aliases.items():
        if old_name not in df.columns and new_name in df.columns:
            df[old_name] = df[new_name]

    trajectory_col = next((c for c in ("trajectory_id", "simulation_id", "run_id", "scenario_id", "agent_id") if c in df.columns), None)
    if trajectory_col is not None:
        df["__trajectory_key"] = df[trajectory_col].astype(str)
    else:
        df["__trajectory_key"] = df.groupby("date", sort=False).cumcount()

    # Release features
    df["release_gap"] = df["target_release_rate"] - df["actual_release_rate"]
    df["release_delta"] = df.groupby("__trajectory_key", sort=False)["actual_release_rate"].diff().fillna(0)

    # KPI performance gaps
    if "target_quality" in df.columns and "quality" in df.columns:
        df["quality_gap"] = df["target_quality"] - df["quality"]

    if "target_competency" in df.columns and "competency" in df.columns:
        df["competency_gap"] = df["target_competency"] - df["competency"]

    if "target_attendance" in df.columns and "attendance" in df.columns:
        df["attendance_gap"] = df["target_attendance"] - df["attendance"]

    # Canonical transfer naming:
    # target_transfer = default KPI target
    # transfer = achieved actual value
    if "target_transfer" not in df.columns and "target_transfer_rate" in df.columns:
        df["target_transfer"] = df["target_transfer_rate"]

    if "target_transfer" in df.columns and "transfer" in df.columns:
        df["transfer_gap"] = df["transfer"] - df["target_transfer"]

    # ---- Survey reliability features ----
    if "total_surveys" in df.columns:
        df["survey_confidence"] = (
            df["total_surveys"] / (df["total_surveys"] + 10)
        ).clip(0, 1)
    else:
        df["total_surveys"] = 0
        df["survey_confidence"] = 0.0

    if "survey_rate" not in df.columns:
        df["survey_rate"] = 0.0


    # ---- NPS percentages (only if aggregates exist) ----
    if all(col in df.columns for col in ["promoters", "passives", "detractors"]):
        total = df["total_calls_received"].replace(0, np.nan)
        df["promoter_pct"] = df["promoters"] / total * 100
        df["passive_pct"] = df["passives"] / total * 100
        df["detractor_pct"] = 100 - df["promoter_pct"] - df["passive_pct"]
        df["promoter_pct"] = df["promoter_pct"].fillna(0)
        df["passive_pct"] = df["passive_pct"].fillna(0)
        df["detractor_pct"] = df["detractor_pct"].fillna(0)
        df["nps_today"] = df["promoter_pct"] - df["detractor_pct"]
    else:
        # For prediction only: create dummy columns that will be dropped later
        df["promoter_pct"] = 0.0
        df["passive_pct"] = 0.0
        df["detractor_pct"] = 0.0
        df["nps_today"] = 0.0

    # Previous/rolling state must stay inside a single simulation trajectory.
    grouped = df.groupby("__trajectory_key", sort=False)
    for col in ["operational_health", "quality", "competency", "transfer", "attendance"]:
        if col in df.columns:
            df[f"{col}_previous_day"] = grouped[col].shift(1).fillna(df[col])

    roll_days = config.roll_days
    for col in ["operational_health", "actual_release_rate", "business_intelligence_factor", "member_intelligence_factor"]:
        for w in roll_days:
            df[f"{col}_roll{w}"] = grouped[col].transform(lambda x, w=w: x.rolling(w, min_periods=1).mean().shift(1))

    if "promoter_pct" in df.columns:
        for col in ["promoter_pct", "passive_pct", "detractor_pct"]:
            for w in roll_days:
                df[f"{col}_roll{w}"] = grouped[col].transform(lambda x, w=w: x.rolling(w, min_periods=1).mean().shift(1))
        for w in roll_days:
            df[f"nps_roll{w}"] = grouped["nps_today"].transform(lambda x, w=w: x.rolling(w, min_periods=1).mean().shift(1))

    # Calendar business context
    df["is_first_week_of_month"] = (df["date"].dt.day <= 7).astype(int)
    df["is_last_week_of_month"] = (df["date"].dt.day >= 24).astype(int)
    df["days_since_month_start"] = df["date"].dt.day
    df["days_until_month_end"] = (
        df["date"].dt.days_in_month - df["date"].dt.day
    )
    df["month_progress"] = (
        df["date"].dt.day / df["date"].dt.days_in_month
    )

    # Cyclical dates
    if config.use_cyclical_dates:
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofweek / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofweek / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
        df["quarter"] = df["date"].dt.quarter
        df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)
        df["week_of_year"] = df["date"].dt.isocalendar().week
        df["day_of_month"] = df["date"].dt.day
        df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
        df["is_month_start"] = df["date"].dt.is_month_start.astype(int)

    # Normalize KPI column names
    if "quality_kpi" in df.columns and "quality" not in df.columns:
        df["quality"] = df["quality_kpi"]

    # Base features (exclude target columns)
    base = [
        "operational_health", "business_intelligence_factor", "member_intelligence_factor",


        # KPI target / actual inputs
        "target_quality",
        "quality",
        "quality_gap",

        "target_competency",
        "competency",
        "competency_gap",

        "target_attendance",
        "attendance",
        "attendance_gap",

        "target_transfer",
        "transfer",
        "transfer_gap",

        # Customer observation reliability
        # total_surveys: T-known operational estimate (call_volume * release_rate * survey_rate).
        # In training it equals the sum of T-day score_0..score_10 (a T-day outcome).
        # It is retained because the forecasting contract permits T-known information
        # to predict T+1, and the serving path estimates it from T-known call volume
        # rather than fabricating it from any target distribution.
        "total_surveys",
        "survey_rate",
        "survey_confidence",

        # Previous operational state

        # Calendar business context
        "is_first_week_of_month",
        "is_last_week_of_month",
        "days_since_month_start",
        "days_until_month_end",
        "month_progress",
    ]

    for col in ["operational_health", "actual_release_rate", "business_intelligence_factor", "member_intelligence_factor"]:
        for w in roll_days:
            base.append(f"{col}_roll{w}")

    if config.use_cyclical_dates:
        base.extend(["day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos", "quarter", "is_weekend", "week_of_year", "day_of_month", "is_month_end", "is_month_start"])
    else:
        base.extend(["day_of_week", "month"])

    # Additional causal NPS drivers — provenance-guarded.
    #
    # Deterministic simulator features that are explicitly defined by the
    # simulation state at date T may be admitted without a companion timestamp.
    # All other external factors still require an explicit {factor}_known_at
    # column proving availability at or before cutoff T.
    #
    # enrollment_factor is a simulator-state feature generated from the
    # enrollment state for the observation date itself, so it is explicitly
    # declared cutoff-known here. This is a narrow exception, not a general
    # provenance bypass.
    CUTOFF_KNOWN_SIMULATOR_FEATURES = {"enrollment_factor"}

    for col in EXTERNAL_FACTOR_COLUMNS:
        if col not in df.columns:
            continue

        if col in CUTOFF_KNOWN_SIMULATOR_FEATURES:
            base.append(col)
            continue

        known_at_col = f"{col}_known_at"

        if known_at_col not in df.columns:
            raise ValueError(
                f"External feature {col!r} has no proven cutoff-time provenance "
                f"(missing '{known_at_col}' column); refusing to treat it as "
                f"known-at-T."
            )

        known_at = pd.to_datetime(
            df[known_at_col],
            errors="coerce",
        )
        cutoff = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

        if known_at.isna().any() or (known_at > cutoff).any():
            raise ValueError(
                f"External feature {col!r} is not known at or before prediction "
                f"cutoff T; refusing temporally-future/unknown feature."
            )

        base.append(col)

    # Remove rolling history features for KPI-driven NPS model
    base = [c for c in base if "_roll" not in c]

    feature_cols = list(dict.fromkeys(base))

    # ---- Ensure all feature columns exist (prevents KeyError at serving time) ----
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    # ---- Keep target columns: aggregate percentages and score columns ----
    target_cols = [c for c in ["promoter_pct", "passive_pct", "detractor_pct", "nps_today"] if c in df.columns]
    # Add score columns if they exist
    score_cols = [f"score_{i}" for i in range(11) if f"score_{i}" in df.columns]
    return df[["date"] + feature_cols + target_cols + score_cols]

# Columns in an inference row / history buffer that must NOT be coerced to
# numeric. ``date`` and trajectory/simulation identity are deliberately kept as
# their native type; score columns and pre-computed aggregates are prediction
# targets, not model inputs.
_NON_NUMERIC_INPUT_COLS = {
    "date",
    "trajectory_id",
    "simulation_id",
    "run_id",
    "scenario_id",
    "agent_id",
    "promoter_pct",
    "passive_pct",
    "detractor_pct",
    "nps_today",
    *(f"score_{i}" for i in range(11)),
}

# Raw numeric inputs consumed by :func:`prepare_features` to build the engineered
# features (gaps, rolling windows, survey reliability). These are the ONLY raw
# columns coerced to numeric at inference, together with the model feature
# columns themselves. Genuinely categorical raw columns that are not model
# features (e.g. ``scenario_regime``, ``event_type``) are deliberately left
# untouched -- never blindly cast.
RAW_NUMERIC_INPUT_COLUMNS = {
    "operational_health",
    "business_intelligence_factor",
    "member_intelligence_factor",
    "operational_intelligence_factor",
    "target_release_rate",
    "actual_release_rate",
    "release_gap",
    "target_quality",
    "quality",
    "actual_quality",
    "quality_gap",
    "target_competency",
    "competency",
    "actual_competency",
    "competency_gap",
    "target_attendance",
    "attendance",
    "actual_attendance",
    "attendance_gap",
    "target_transfer",
    "transfer",
    "transfer_rate",
    "target_transfer_rate",
    "actual_transfer_rate",
    "transfer_gap",
    "quality_kpi",
    "total_surveys",
    "survey_rate",
    "total_calls_received",
    "total_release_calls",
    "promoters",
    "passives",
    "detractors",
    "enrollment_factor",
}


def _coercion_targets(feature_names):
    """Return the set of columns to coerce to numeric at inference.

    Only actual model features plus the raw numeric inputs used to build them
    are coerced. Categorical non-feature columns are never cast.
    """
    return set(feature_names) | RAW_NUMERIC_INPUT_COLUMNS


def _coerce_numeric_columns(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Coerce the given columns to numeric; reject non-convertible values.

    The training pipeline casts the entire feature matrix to ``float32``, so
    every model feature is numeric. A row arriving with ``operational_health``
    as a string / ``object`` / pandas ``str`` dtype (e.g. ``"75.5"`` or a value
    mixed with missing entries) must be coerced to numeric, exactly as in
    training, or XGBoost / CatBoost / ExtraTrees reject the frame with
    "DataFrame.dtypes ... must be int, float, bool or category".

    A genuinely non-convertible value (e.g. ``"Good"``) is rejected explicitly
    rather than silently stringified or coerced to a sentinel. Columns not in
    ``columns`` (e.g. categorical metadata) are left untouched.
    """
    for col in columns:
        if col not in df.columns or col in _NON_NUMERIC_INPUT_COLS:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            continue
        coerced = pd.to_numeric(series, errors="coerce")
        bad = series.notna() & coerced.isna()
        if bad.any():
            bad_values = [repr(v) for v in series[bad].unique()[:5]]
            raise ValueError(
                f"Inference feature {col!r} contains non-numeric value(s) "
                f"{bad_values} that cannot be converted to the required "
                f"numeric type; refusing to continue."
            )
        df[col] = coerced
    return df


def _validate_inference_dtypes(X: pd.DataFrame) -> pd.DataFrame:
    """Reject an inference feature matrix whose dtypes the trained model cannot
    accept, or that contains NaN/inf after preprocessing.

    The trained NPS models (XGBoost / CatBoost / ExtraTrees / ...) require every
    feature column to be numeric. This check runs immediately before the model
    is called so a malformed row surfaces a clear error instead of a cryptic
    library failure or a silent fallback.
    """
    bad = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if bad:
        raise ValueError(
            "Inference feature matrix contains non-numeric column(s) that the "
            "trained NPS model cannot accept: "
            + ", ".join(str(c) for c in bad)
            + ". Numeric coercion failed; refusing to predict."
        )
    try:
        values = X.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Inference feature matrix could not be converted to numeric "
            "values for validation; refusing to predict."
        ) from exc
    if not np.isfinite(values).all():
        raise ValueError(
            "Inference feature matrix contains NaN or infinite values after "
            "preprocessing; refusing to predict."
        )
    return X


def align_features(row_data: dict, feature_names: list, feature_stats: dict, history_buffer: pd.DataFrame = None) -> pd.DataFrame:
    """
    Align a single prediction row with training features.
    history_buffer must contain raw columns (not engineered).

    Raw numeric inputs are coerced to numeric (rejecting non-convertible
    values), features are reindexed to exactly ``feature_names`` in training
    order, missing values are imputed with the training medians, and the final
    matrix is validated to be all-numeric and finite before it is returned to
    the model.
    """
    config = DEFAULT_CONFIG

    if history_buffer is not None and len(history_buffer) > 0:
        df = pd.concat([history_buffer.copy(), pd.DataFrame([row_data])], ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        _coerce_numeric_columns(df, _coercion_targets(feature_names))
        X_full = prepare_features(df, config)
        X_full = X_full.iloc[[-1]]
    else:
        df = pd.DataFrame([row_data])
        _coerce_numeric_columns(df, _coercion_targets(feature_names))
        X_full = prepare_features(df, config)

    # Drop date and target columns (they are not features)
    drop_cols = ["date", "promoter_pct", "passive_pct", "detractor_pct", "nps_today"]
    # Also drop any score columns if they exist (prediction won't have them)
    drop_cols.extend([f"score_{i}" for i in range(11)])
    X = X_full.drop(columns=[c for c in drop_cols if c in X_full.columns], errors="ignore")

    if feature_names:
        # Feature ordering exactly matches the training schema.
        X = X.reindex(columns=feature_names, fill_value=0)

    # Force every model feature to numeric using the training statistics.
    _coerce_numeric_columns(X, feature_names)

    if feature_stats:
        for col in X.columns:
            if X[col].isnull().any():
                med = feature_stats.get(f"{col}_median", 0)
                X[col] = X[col].fillna(med)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Remove historical rolling features for KPI-driven NPS model
    X = X[[c for c in X.columns if "_roll" not in c]]

    # Final guarantee: numeric, finite, exactly the training columns.
    return _validate_inference_dtypes(X)
