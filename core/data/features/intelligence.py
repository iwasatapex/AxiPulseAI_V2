from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.data.schema import ColumnSchema, discover_schema


@dataclass(frozen=True)
class FeatureAssessment:
    name: str
    kind: str
    eligible: bool
    reason: str
    missing_fraction: float
    unique_fraction: float
    is_target: bool = False
    is_identifier: bool = False
    is_datetime: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureProfile:
    name: str
    kind: str
    dtype: str
    rows: int
    non_null: int
    unique_values: int
    unique_fraction: float
    missing_fraction: float
    cardinality: str
    text_like: bool
    likely_identifier: bool
    likely_free_text: bool


def profile_feature(
    series: pd.Series,
) -> FeatureProfile:
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")

    rows = int(len(series))
    non_null = int(series.notna().sum())
    unique_values = int(series.dropna().nunique())

    if non_null == 0:
        unique_fraction = 0.0
    else:
        unique_fraction = float(unique_values / non_null)

    missing_fraction = (
        0.0
        if rows == 0
        else float(series.isna().mean())
    )

    name = str(series.name).strip().lower()

    # This is intentionally independent from the existing schema
    # classifier. It provides a richer interpretation without
    # rewriting the canonical ColumnKind contract.
    identifier_name = (
        name in {"id", "identifier", "uuid", "guid"}
        or name.endswith("_id")
        or name.endswith("-id")
        or name.endswith(" id")
    )

    text_like = (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    )

    average_length = 0.0

    if text_like and non_null:
        average_length = float(
            series.dropna()
            .astype(str)
            .str.len()
            .mean()
        )

    likely_free_text = (
        text_like
        and average_length >= 40
        and unique_fraction >= 0.50
        and not identifier_name
    )

    likely_identifier = (
        identifier_name
        or (
            text_like
            and unique_fraction >= 0.95
            and not likely_free_text
            and average_length < 40
        )
    )

    if unique_fraction >= 0.95:
        cardinality = "high"
    elif unique_fraction >= 0.50:
        cardinality = "medium"
    else:
        cardinality = "low"

    return FeatureProfile(
        name=str(series.name),
        kind="unknown",
        dtype=str(series.dtype),
        rows=rows,
        non_null=non_null,
        unique_values=unique_values,
        unique_fraction=unique_fraction,
        missing_fraction=missing_fraction,
        cardinality=cardinality,
        text_like=text_like,
        likely_identifier=likely_identifier,
        likely_free_text=likely_free_text,
    )


def profile_features(
    data: pd.DataFrame,
) -> list[FeatureProfile]:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    schema = discover_schema(data)
    kinds = {item.name: item.kind for item in schema}

    profiles = []

    for column in data.columns:
        profile = profile_feature(data[column])

        profiles.append(
            FeatureProfile(
                name=profile.name,
                kind=kinds.get(profile.name, profile.kind),
                dtype=profile.dtype,
                rows=profile.rows,
                non_null=profile.non_null,
                unique_values=profile.unique_values,
                unique_fraction=profile.unique_fraction,
                missing_fraction=profile.missing_fraction,
                cardinality=profile.cardinality,
                text_like=profile.text_like,
                likely_identifier=profile.likely_identifier,
                likely_free_text=profile.likely_free_text,
            )
        )

    return profiles


@dataclass(frozen=True)
class LeakageFinding:
    feature: str
    leakage_type: str
    severity: str
    reason: str


@dataclass(frozen=True)
class FeatureIntelligenceResult:
    features: tuple[FeatureAssessment, ...]
    target: str | None
    leakage: tuple[LeakageFinding, ...]
    duplicate_rows: int
    missing_cells: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _missing_fraction(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.isna().mean())


def _unique_fraction(series: pd.Series) -> float:
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
    return float(non_null.nunique() / len(non_null))


def _target_candidates(
    data: pd.DataFrame,
    schema: list[ColumnSchema],
) -> list[str]:
    candidates: list[str] = []

    for item in schema:
        name = item.name.strip().lower()

        if name in {
            "target",
            "label",
            "outcome",
            "y",
            "prediction_target",
        }:
            candidates.append(item.name)

    return candidates


def identify_target(
    data: pd.DataFrame,
    target: str | None = None,
) -> str | None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    if target is not None:
        if target not in data.columns:
            raise ValueError(f"target column not found: {target}")
        return target

    schema = discover_schema(data)
    candidates = _target_candidates(data, schema)

    if len(candidates) == 1:
        return candidates[0]

    # Conservative by design:
    # ambiguous datasets do not receive an invented target.
    return None


def assess_features(
    data: pd.DataFrame,
    *,
    target: str | None = None,
    max_missing_fraction: float = 0.50,
    exclude_identifiers: bool = True,
    exclude_datetime: bool = False,
) -> list[FeatureAssessment]:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    if not 0.0 <= max_missing_fraction <= 1.0:
        raise ValueError("max_missing_fraction must be between 0 and 1")

    if target is not None and target not in data.columns:
        raise ValueError(f"target column not found: {target}")

    schema = discover_schema(data)
    assessments: list[FeatureAssessment] = []

    for item in schema:
        series = data[item.name]
        missing = _missing_fraction(series)
        unique = _unique_fraction(series)

        is_target = item.name == target
        is_identifier = item.kind == "identifier"
        is_datetime = item.kind == "datetime"

        eligible = True
        reason = "eligible"

        if is_target:
            eligible = False
            reason = "target column"

        elif missing > max_missing_fraction:
            eligible = False
            reason = "exceeds missing-data threshold"

        elif exclude_identifiers and is_identifier:
            eligible = False
            reason = "identifier excluded from features"

        elif exclude_datetime and is_datetime:
            eligible = False
            reason = "datetime excluded by configuration"

        assessments.append(
            FeatureAssessment(
                name=item.name,
                kind=item.kind,
                eligible=eligible,
                reason=reason,
                missing_fraction=missing,
                unique_fraction=unique,
                is_target=is_target,
                is_identifier=is_identifier,
                is_datetime=is_datetime,
                metadata={
                    "dtype": item.dtype,
                    "nullable": item.nullable,
                },
            )
        )

    return assessments


def detect_leakage(
    data: pd.DataFrame,
    *,
    target: str | None = None,
    datetime_column: str | None = None,
) -> list[LeakageFinding]:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    if target is not None and target not in data.columns:
        raise ValueError(f"target column not found: {target}")

    findings: list[LeakageFinding] = []

    if target is None:
        return findings

    target_series = data[target]

    for column in data.columns:
        if column == target:
            continue

        series = data[column]

        # Exact value equality is a strong leakage signal for a
        # non-identifier feature and is reported conservatively.
        try:
            if len(series) == len(target_series) and series.equals(target_series):
                findings.append(
                    LeakageFinding(
                        feature=column,
                        leakage_type="target_copy",
                        severity="high",
                        reason="feature values exactly match target values",
                    )
                )
        except Exception:
            pass

        name = str(column).strip().lower()
        target_name = str(target).strip().lower()

        if target_name and (
            target_name in name
            and name != target_name
        ):
            findings.append(
                LeakageFinding(
                    feature=column,
                    leakage_type="target_name",
                    severity="medium",
                    reason="feature name contains target name",
                )
            )

    if datetime_column is not None:
        if datetime_column not in data.columns:
            raise ValueError(
                f"datetime column not found: {datetime_column}"
            )

        parsed = pd.to_datetime(
            data[datetime_column],
            errors="coerce",
        )

        if parsed.notna().any():
            target_rows = target_series.notna()

            if target_rows.any():
                latest_target_time = parsed[target_rows].max()

                for column in data.columns:
                    if column in {target, datetime_column}:
                        continue

                    name = str(column).lower()

                    if any(
                        token in name
                        for token in (
                            "future",
                            "next",
                            "post",
                            "after",
                            "outcome",
                        )
                    ):
                        findings.append(
                            LeakageFinding(
                                feature=column,
                                leakage_type="temporal_name_signal",
                                severity="medium",
                                reason=(
                                    "feature name suggests future or "
                                    "post-outcome information"
                                ),
                            )
                        )

                _ = latest_target_time

    return findings


def analyze_features(
    data: pd.DataFrame,
    *,
    target: str | None = None,
    datetime_column: str | None = None,
    max_missing_fraction: float = 0.50,
    exclude_identifiers: bool = True,
    exclude_datetime: bool = False,
) -> FeatureIntelligenceResult:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    resolved_target = identify_target(
        data,
        target=target,
    )

    features = assess_features(
        data,
        target=resolved_target,
        max_missing_fraction=max_missing_fraction,
        exclude_identifiers=exclude_identifiers,
        exclude_datetime=exclude_datetime,
    )

    leakage = detect_leakage(
        data,
        target=resolved_target,
        datetime_column=datetime_column,
    )

    return FeatureIntelligenceResult(
        features=tuple(features),
        target=resolved_target,
        leakage=tuple(leakage),
        duplicate_rows=int(data.duplicated().sum()),
        missing_cells=int(data.isna().sum().sum()),
        metadata={
            "rows": int(len(data)),
            "columns": int(len(data.columns)),
        },
    )
