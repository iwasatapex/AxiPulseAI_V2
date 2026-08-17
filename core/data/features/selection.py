from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .intelligence import FeatureProfile, profile_features


@dataclass(frozen=True)
class FeatureSelection:
    selected: tuple[str, ...]
    excluded: tuple[str, ...]
    reasons: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)


def select_features(
    data: pd.DataFrame,
    *,
    target: str | None = None,
    max_missing_fraction: float = 0.50,
    exclude_identifiers: bool = True,
    exclude_free_text: bool = True,
    exclude_datetime: bool = True,
) -> FeatureSelection:
    """
    Select model-eligible features conservatively.

    This layer does not mutate the input dataframe and does not alter
    the existing schema classifier.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    if not 0.0 <= max_missing_fraction <= 1.0:
        raise ValueError(
            "max_missing_fraction must be between 0 and 1"
        )

    if target is not None and target not in data.columns:
        raise ValueError(f"target column not found: {target}")

    profiles = profile_features(data)

    selected: list[str] = []
    excluded: list[str] = []
    reasons: dict[str, str] = {}

    for profile in profiles:
        name = profile.name

        if target is not None and name == target:
            excluded.append(name)
            reasons[name] = "target column"
            continue

        if profile.missing_fraction > max_missing_fraction:
            excluded.append(name)
            reasons[name] = "exceeds missing-data threshold"
            continue

        if exclude_identifiers and profile.likely_identifier:
            excluded.append(name)
            reasons[name] = "identifier excluded from features"
            continue

        if exclude_free_text and profile.likely_free_text:
            excluded.append(name)
            reasons[name] = "free-text excluded from features"
            continue

        if exclude_datetime and profile.kind == "datetime":
            excluded.append(name)
            reasons[name] = "datetime excluded from features"
            continue

        selected.append(name)
        reasons[name] = "selected"

    return FeatureSelection(
        selected=tuple(selected),
        excluded=tuple(excluded),
        reasons=dict(reasons),
        metadata={
            "rows": int(len(data)),
            "columns": int(len(data.columns)),
            "selected_count": len(selected),
            "excluded_count": len(excluded),
            "target": target,
        },
    )
