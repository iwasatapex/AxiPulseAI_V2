"""
Feature engineering: derived scores, gaps, cyclical dates, lag features,
and feature alignment for prediction.

Moved verbatim out of OperationalHealthPredictor in operation_health_predictor.py
(Phase 2, Step 3 — no logic changed).
"""

from typing import Optional

import numpy as np
import pandas as pd

from .constants import ISSUE_PREFIX, RELEASE_TARGET, TRANSFER_TARGET
from .utils import logger


class FeatureEngineeringMixin:
    # ---------------------------------------------------------------
    # Feature Engineering
    # ---------------------------------------------------------------
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        trajectory_col = next((c for c in ("trajectory_id", "simulation_id", "run_id", "scenario_id", "agent_id") if c in df.columns), None)
        df["__trajectory_key"] = df[trajectory_col].astype(str) if trajectory_col is not None else df.groupby("date", sort=False).cumcount()


        df["release_score"] = np.where(
            df["actual_release_rate"] >= RELEASE_TARGET,
            100,
            (df["actual_release_rate"] / RELEASE_TARGET) * 100,
        )
        transfer = df["actual_transfer_rate"].replace(0, np.nan)
        df["transfer_score"] = np.where(
            transfer <= TRANSFER_TARGET,
            100,
            np.minimum(100, 100 * (TRANSFER_TARGET / transfer)),
        )
        df["transfer_score"] = df["transfer_score"].replace([np.inf, -np.inf], 100).fillna(100)
        df["total_release_calls"] = df["total_calls_received"] * df["actual_release_rate"] / 100

        gap_pairs = [
            ("quality", "target_quality", "actual_quality"),
            ("competency", "target_competency", "actual_competency"),
            ("attendance", "target_attendance", "actual_attendance"),
            ("release", "target_release_rate", "actual_release_rate"),
            ("transfer", "target_transfer_rate", "actual_transfer_rate"),
        ]
        for name, target, actual in gap_pairs:
            df[f"{name}_gap"] = df[target] - df[actual]

        if self.config.use_cyclical_dates:
            df["day_of_week_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofweek / 7)
            df["day_of_week_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofweek / 7)
            df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
            df["quarter"] = df["date"].dt.quarter
            df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)
            df.drop(columns=["day_of_week", "month"], errors="ignore", inplace=True)

        issue_cols = [c for c in df.columns if c.startswith(ISSUE_PREFIX)]
        base_features = [
            "actual_quality", "quality_gap",
            "actual_competency", "competency_gap",
            "actual_attendance", "attendance_gap",
            "actual_release_rate", "release_gap",
            "actual_transfer_rate", "transfer_gap",
            "total_calls_received", "total_release_calls",
            "operational_intelligence_factor",
        ]

        if self.config.use_cyclical_dates:
            base_features.extend(["day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos", "quarter", "is_weekend"])
        else:
            base_features.extend(["day_of_week", "month"])

        if self.config.use_lag_features and len(df) > 7:
            for col in ["actual_quality", "actual_competency", "actual_attendance",
                        "actual_release_rate", "actual_transfer_rate", "total_calls_received"]:
                grouped = df.groupby("__trajectory_key", sort=False)
                df[f"{col}_lag1"] = grouped[col].shift(1)
                df[f"{col}_roll3"] = grouped[col].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
                df[f"{col}_roll7"] = grouped[col].transform(lambda x: x.rolling(7, min_periods=1).mean().shift(1))
            df.fillna(0, inplace=True)
            base_features.extend([c for c in df.columns if "_lag1" in c or "_roll3" in c or "_roll7" in c])

        feature_list = base_features + issue_cols
        feature_list = list(dict.fromkeys(feature_list))

        for col in feature_list:
            if col not in df.columns:
                df[col] = 0.0

        return df[feature_list]

    def _align_features(self, row_data: dict, history_buffer: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        df = pd.DataFrame([row_data])
        if history_buffer is not None and self.config.use_lag_features:
            if len(history_buffer) < 7:
                logger.warning("History buffer too small for lag features; lag features disabled.")
                original_lag = self.config.use_lag_features
                self.config.use_lag_features = False
                X = self.prepare_features(df)
                self.config.use_lag_features = original_lag
            else:
                combined = pd.concat([history_buffer, df], ignore_index=True)
                combined["date"] = pd.to_datetime(combined["date"])
                combined = combined.sort_values("date").reset_index(drop=True)
                X = self.prepare_features(combined)
                X = X.iloc[[-1]]
        else:
            X = self.prepare_features(df)

        if self.feature_names:
            X = X.reindex(columns=self.feature_names, fill_value=0)
            if self._feature_stats:
                for col in X.columns:
                    if X[col].isnull().any():
                        med = self._feature_stats.get(f"{col}_median", 0)
                        X[col] = X[col].fillna(med)
        X = X.replace([np.inf, -np.inf], np.nan)
        if self._feature_stats:
            for col in X.columns:
                if X[col].isnull().any():
                    med = self._feature_stats.get(f"{col}_median", 0)
                    X[col] = X[col].fillna(med)
        else:
            X = X.fillna(0)
        return X


def prepare_features(predictor, df):
    """Compatibility API for the canonical FeatureEngineeringMixin."""
    if predictor is None:
        raise TypeError(
            "prepare_features() requires an OperationalHealthPredictor "
            "instance as the first argument."
        )
    return FeatureEngineeringMixin.prepare_features(predictor, df)
