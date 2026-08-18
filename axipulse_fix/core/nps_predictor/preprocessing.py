import numpy as np
import pandas as pd

def compute_feature_stats(X):
    stats = {}
    for col in X.columns:
        stats[f"{col}_median"] = X[col].median()
        stats[f"{col}_std"] = X[col].std()
        stats[f"{col}_q1"] = X[col].quantile(0.25)
        stats[f"{col}_q3"] = X[col].quantile(0.75)
    return stats

def impute_missing(X, stats, *, copy: bool = True):
    """Impute missing values with per-column medians from ``stats``.

    ``copy=True`` (default) preserves the original caller's frame.  The NPS
    trainer owns its freshly built feature matrix and passes ``copy=False`` so
    the 1M-row training frame is not duplicated on the heap during refit.
    """
    if copy:
        X = X.copy()
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(stats.get(f"{col}_median", 0))
    return X

def clip_outliers_iqr(X, stats, *, copy: bool = True):
    """Clip outliers per column to [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

    ``copy=True`` (default) preserves the original caller's frame.  The NPS
    trainer owns its freshly built feature matrix and passes ``copy=False`` so
    the 1M-row training frame is not duplicated on the heap during refit.
    """
    if copy:
        X = X.copy()
    for col in X.columns:
        q1 = stats.get(f"{col}_q1", X[col].quantile(0.25))
        q3 = stats.get(f"{col}_q3", X[col].quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        X[col] = X[col].clip(lower, upper)
    return X
