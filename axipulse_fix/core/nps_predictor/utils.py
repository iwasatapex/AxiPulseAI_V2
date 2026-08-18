"""
AxiPulseAI – Utilities
"""
import pandas as pd
import numpy as np

def safe_divide(a, b, default=0.0):
    if b == 0 or np.isnan(b) or np.isinf(b):
        return default
    return a / b

def round_to_int(x):
    return int(round(x))

def ensure_datetime(df, col="date"):
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df
