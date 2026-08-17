"""
AxiPulseAI – Validation and Drift Detection
"""
import numpy as np
import pandas as pd
from typing import Dict

def detect_drift(training_stats: Dict, new_data: pd.DataFrame, threshold: float = 0.2) -> Dict[str, float]:
    drift = {}
    for col in new_data.columns:
        key = f"{col}_median"
        if key in training_stats:
            med_train = training_stats[key]
            med_new = new_data[col].median()
            std_train = training_stats.get(f"{col}_std", 1)
            # Simple normalized squared distance (not real PSI, but useful)
            psi = ((med_train - med_new) ** 2) / (std_train ** 2 + 1e-6)
            drift[col] = psi
    return drift

def needs_retraining(drift_results: Dict, threshold: float = 0.2) -> bool:
    for col, psi in drift_results.items():
        if psi > threshold:
            return True
    return False
