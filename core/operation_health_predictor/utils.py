"""
Shared setup and optional-dependency imports for the AxiPulseAI
Index engine.

Moved verbatim out of operation_health_predictor.py (Phase 2, Step 1): logging config,
reproducibility seeding, and the optional-library try/except imports.

Note: in the original single-file module, XGBRegressor / LGBMRegressor /
CatBoostRegressor / shap / tf / keras / layers / callbacks / tqdm were only
ever referenced from inside blocks already guarded by the matching
*_AVAILABLE flag, so an unbound name never actually surfaced. Here each
`except ImportError` branch explicitly sets the names to None so other
modules can import them unconditionally (e.g. `from .utils import keras`)
without needing tensorflow/xgboost/etc. installed. This changes nothing
about run-time behavior — the guarded call sites are unchanged.
"""


import logging
import os
import random
import warnings

import numpy as np


from contextlib import contextmanager

@contextmanager
def tqdm_joblib(*args, **kwargs):
    yield


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- Reproducibility ---
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(42)
np.random.seed(42)

# --- Optional libraries ---
TQDM_AVAILABLE = True


# ----------------------------------------------------------------------
# Compatibility exports expected by the refactored modules
# ----------------------------------------------------------------------

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

# TensorFlow / Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_AVAILABLE = True
except Exception:
    tf = None
    keras = None
    layers = None
    callbacks = None
    TF_AVAILABLE = False

# XGBoost
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGBRegressor = None
    XGB_AVAILABLE = False

# LightGBM
try:
    from lightgbm import LGBMRegressor
    LGB_AVAILABLE = True
except Exception:
    LGBMRegressor = None
    LGB_AVAILABLE = False

# CatBoost
try:
    from catboost import CatBoostRegressor
    CAT_AVAILABLE = True
except Exception:
    CatBoostRegressor = None
    CAT_AVAILABLE = False

# tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    tqdm = None
    TQDM_AVAILABLE = False
