"""
Configuration dataclass for the AxiPulseAI engine.

Moved verbatim out of operation_health_predictor.py (Phase 2, Step 1). No values or
behavior changed — same fields, same defaults, same __post_init__.
"""

import logging
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    n_estimators: int = 300
    learning_rate: float = 0.05
    random_state: int = 42
    max_iter: int = 500

    mlp_hidden_layers: Tuple[int, int, int] = (256, 128, 64)
    mlp_activation: str = "relu"
    mlp_solver: str = "adam"
    mlp_max_iter: int = 2000
    mlp_early_stopping: bool = True
    mlp_validation_fraction: float = 0.1
    mlp_n_iter_no_change: int = 30

    xgb_early_stopping_rounds: int = 20
    lgb_early_stopping_rounds: int = 20
    cat_early_stopping_rounds: int = 20

    min_days: int = 5
    cold_start_threshold: int = 30
    validation_split: float = 0.8

    de_maxiter: int = 20
    de_popsize: int = 8
    de_tol: float = 0.01
    nm_maxiter: int = 50
    nm_xatol: float = 0.01
    nm_fatol: float = 0.01

    # clip_predictions removed – no clipping ever

    reject_duplicate_dates: bool = False

    tune_n_iter: int = 10
    tune_cv: int = 3

    sample_for_selection: bool = True
    sample_size: int = 10000

    # Temporal CV folds used for normal production model selection.
    # Defaults to 2 to keep selection cheap on large datasets while still
    # producing a rolling-origin signal. Override when more folds are needed.
    cv_folds: int = 2

    # CV parallelism for the production training path. Defaults to a single
    # serial worker so candidate CV never spawns worker processes while the
    # full 1M-row matrix is resident. Raise only when explicitly requested.
    cv_n_jobs: int = 1

    # CV per-model-per-fold timeout in seconds. A candidate fold is executed
    # inside an isolated subprocess; if it does not finish within ``cv_timeout``
    # the worker is SIGKILLed and the candidate is excluded from selection
    # rather than hanging the whole training run.
    cv_timeout: float = 60.0

    # CPU parallelism used ONLY for the FINAL full-data refit of the selected
    # model. CPU tree ensembles (ExtraTrees / RandomForest) and any boosters are
    # forced to n_jobs=1 for the final fit so the final-fit resource guard's
    # estimate is truthful and nested parallelism can never OOM the machine.
    # Model-selection CV is unaffected (it stays serial already via cv_n_jobs).
    final_cpu_n_jobs: int = 1

    # Hard RAM budget (MiB) for the FINAL full-data fit. After CV, every OH
    # candidate's final-fit footprint at the FULL training row count is checked
    # against this budget using the SAME resource estimator as the NPS guard.
    # Candidates whose estimate exceeds the budget are excluded from winner
    # selection (recorded with an explicit reason); the final-fit guard is kept
    # as a second safety layer immediately before the single full-data fit.
    final_fit_memory_budget_mb: float = 4096.0

    # When True, the final-fit memory guard may downscale the selected model's
    # estimator count (e.g. n_estimators / iterations / max_iter) as a last
    # resort to fit within final_fit_memory_budget_mb, preserving the model
    # family but reducing its size. When False (default) an over-budget final
    # fit FAILS with a clear diagnostic rather than being silently altered.
    final_fit_auto_downscale: bool = False

    use_tensorflow: bool = False
    tf_epochs: int = 100
    tf_batch_size: int = 256

    log_level: str = "INFO"
    verbose: bool = False

    use_lag_features: bool = False
    use_cyclical_dates: bool = True
    clip_outliers: bool = True
    ensemble_weighted: bool = True
    strict_prediction: bool = False

    def __post_init__(self):
        logging.getLogger().setLevel(self.log_level)


DEFAULT_CONFIG = Config()
