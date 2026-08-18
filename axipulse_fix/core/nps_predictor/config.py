
"""
AxiPulseAI – Configuration
"""
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class Config:
    # Model hyperparameters
    n_estimators: int = 500
    learning_rate: float = 0.05
    random_state: int = 42
    max_iter: int = 500

    # MLP
    mlp_hidden_layers: Tuple[int, int, int] = (256, 128, 64)
    mlp_activation: str = "relu"
    mlp_solver: str = "adam"
    mlp_max_iter: int = 2000
    mlp_early_stopping: bool = True
    mlp_validation_fraction: float = 0.1
    mlp_n_iter_no_change: int = 30

    # Early stopping for boosting
    xgb_early_stopping_rounds: int = 20
    lgb_early_stopping_rounds: int = 20
    cat_early_stopping_rounds: int = 20

    # Validation
    min_days: int = 5
    cold_start_threshold: int = 30
    validation_split: float = 0.8

    # Optimizer (reverse)
    de_maxiter: int = 20
    de_popsize: int = 8
    de_tol: float = 0.01
    nm_maxiter: int = 50
    nm_xatol: float = 0.01
    nm_fatol: float = 0.01

    # Data
    reject_duplicate_dates: bool = False

    # Tuning
    tune_n_iter: int = 10
    tune_cv: int = 3

    # Sampling
    sample_for_selection: bool = True
    # Model-selection CV runs on a bounded, temporally representative sample.
    # Kept deliberately small (500 rows) so candidate CV is RAM-safe even when
    # the full training matrix is very large. Raise only when a larger
    # selection signal is explicitly required.
    sample_size: int = 500

    # Temporal CV folds used for normal production model selection.
    # Defaults to 2 to keep selection cheap on large datasets while still
    # producing a rolling-origin signal. Override when more folds are needed.
    cv_folds: int = 2

    # CV parallelism for the production training path.
    # Defaults to a single serial worker (n_jobs=1) so candidate CV never
    # spawns worker processes while the full 1M-row matrix is resident.
    # Raise only when explicitly requested.
    cv_n_jobs: int = 1

    # CPU parallelism used ONLY for the FINAL full-data refit of the selected
    # model. CPU tree ensembles (ExtraTrees / RandomForest) and any
    # MultiOutputRegressor wrapper are forced to n_jobs=1 for the final fit so
    # one-tree-batch-per-core memory duplication can never OOM the machine.
    # Model-selection CV is unaffected (it stays serial already via cv_n_jobs).
    final_cpu_n_jobs: int = 1

    # Hard RAM budget (MiB) for the FINAL full-data fit. Before .fit() the
    # resource guard estimates the selected estimator's footprint and first
    # reduces safe parallelism (final_cpu_n_jobs -> 1). If the estimate still
    # exceeds this budget, training FAILS with a clear resource diagnostic
    # instead of letting the OS OOM-kill the machine.
    final_fit_memory_budget_mb: float = 4096.0

    # When True, the final-fit memory guard may downscale the selected model's
    # estimator count (e.g. n_estimators / iterations / max_iter) as a last
    # resort to fit within final_fit_memory_budget_mb, preserving the model
    # family but reducing its size. When False (default) an over-budget final
    # fit FAILS with a clear diagnostic rather than being silently altered.
    final_fit_auto_downscale: bool = False

    # CV per-model-per-fold timeout in seconds.
    #
    # A candidate's fit+predict+metric for a SINGLE fold is executed inside an
    # isolated subprocess. If that fold does not finish within ``cv_timeout``
    # seconds the subprocess is terminated and the candidate is excluded from
    # selection (never left to hang the whole run). Set to a small value to
    # keep selection fast on large datasets; raise for very hard workloads.
    cv_timeout: float = 60.0

    # Hard per-fold RAM ceiling for a single CV worker, in MiB.
    #
    # The parent polls each CV subprocess's RSS while a fold is running. If a
    # worker's RSS exceeds this ceiling the worker is terminated and that
    # candidate is excluded from selection (logged with an explicit reason),
    # instead of risking an out-of-memory crash of the whole machine. Default
    # 2 GiB per CV worker.
    cv_memory_ceiling_mb: float = 2048.0

    # Stricter per-fold timeout used specifically for the MLP candidate, in
    # seconds. The MLP pipeline (StandardScaler + MLPRegressor) is the most
    # RAM/time-hungry candidate and does not support early-stopping the same
    # way the gradient boosters do. When set, MLP folds use this tighter limit
    # and the memory guard above; if MLP still exceeds them it is excluded from
    # the run with an explicit reason rather than removed from the pool.
    cv_mlp_timeout: float = 30.0

    # Logging
    log_level: str = "INFO"
    verbose: bool = False

    # Features
    use_cyclical_dates: bool = True
    clip_outliers: bool = True
    ensemble_weighted: bool = True
    strict_prediction: bool = False

    # Rolling window sizes (days)
    roll_days: Tuple[int, int, int] = (3, 7, 14)

    # Multi-output mode
    use_catboost_multi: bool = True

    # Bounds (moved from hardcoded values)
    call_volume_min: int = 1
    call_volume_max: int = 20000
    ops_health_min: int = 0
    ops_health_max: int = 120
    bc_me_min: int = -100
    bc_me_max: int = 100
    release_rate_min: int = 0
    release_rate_max: int = 100

    # SHAP
    enable_shap: bool = False

    # History buffer length
    history_buffer_days: int = 30

    # Ensemble usage
    use_ensemble: bool = False

    # Optional GPU acceleration for the FINAL full-data fit only.
    # CV / model selection always run on CPU. When True, the final selected
    # model may train on GPU if the model family and installed libraries
    # support it; otherwise training automatically falls back to CPU.
    use_gpu: bool = True

    # Minimum free VRAM (MiB) required to treat a GPU final fit as feasible.
    # When the detected free VRAM is below this threshold the GPU is NOT
    # selected (training stays on CPU) even if a driver is present. This
    # makes "GPU final fit feasible" stricter than "GPU driver exists".
    gpu_min_free_vram_mb: float = 0.0

    # ---- NEW: distribution prediction mode ----
    predict_mode: str = "distribution"  # "distribution" or "binary" (backward compat)
    num_score_buckets: int = 11        # scores 0..10
    score_labels: Tuple[int, ...] = (0,1,2,3,4,5,6,7,8,9,10)

    def __post_init__(self):
        import logging
        logging.getLogger().setLevel(self.log_level)

DEFAULT_CONFIG = Config()
