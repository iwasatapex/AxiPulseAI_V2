"""
Training pipeline: cold-start / rolling-origin model selection, refit,
feature importance, and metadata bookkeeping.

Moved verbatim out of OperationalHealthPredictor in operation_health_predictor.py
(Phase 2, Step 7 — no logic changed). This was the largest chunk of the
original file; it is split further here into its own module but the
methods themselves are untouched.
"""

import hashlib
import importlib.metadata
import logging as _logging
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from .constants import MODEL_VERSION
from ..common.temporal_dataset import shift_target_next_day, date_aware_splits
from ..common.cv_runner import evaluate_fold_in_subprocess
from ..nps_predictor.resource_guard import final_fit_feasible, guard_final_fit
from .utils import (
    CatBoostRegressor,
    LGBMRegressor,
    TF_AVAILABLE,
    TQDM_AVAILABLE,
    XGBRegressor,
    callbacks,
    logger,
    tqdm,
)


def _resolve_trajectory_ids(df: pd.DataFrame):
    """Return a trajectory identity Series aligned to df rows, or None.

    A column is used as trajectory identity only when it is actually a
    trajectory: the same value must occur on multiple distinct dates.  Columns
    such as ``scenario_id`` are sometimes just per-row simulation labels (each
    row a distinct scenario on a single date) — using those as trajectory keys
    would produce zero valid ``T -> T+1`` pairs.  When no genuine multi-day
    trajectory exists, ``None`` is returned so the temporal helper falls back
    to its stable occurrence-based alignment.
    """
    for col in ("trajectory_id", "simulation_id", "run_id", "agent_id"):
        if col in df.columns:
            # These are explicit trajectory ids by contract.
            return df[col].reset_index(drop=True)

    if "scenario_id" in df.columns:
        spans = df.groupby(df["scenario_id"], sort=False)["date"].transform("nunique")
        if (spans > 1).any():
            return df["scenario_id"].reset_index(drop=True)

    return None


def evaluate_fold(predictor, X, y, base_models, model_names, fold_idx, train_idx, val_idx):
    """Canonical single-fold evaluation for rolling-origin model selection."""
    _logging.disable(_logging.CRITICAL)
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    fold_results = {}
    for name in model_names:
        try:
            model = clone(base_models[name])
            if name == "XGBoost" and XGBRegressor is not None:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            elif name == "LightGBM" and LGBMRegressor is not None:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            elif name == "CatBoost" and CatBoostRegressor is not None:
                model.fit(X_train, y_train, eval_set=(X_val, y_val))
            elif name == "TensorFlow" and TF_AVAILABLE and predictor._tf_model is not None:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                tf_model = predictor._build_tf_model()
                tf_model.fit(
                    X_train_scaled, y_train,
                    epochs=min(20, predictor.config.tf_epochs),
                    batch_size=predictor.config.tf_batch_size,
                    validation_split=0.1,
                    verbose=0,
                )
                pred_arr = tf_model.predict(X_val_scaled)
                mae = mean_absolute_error(y_val, pred_arr)
                fold_results[name] = mae
                continue
            else:
                model.fit(X_train, y_train)
            pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, pred)
            fold_results[name] = mae
        except Exception as e:
            fold_results[name] = None
            if predictor.config.verbose:
                logger.warning(f"Model {name} failed at fold {fold_idx}: {e}")
    _logging.disable(_logging.NOTSET)
    return fold_results


class TrainerMixin:
    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------
    def _progress_emit(self, progress, stage, message=None, percent=None):
        """Best-effort progress emission; never raises during training."""
        if progress is None:
            return
        try:
            progress.set_stage(stage, message=message, percent=percent)
        except Exception:  # pragma: no cover - advisory only
            pass

    def train(self, filepath, tune=False, progress=None):
        self.history_file = filepath
        self._progress_emit(progress, "loading", f"Loading dataset {filepath}")
        df, issue_cols = self.load_data(filepath)
        self._issue_cols = issue_cols

        self._clear_reverse_cache()

        # ---------------------------------------------------------------
        # Numeric cleanup MUST happen before validation
        # ---------------------------------------------------------------
        numeric_cleanup_cols = [
            "operational_intelligence_factor",
            "operational_health",
            "target_quality", "actual_quality",
            "target_competency", "actual_competency",
            "target_attendance", "actual_attendance",
            "target_release_rate", "actual_release_rate",
            "target_transfer_rate", "actual_transfer_rate",
            "total_calls_received",
        ]

        for col in numeric_cleanup_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in self._issue_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        if "operational_intelligence_factor" in df.columns:
            df["operational_intelligence_factor"] = (
                df["operational_intelligence_factor"]
                .fillna(0.0)
                .clip(-100, 100)
            )

        for idx, row in df.iterrows():
            valid, warns = self.validate_training(row)
            if not valid:
                raise ValueError(f"Row {idx} (date {row['date']}) failed training validation: {warns}")

        trajectory_ids = _resolve_trajectory_ids(df)
        self._progress_emit(progress, "feature_engineering", "Engineering features")
        X = self.prepare_features(df)
        y_raw = pd.to_numeric(df["operational_health"], errors="coerce")

        # Align only rows that carry an actual OH value.
        df_clean = df.dropna(subset=["operational_health"])
        X = X.loc[df_clean.index].reset_index(drop=True)

        # Forecasting contract: feature_time[T] < target_time[T+1].
        # Forward-align the target so row T is labeled by OH realized at T+1.
        # The final source day has no T+1 target and is dropped here.
        self._progress_emit(progress, "preparing_targets", "Aligning temporal targets")
        y, _target_times = shift_target_next_day(
            y_raw.loc[df_clean.index].reset_index(drop=True),
            df_clean["date"].reset_index(drop=True),
            trajectory_ids=trajectory_ids.loc[df_clean.index].reset_index(drop=True) if trajectory_ids is not None else None,
            field_name="operational_health",
        )
        has_target = y.notna()
        X = X.loc[has_target]
        y = y.loc[has_target]

        if len(X) < self.config.min_days:
            raise ValueError(f"Need at least {self.config.min_days} days of data.")

        self.history_days = int(df_clean["date"].nunique())
        self._training_dates = df_clean["date"].reset_index(drop=True).loc[has_target].reset_index(drop=True)
        self._fallback_value = y.mean() if len(y) > 0 else 100.0

        X = X.replace([np.inf, -np.inf], np.nan)

        self._feature_stats = self._compute_feature_stats(X)

        for col in X.columns:
            med = self._feature_stats.get(f"{col}_median", 0)
            X[col] = X[col].fillna(med)

        if self.config.clip_outliers:
            for col in X.columns:
                if col in self._feature_stats:
                    med = self._feature_stats.get(f"{col}_median", 0)
                    q1 = self._feature_stats.get(f"{col}_q1", 0)
                    q3 = self._feature_stats.get(f"{col}_q3", 0)
                    iqr = q3 - q1
                    if iqr > 0:
                        low = q1 - 3 * iqr
                        high = q3 + 3 * iqr
                        X[col] = np.clip(X[col], low, high)
                    else:
                        std = self._feature_stats.get(f"{col}_std", 1)
                        low = med - 5 * std
                        high = med + 5 * std
                        X[col] = np.clip(X[col], low, high)

        self.feature_names = list(X.columns)
        logger.debug(f"Feature names: {self.feature_names}")

        if self.config.sample_for_selection and len(X) > self.config.sample_size:
            # Sample at most the configured number of ROWS (sample_size is a
            # row-count limit, not a date-count limit). The 1M-row OH dataset
            # has ~2.5k distinct dates but >1M rows, so the previous
            # ``history_days > sample_size`` guard never activated.
            #
            # We take the most recent sample_size rows (tail), which preserves
            # temporal order, and we carry the EXACT sampled dates alongside so
            # rolling-origin CV splits the same rows the selection actually
            # trains on — never pairing tail-sampled X/y with unrelated first-N
            # dates.
            X_sample = X.iloc[-self.config.sample_size:]
            y_sample = y.iloc[-self.config.sample_size:]
            self._training_dates = self._training_dates.iloc[-self.config.sample_size:].reset_index(drop=True)
            logger.info(
                "Sampled %d most recent rows for selection CV "
                "(full rows=%d, distinct dates=%d).",
                self.config.sample_size, len(X), self.history_days,
            )
        else:
            X_sample = X
            y_sample = y

        if self.config.use_tensorflow and TF_AVAILABLE:
            self._tf_model = self._build_tf_model()
            if self._tf_model is not None:
                self._tf_scaler = StandardScaler()

        # Record the FULL training dimensions so the resource-aware selection
        # can evaluate every candidate's final-fit feasibility at the real
        # full-data row/col count (CV runs on the bounded sample). A caller may
        # pre-set these for staging/tests; otherwise they default to the full X.
        if getattr(self, "_full_training_rows", None) is None:
            self._full_training_rows = len(X)
        if getattr(self, "_full_training_cols", None) is None:
            self._full_training_cols = X.shape[1]

        if self.history_days < self.config.cold_start_threshold:
            logger.info("⚡ Cold-start mode.")
            self._progress_emit(progress, "model_selection", "Cold-start model selection")
            self._cold_start_train(X_sample, y_sample)
        else:
            logger.info("🔄 Rolling-origin mode with TimeSeriesSplit.")
            self._progress_emit(progress, "model_selection", "Rolling-origin model selection (CV)")
            self._rolling_origin_train(X_sample, y_sample, tune, progress=progress)

        if self.model is None:
            self.trained = False
            raise RuntimeError("OH model selection did not produce a production model.")
        selected_name = self.model_name
        try:
            logger.info("🔄 Full-data refit of selected OH model (%s) on %d rows...", selected_name, len(X))
            if progress is not None:
                try:
                    progress.set_final_fit(selected_name, device="cpu", rows=int(len(X)))
                except Exception:  # pragma: no cover - advisory
                    pass
            if selected_name == "TensorFlow" and TF_AVAILABLE and self._tf_model is not None:
                self._train_tf(X, y)
            else:
                # Second safety layer: guard the FINAL full-data fit exactly as
                # NPS does. Selection already excluded infeasible candidates,
                # but the guard is retained so no path can OOM the machine on
                # the peak-RAM full-data refit. CPU-only for OH (no GPU fit).
                # The OH target is a 1-D Series while the shared guard expects
                # a 2-D target (NPS is multi-output), and OH candidates are raw
                # single-output estimators while the guard unwraps `.estimator`
                # (the shape NPS models have). Present the guard with the same
                # shapes it is designed for via a small proxy; the real fitted
                # model is never substituted or mutated.
                guard_y = y.to_frame() if isinstance(y, pd.Series) else y
                guard_proxy = SimpleNamespace(
                    model=MultiOutputRegressor(self.model),
                    model_name=selected_name,
                    config=self.config,
                )
                guard_final_fit(guard_proxy, X, guard_y, device="cpu")
                self.model.fit(X, y)
        except Exception as exc:
            self.trained = False
            self.model = None
            if progress is not None:
                try:
                    progress.fail(f"OH final refit failed for {selected_name}")
                except Exception:  # pragma: no cover - advisory
                    pass
            raise RuntimeError(f"OH full-data refit failed for selected model '{selected_name}'.") from exc
        self._all_models = {selected_name: self.model}
        self.trained = True
        logger.info("✅ OH full-data refit complete.")

        if progress is not None:
            try:
                progress.complete(model_name=selected_name, rows=int(len(X)), history_days=self.history_days)
            except Exception:  # pragma: no cover - advisory
                pass

        self._update_metadata(X, y)
        self._compute_feature_importance(X, y)
        self._compute_shap(X)
        self._clear_reverse_cache()
        logger.info(f"🎯 Training complete. Model: {self.model_name}, days: {self.history_days}")

    def _train_tf(self, X, y):
        if not TF_AVAILABLE or self._tf_model is None:
            return
        try:
            X_scaled = self._tf_scaler.fit_transform(X)
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            self._tf_model.fit(
                X_scaled, y,
                epochs=self.config.tf_epochs,
                batch_size=self.config.tf_batch_size,
                validation_split=0.1,
                verbose=1 if self.config.verbose else 0,
                callbacks=[early_stop],
            )
            self._all_models["TensorFlow"] = self._tf_model
            logger.info("TensorFlow model trained.")
        except Exception as e:
            logger.warning(f"TensorFlow training failed: {e}")

    # ---------------------------------------------------------------
    # Cold-start
    # ---------------------------------------------------------------
    def _cold_start_train(self, X, y):
        models = self.create_model_registry(self.config, cold_start=True)
        iterator = tqdm(models.items(), desc="Training models", unit="model") if TQDM_AVAILABLE and self.config.verbose else models.items()

        for name, model in iterator:
            try:
                model.fit(X, y)
                self._all_models[name] = model
            except Exception as e:
                logger.warning(f"Model {name} failed to train (cold-start): {e}")
        if not self._all_models:
            self.trained = False
            raise RuntimeError("No models trained in cold-start mode.")
        self.model_name = next(iter(self._all_models))
        self.model = self._all_models[self.model_name]
        self.trained = False  # caller performs the sole full-data refit
        logger.info(f"✅ Cold-start: selected {self.model_name}; {len(self._all_models)} candidates stored.")

    # ---------------------------------------------------------------
    # Rolling-origin
    # ---------------------------------------------------------------
    def _rolling_origin_train(self, X, y, tune=False, progress=None):
        n = len(X)
        if n < 10:
            split_idx = int(0.8 * n)
            X_train, y_train = X.iloc[:split_idx], y.iloc[split_idx:]
            X_val, y_val = X.iloc[split_idx:], y.iloc[split_idx:]
            self._train_and_select_simple(X_train, y_train, X_val, y_val)
            return

        # Configurable date-aware folds (default 2). The temporal splitting
        # algorithm (date_aware_splits) is unchanged — only the fold count is
        # configurable now.
        cv_folds = max(1, int(getattr(self.config, "cv_folds", 2)))
        split_list = list(
            date_aware_splits(
                self._training_dates.iloc[:n].reset_index(drop=True),
                n_splits=cv_folds,
            )
        )

        base_models = self.create_model_registry(self.config, cold_start=False)
        model_names = list(base_models.keys())

        timeout = float(getattr(self.config, "cv_timeout", 60.0))

        total_models = len(model_names)
        total_folds = len(split_list)

        # Per-run timing stats (fold times, per-model totals, timeouts, errors).
        cv_timing = {
            "timeout": timeout,
            "model_fold_elapsed": {},  # {name: [fold_elapsed, ...]}
            "model_total_elapsed": {},  # {name: total seconds}
            "fold_times": [],          # flat list of every completed fold's seconds
            "timeouts": [],            # [name, ...]
            "errors": [],              # [(name, error), ...]
            "completed": [],           # [name, ...]
        }

        if progress is not None:
            try:
                progress.set_models(total_models)
            except Exception:  # pragma: no cover - advisory
                pass

        perf = {name: [] for name in model_names}

        for idx, (name, model) in enumerate(base_models.items(), start=1):
            if progress is not None:
                try:
                    progress.start_candidate(name, total_models=total_models)
                    progress.set_stage(
                        "model_selection",
                        message=f"Evaluating {name} ({idx}/{total_models})",
                    )
                except Exception:  # pragma: no cover - advisory
                    pass

            model_start = time.monotonic()
            fold_maes = []
            excluded = False

            for fold_number, (train_idx, val_idx) in enumerate(
                split_list, start=1
            ):
                if progress is not None:
                    try:
                        progress.start_fold(fold_number, total_folds=len(split_list))
                    except Exception:  # pragma: no cover - advisory
                        pass

                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

                logger.info(
                    "CV model %s fold %d/%d START",
                    name,
                    fold_number,
                    total_folds,
                )

                def _heartbeat():
                    if progress is not None:
                        try:
                            progress.set_stage(
                                "model_selection",
                                message=(
                                    f"Evaluating {name} ({idx}/{total_models}) "
                                    f"fold {fold_number}/{total_folds} ..."
                                ),
                            )
                        except Exception:  # pragma: no cover - advisory
                            pass

                # Each fold runs in an isolated subprocess with a hard timeout.
                # CV is CPU-only; GPU is never used here.
                result = evaluate_fold_in_subprocess(
                    name,
                    model,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    timeout=timeout,
                    metric="mae",
                    heartbeat=_heartbeat,
                )

                elapsed = float(result.get("elapsed", 0.0))
                cv_timing["model_fold_elapsed"].setdefault(name, []).append(elapsed)
                cv_timing["fold_times"].append(elapsed)
                status = result.get("status")

                if status == "ok":
                    mae = float(result["score"])
                    logger.info(
                        "CV model %s fold %d/%d DONE elapsed=%.2fs score=MAE=%.4f",
                        name,
                        fold_number,
                        total_folds,
                        elapsed,
                        mae,
                    )
                    fold_maes.append(mae)
                elif status == "timeout":
                    logger.warning(
                        "CV TIMEOUT: %s fold %d (%.2fs > %.2fs limit). "
                        "Candidate excluded from selection.",
                        name,
                        fold_number,
                        elapsed,
                        timeout,
                    )
                    cv_timing["timeouts"].append(name)
                    excluded = True
                    break
                else:  # status == "error"
                    err = result.get("error", "unknown error")
                    logger.warning(
                        "CV model %s fold %d failed: %s. Candidate excluded.",
                        name,
                        fold_number,
                        err,
                    )
                    cv_timing["errors"].append((name, err))
                    excluded = True
                    break

                if progress is not None:
                    try:
                        progress.set_stage(
                            "model_selection",
                            message=(
                                f"Evaluated {name} fold {fold_number}/{total_folds} "
                                f"({elapsed:.1f}s)"
                            ),
                        )
                    except Exception:  # pragma: no cover - advisory
                        pass

            cv_timing["model_total_elapsed"][name] = (
                time.monotonic() - model_start
            )

            if excluded or not fold_maes:
                continue

            cv_timing["completed"].append(name)
            perf[name] = fold_maes

            if progress is not None:
                try:
                    progress.complete_candidate()
                except Exception:  # pragma: no cover - advisory
                    pass

        avg_perf = {k: float(np.mean(v)) for k, v in perf.items() if v}
        if not avg_perf:
            raise RuntimeError(
                "No model produced valid predictions during OH temporal CV. "
                "All candidates failed or timed out."
            )

        if cv_timing["timeouts"]:
            logger.warning(
                "OH CV candidates excluded due to timeout: %s",
                ", ".join(sorted(set(cv_timing["timeouts"]))),
            )
        if cv_timing["errors"]:
            logger.warning(
                "OH CV candidates excluded due to error: %s",
                ", ".join(sorted({n for n, _ in cv_timing["errors"]})),
            )

        # ------------------------------------------------------------
        # Resource-aware final-fit feasibility (deployment feasibility).
        #
        # A candidate may win CV yet be unable to safely perform the FULL final
        # refit under final_fit_memory_budget_mb at the full training row count.
        # Evaluate every candidate after CV; exclude infeasible ones from winner
        # selection with an explicit reason, then pick the best OH MAE among the
        # FEASIBLE candidates only. All candidates remain in the registry — this
        # is resource-aware selection, not candidate deletion. The final-fit
        # guard is retained as a second safety layer before the single full-data
        # fit. The estimator reused here (final_fit_feasible) is exactly the NPS
        # resource guard's — no separate memory formula is invented.
        # ------------------------------------------------------------
        budget_mb = float(getattr(self.config, "final_fit_memory_budget_mb", 4096.0))
        final_n_jobs = int(getattr(self.config, "final_cpu_n_jobs", 1))
        full_rows = int(getattr(self, "_full_training_rows", len(X)))
        full_cols = int(getattr(self, "_full_training_cols", X.shape[1]))

        resource_diagnostics = {}
        infeasible_reasons = {}
        feasible_names = []

        for name in base_models.keys():
            diag = {
                "cv_score": avg_perf.get(name),
                "final_fit_estimated_memory_mb": None,
                "final_fit_feasible": True,
                "reason_if_not_feasible": None,
            }
            # A candidate that never produced a CV score is already excluded; it
            # cannot win, but still report its resource status best-effort.
            if name not in avg_perf:
                diag["final_fit_feasible"] = False
                diag["reason_if_not_feasible"] = "no CV score (failed/timed out/excluded)"
                resource_diagnostics[name] = diag
                continue

            feasible, reason, fdiag = final_fit_feasible(
                name,
                # OH candidates are raw single-output estimators, whereas the
                # NPS guard is designed around MultiOutputRegressor-wrapped
                # candidates (its inner_estimator unwraps `.estimator`). Present
                # the same wrapper shape so the EXACT NPS estimation path runs;
                # the raw estimator is never mutated.
                MultiOutputRegressor(base_models[name]),
                rows=full_rows,
                cols=full_cols,
                n_outputs=1,  # OH is single-output
                budget_mb=budget_mb,
                n_jobs=final_n_jobs,
                device="cpu",  # OH final fit runs on CPU (no GPU path)
            )
            diag.update(fdiag)
            diag["final_fit_feasible"] = feasible
            diag["reason_if_not_feasible"] = reason
            resource_diagnostics[name] = diag

            if feasible:
                feasible_names.append(name)
            else:
                infeasible_reasons[name] = reason

        self.model_selection_diagnostics = resource_diagnostics

        for name, reason in infeasible_reasons.items():
            logger.warning(
                "%s excluded from OH winner selection: %s",
                name,
                reason,
            )

        if not feasible_names:
            details = "; ".join(
                f"{n} ({resource_diagnostics[n].get('reason_if_not_feasible') or 'infeasible'})"
                for n in base_models.keys()
                if n in resource_diagnostics
            )
            raise RuntimeError(
                "No OH candidate is final-fit feasible under "
                "final_fit_memory_budget_mb=%.0fMB at %d rows. "
                "Every candidate's estimated memory/resource reason: %s. "
                "To proceed: raise final_fit_memory_budget_mb, enable "
                "final_fit_auto_downscale=True, or train on fewer rows."
                % (budget_mb, full_rows, details)
            )

        # Pick the best CV MAE among the FEASIBLE candidates only.
        feasible_perf = {name: avg_perf[name] for name in feasible_names}
        best_name = min(feasible_perf, key=feasible_perf.get)
        best_mae = avg_perf[best_name]

        self.cv_timing = cv_timing
        self.algorithm_performance = avg_perf

        # Leave self.model as an UNFIT clone; the SOLE full-data refit happens
        # in train(). No candidate is full-fitted here and no sample-fitted
        # winner is retained, preserving the final-fit RAM optimization.
        self.model = clone(base_models[best_name])
        self.model_name = best_name
        self._all_models = {}

        logger.info(
            "✅ Rolling-origin: selected %s (avg MAE=%.2f). "
            "Full-data refit deferred to train().",
            best_name,
            best_mae,
        )

    def _train_and_select_simple(self, X_train, y_train, X_val, y_val):
        models = self.create_model_registry(self.config, cold_start=False)
        best_mae = float("inf")
        best_model = None
        best_name = None
        perfs = {}
        for name, model in tqdm(models.items(), desc="Training models", leave=True):
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, pred)
                perfs[name] = mae
                self._all_models[name] = model
                if mae < best_mae:
                    best_mae = mae
                    best_model = model
                    best_name = name
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
        if best_model is None:
            raise RuntimeError("No model succeeded in training.")
        self.model = best_model
        self.model_name = best_name
        self.algorithm_performance = perfs
        logger.info(f"✅ Simple split: selected {best_name} (MAE={best_mae:.2f}).")

    # ---------------------------------------------------------------
    # Feature Importance
    # ---------------------------------------------------------------
    def _compute_feature_importance(self, X, y):
        if self.model is not None:
            try:
                result = permutation_importance(
                    self.model, X, y,
                    n_repeats=5,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                )
                importances = result.importances_mean
                if len(importances) == len(self.feature_names):
                    self._feature_importance = dict(zip(self.feature_names, importances))
                else:
                    self._feature_importance = {}
                logger.info("Permutation importance computed.")
                return
            except Exception as e:
                logger.warning(f"Permutation importance failed: {e}")

        if self._all_models and len(self._all_models) > 0:
            importances = {}
            for name, model in self._all_models.items():
                try:
                    if hasattr(model, 'feature_importances_'):
                        imp = model.feature_importances_
                        if len(imp) == len(self.feature_names):
                            for f, v in zip(self.feature_names, imp):
                                importances[f] = importances.get(f, 0) + v / len(self._all_models)
                except Exception:
                    pass
            self._feature_importance = importances if importances else {}
        else:
            self._feature_importance = {}

    def get_feature_importance(self):
        if not self.trained:
            raise RuntimeError("Model not trained.")
        return dict(sorted(self._feature_importance.items(), key=lambda x: x[1], reverse=True))

    # ---------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------
    def _update_metadata(self, X, y):
        lib_versions = {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": importlib.metadata.version("scikit-learn"),
            "joblib": joblib.__version__,
        }
        for lib in ["xgboost", "lightgbm", "catboost", "shap"]:
            try:
                lib_versions[lib] = importlib.metadata.version(lib)
            except Exception:
                lib_versions[lib] = None

        file_hash = None
        if self.history_file and Path(self.history_file).exists():
            try:
                sha = hashlib.sha256()
                with open(self.history_file, "rb") as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        sha.update(chunk)
                file_hash = sha.hexdigest()
            except Exception:
                file_hash = None

        self.metadata = {
            "engine_version": MODEL_VERSION,
            "trained_at": datetime.now().isoformat(),
            "training_rows": len(y),
            "feature_names": self.feature_names,
            "model_name": self.model_name,
            "algorithm_performance": self.algorithm_performance,
            "fallback_value": self._fallback_value,
            "library_versions": lib_versions,
            "file_hash": file_hash,
            "target_mean": float(y.mean()),
            "target_std": float(y.std()),
            "feature_count": len(self.feature_names),
            "issue_count": len(self._issue_cols),
            "config": {k: v for k, v in self.config.__dict__.items() if not k.startswith("_")},
        }

        # Persist resource-aware selection diagnostics (final-fit feasibility)
        # as per-candidate metadata. Never fabricated: only computed values from
        # the selection stage are recorded.
        if getattr(self, "model_selection_diagnostics", None):
            self.metadata["model_selection_diagnostics"] = self.model_selection_diagnostics


def train(predictor, filepath, tune=False, progress=None):
    """Compatibility API for TrainerMixin.train()."""
    if predictor is None:
        raise TypeError(
            "train() requires an OperationalHealthPredictor "
            "instance as the first argument."
        )
    return TrainerMixin.train(
        predictor,
        filepath,
        tune=tune,
        progress=progress,
    )


def get_feature_importance(predictor):
    """Compatibility API for TrainerMixin.get_feature_importance()."""
    if predictor is None:
        raise TypeError(
            "get_feature_importance() requires an OperationalHealthPredictor "
            "instance as the first argument."
        )
    return TrainerMixin.get_feature_importance(predictor)
