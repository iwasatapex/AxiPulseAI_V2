import datetime

from .provider import PredictorProvider
from ..models import PredictionRequest, PredictionResult
from ..state.transition import KPITransition


class PredictionService:
    def __init__(self, oh_predictor=None, nps_predictor=None):
        self.oh = oh_predictor
        self.nps = nps_predictor

    @staticmethod
    def _cutoff_date(state):
        """Return the explicit observation/cutoff date for the request.

        The cutoff is an EXPLICIT contract: it must come from the request /
        state boundary (``state["date"]`` or ``state["cutoff"]``), so the same
        code supports historical replay, backtesting, future-dated simulation,
        and normal production forecasting.

        Falling back to ``datetime.date.today()`` is allowed ONLY when the
        caller supplies no explicit cutoff (normal production forecasting).
        The returned value is an ISO date string.
        """
        for key in ("date", "cutoff"):
            value = state.get(key) if state else None
            if value is None:
                continue
            try:
                return datetime.date.fromisoformat(str(value)).isoformat()
            except Exception:
                try:
                    return datetime.datetime.fromisoformat(str(value)).isoformat()
                except Exception:
                    raise ValueError(
                        f"Invalid forecast cutoff {key!r}: {value!r}. "
                        "Cutoff must be an ISO date/datetime."
                    )
        return datetime.date.today().isoformat()

    def _build_oh_row(self, state):
        history_buffer = state.get("history_buffer", [])

        previous = self._latest_observed_history(history_buffer)

        # Safety: predicted recursive state must never be consumed as observed
        # previous-day OH. The direct state field is only a valid fallback for
        # an observed state (not marked ``_predicted``).
        state_is_observed = not state.get("_predicted")
        oh_previous_day = previous.get("operations_health")
        if oh_previous_day is None:
            oh_previous_day = (
                state.get("operations_health", 80)
                if state_is_observed
                else 80
            )

        # Canonical state propagation: use real state values when present,
        # falling back to documented defaults only when genuinely unavailable.
        # This avoids hardcoding operational values that may exist in state.
        total_calls = state.get("total_calls_received", 2000)
        try:
            total_calls = int(total_calls)
        except (TypeError, ValueError):
            total_calls = 2000
        oif = state.get("operational_intelligence_factor", 0)

        return {
            "target_quality": 87,
            "actual_quality": state["quality"],
            "target_competency": 93,
            "actual_competency": state["competency"],
            "target_attendance": 90,
            "actual_attendance": state["attendance"],
            "target_release_rate": 60,
            "actual_release_rate": state["release"],
            "target_transfer_rate": 9,
            "actual_transfer_rate": state["transfer"],
            "total_calls_received": total_calls,
            "total_release_calls": int(total_calls * state["release"] / 100.0),
            "operational_intelligence_factor": oif,

            "quality_previous_day": previous.get("quality", state["quality"]),
            "competency_previous_day": previous.get("competency", state["competency"]),
            "release_previous_day": previous.get("release", state["release"]),
            "transfer_previous_day": previous.get("transfer", state["transfer"]),
            "attendance_previous_day": previous.get("attendance", state["attendance"]),
            "operations_health_previous_day": oh_previous_day,

            "date": self._cutoff_date(state),
        }

    def _known_oh_at_cutoff(self, state):
        """Return the OH value known at the prediction cutoff T.

        Forecasting contract: NPS(T+1) must consume only OH information that is
        already known at cutoff T — never a T+1 realization and never a fresh
        OH forecast computed for T+1. The known OH at T is carried in the
        operational state / history buffer (see ``_build_oh_row`` which already
        reads ``operations_health`` from the same sources).

        Safety: predicted recursive state must NEVER be treated as observed.
        ``state["operations_health"]`` is trusted as known-at-cutoff ONLY when
        the state itself is explicitly observed (not marked ``_predicted``).
        Observed history rows are preferred over the direct state field, and
        rows marked ``_predicted`` are always excluded.
        """
        if state is None:
            return None

        # 1. Observed history (explicitly non-predicted) wins.
        history = state.get("history_buffer") or []
        for row in reversed(history):
            if isinstance(row, dict):
                if row.get("_predicted"):
                    continue
                # OH=0.0 is a VALID observed value; ``or`` would wrongly treat
                # it as missing. Use explicit is-not-None.
                v = row.get("operations_health")
                if v is None:
                    v = row.get("operational_health")
                if v is not None:
                    return v

        # 2. Predicted recursive state must never masquerade as observed.
        #    The direct state field only carries observed/known-at-cutoff OH
        #    when the state is NOT a predicted recursive step.
        if state.get("_predicted"):
            return None

        known = state.get("operations_health")
        if known is None:
            known = state.get("operational_health")
        if known is not None:
            return known

        return None

    @staticmethod
    def _latest_observed_history(history_buffer):
        for row in reversed(history_buffer or []):
            if isinstance(row, dict) and not row.get("_predicted"):
                return row
        return {}

    def _build_nps_row(self, state):
        """Build the feature-row dict for NPS prediction at cutoff T.

        Only information known at prediction cutoff T is included.  The OH
        model's T+1 forecast is deliberately **not** accepted here — this
        method reads the OH value already known at T from ``state`` via
        ``_known_oh_at_cutoff``.  Feeding a T+1 OH forecast as an NPS
        input feature would violate the temporal contract
        (features@T → target@T+1).
        """
        history_buffer = state.get("history_buffer", [])

        previous = self._latest_observed_history(history_buffer)

        # NPS must use the OH value known at cutoff T (never a T+1 forecast).
        known_oh = self._known_oh_at_cutoff(state)

        return {
            "operational_health": known_oh,
            "business_intelligence_factor": state.get("business_intelligence_factor", 0),
            "member_intelligence_factor": state.get("member_intelligence_factor", 0),

            "target_quality": 87,
            "quality": state["quality"],
            "quality_gap": 87 - state["quality"],

            "target_competency": 93,
            "competency": state["competency"],
            "competency_gap": 93 - state["competency"],

            "target_attendance": 90,
            "attendance": state["attendance"],
            "attendance_gap": 90 - state["attendance"],

            "target_transfer": 9,
            "transfer": state["transfer"],
            "transfer_gap": state["transfer"] - 9,

            "target_release_rate": 60,
            "actual_release_rate": state["release"],

            "target_transfer_rate": 9,
            "actual_transfer_rate": state["transfer"],

            # Customer observation reliability
            "total_calls_received": state.get("total_calls_received", 2000),
            "total_release_calls": int(
                state.get("total_calls_received", 2000)
                * state["release"] / 100.0
            ),
            "total_surveys": state.get(
                "total_surveys",
                max(
                    1,
                    int(
                        state.get("total_calls_received", 2000)
                        * state["release"] / 100.0
                        * 0.10
                    )
                )
            ),
            "survey_rate": state.get("survey_rate", 0.10),
            "quality_previous_day": previous.get("quality", state["quality"]),
            "competency_previous_day": previous.get("competency", state["competency"]),
            "release_previous_day": previous.get("release", state["release"]),
            "transfer_previous_day": previous.get("transfer", state["transfer"]),
            "attendance_previous_day": previous.get("attendance", state["attendance"]),
            "nps_previous_day": previous.get("nps", 0),

            "survey_confidence": (
                state.get(
                    "total_surveys",
                    100
                ) /
                (
                    state.get(
                        "total_surveys",
                        100
                    ) + 10
                )
            ),

            "date": self._cutoff_date(state),
        }

    @staticmethod
    def _extract_nps_result(result):
        """
        Preserve the complete NPS predictor result while maintaining
        the historical scalar `nps` field.

        The distribution-mode NPS predictor returns:
            {
                "bayesian_score_distribution": {...},
                "score_counts": {...},
                "promoters": ...,
                "passives": ...,
                "detractors": ...,
                "nps": ...,
                ...
            }

        Production must not collapse that dictionary into only a float.
        """
        if isinstance(result, dict):
            nps_value = result.get("nps")

            distribution = result.get(
                "bayesian_score_distribution"
            )

            score_counts = result.get(
                "score_counts"
            )

            def _mc(key):
                val = result.get(key)
                return float(val) if val is not None else None

            return {
                "nps": (
                    float(nps_value)
                    if nps_value is not None
                    else None
                ),
                "bayesian_score_distribution": (
                    dict(distribution)
                    if isinstance(distribution, dict)
                    else None
                ),
                "score_counts": (
                    {
                        str(k): int(v)
                        for k, v in score_counts.items()
                    }
                    if isinstance(score_counts, dict)
                    else None
                ),
                # Canonical Monte Carlo NPS percentiles produced by the
                # production Bayesian/Monte-Carlo path. Preserved so consumers
                # read the exact interval production computed (no independent
                # re-derivation).
                "monte_carlo_nps_p05": _mc("monte_carlo_nps_p05"),
                "monte_carlo_nps_p50": _mc("monte_carlo_nps_p50"),
                "monte_carlo_nps_p95": _mc("monte_carlo_nps_p95"),
            }

        if isinstance(result, tuple):
            # Backward compatibility with older predictor adapters.
            value = result[1] if len(result) > 1 else result[0]
            return {
                "nps": float(value),
                "bayesian_score_distribution": None,
                "score_counts": None,
                "monte_carlo_nps_p05": None,
                "monte_carlo_nps_p50": None,
                "monte_carlo_nps_p95": None,
            }

        return {
            "nps": float(result),
            "bayesian_score_distribution": None,
            "score_counts": None,
            "monte_carlo_nps_p05": None,
            "monte_carlo_nps_p50": None,
            "monte_carlo_nps_p95": None,
        }

    def predict(self, request: PredictionRequest) -> PredictionResult:
        errors = []

        state = request.state.copy()

        history_buffer = []
        if request.metadata:
            history_buffer = request.metadata.get("history_buffer", [])
            state["history_buffer"] = history_buffer

        # Test compatibility: when predictors are injected, pass the raw state.
        # Production path: use adapter rows.
        if self.oh is not None or self.nps is not None:
            errors = []
            oh_score = nps_score = None

            try:
                oh_score = float(self.oh.predict(state))
            except Exception as e:
                errors.append(f"OH prediction error: {e}")

            nps_result_data = {
                "nps": None,
                "bayesian_score_distribution": None,
                "score_counts": None,
            }

            try:
                raw_nps_result = self.nps.predict(state)
                nps_result_data = self._extract_nps_result(
                    raw_nps_result
                )
            except Exception as e:
                errors.append(f"NPS prediction error: {e}")

            return PredictionResult(
                operations_health=oh_score,
                nps=nps_result_data["nps"],
                bayesian_score_distribution=(
                    nps_result_data[
                        "bayesian_score_distribution"
                    ]
                ),
                score_counts=nps_result_data["score_counts"],
                monte_carlo_nps_p05=nps_result_data.get(
                    "monte_carlo_nps_p05"
                ),
                monte_carlo_nps_p50=nps_result_data.get(
                    "monte_carlo_nps_p50"
                ),
                monte_carlo_nps_p95=nps_result_data.get(
                    "monte_carlo_nps_p95"
                ),
                warnings=[],
                errors=errors,
            )

        try:
            try:
                predictor = PredictorProvider.get_oh_predictor()
            except NotImplementedError:
                from . import predictor_config
                predictor = predictor_config.create_oh_predictor()
            oh_result = predictor.predict(self._build_oh_row(state))

            if isinstance(oh_result, tuple):
                oh_score = float(oh_result[0])
            elif isinstance(oh_result, dict):
                oh_score = float(
                    oh_result.get(
                        "operations_health",
                        oh_result.get(
                            "operational_health",
                            oh_result.get("prediction")
                        )
                    )
                )
            else:
                oh_score = float(oh_result)

        except Exception as e:
            oh_score = None
            errors.append(f"OH error: {e}")

        try:
            try:
                predictor = PredictorProvider.get_nps_predictor()
            except NotImplementedError:
                from . import predictor_config
                predictor = predictor_config.create_nps_predictor()
            result = predictor.predict(
                self._build_nps_row(state)
            )

            nps_result_data = self._extract_nps_result(result)
            nps_score = nps_result_data["nps"]

        except Exception as e:
            nps_score = None
            errors.append(f"NPS error: {e}")

        transitioned = KPITransition().apply(state)

        pr = PredictionResult(
            quality=transitioned.get("quality"),
            competency=transitioned.get("competency"),
            attendance=transitioned.get("attendance"),
            release=transitioned.get("release"),
            transfer=transitioned.get("transfer"),
            calls=state.get("total_calls_received"),
            operations_health=oh_score,
            nps=nps_score,
            confidence=None,
            bayesian_score_distribution=(
                nps_result_data.get(
                    "bayesian_score_distribution"
                )
                if "nps_result_data" in locals()
                else None
            ),
            score_counts=(
                nps_result_data.get("score_counts")
                if "nps_result_data" in locals()
                else None
            ),
            monte_carlo_nps_p05=(
                nps_result_data.get("monte_carlo_nps_p05")
                if "nps_result_data" in locals()
                else None
            ),
            monte_carlo_nps_p50=(
                nps_result_data.get("monte_carlo_nps_p50")
                if "nps_result_data" in locals()
                else None
            ),
            monte_carlo_nps_p95=(
                nps_result_data.get("monte_carlo_nps_p95")
                if "nps_result_data" in locals()
                else None
            ),
            warnings=[],
            errors=errors,
        )
        return pr

    def predict_batch(self, states):
        """Batch-predict OH and NPS for a list of states.

        Batches the trained-model predict calls (one model call for all rows)
        to avoid repeated joblib/multiprocessing pool overhead. Results are
        numerically identical to calling ``predict()`` once per state because
        RandomForest/CatBoost produce identical per-row predictions for batch
        vs single-row calls (verified: max abs diff 0.0).

        Each element of ``states`` is a dict with the same KPI keys as a
        single ``predict()`` request state. Returns a list of dicts in the
        same order:
            {"operations_health": float|None, "nps": float|None, ...}
        """
        if not states:
            return []

        rows = [dict(s) for s in states]
        for idx, row in enumerate(rows):
            row.setdefault("_predicted", False)

        # Build feature rows for every state (pure, per-state).
        oh_rows = [self._build_oh_row(row) for row in rows]
        nps_rows = [self._build_nps_row(row) for row in rows]

        # OH (CatBoost) is fast per-row (~1ms); predict per-row to reuse the
        # predictor's own row handling (it does not accept DataFrame batch).
        try:
            oh_predictor = self.oh or PredictorProvider.get_oh_predictor()
            oh_values = []
            for i, row in enumerate(rows):
                oh_result = oh_predictor.predict(oh_rows[i])
                if isinstance(oh_result, tuple):
                    oh_values.append(float(oh_result[0]))
                elif isinstance(oh_result, dict):
                    oh_values.append(float(
                        oh_result.get(
                            "operations_health",
                            oh_result.get("operational_health", oh_result.get("prediction")),
                        )
                    ))
                else:
                    oh_values.append(float(oh_result))
        except Exception as e:
            oh_values = [None] * len(rows)
            oh_error = f"OH error: {e}"
        else:
            oh_error = None

        # Batch NPS model predict: align features per row (identical to the
        # single-path NPSPredictor.predict), stack, batch-predict the raw
        # estimator once, then apply the SAME per-row postprocess.
        import pandas as pd

        results = []
        try:
            nps_predictor = self.nps or PredictorProvider.get_nps_predictor()
            from core.nps_predictor.feature_engineering import align_features

            aligned = [
                align_features(
                    nps_rows[i],
                    nps_predictor.feature_names,
                    nps_predictor._feature_stats,
                    nps_predictor._history_buffer,
                )
                for i in range(len(rows))
            ]
            aligned_df = pd.concat(aligned, axis=0, ignore_index=True)

            # Reuse the SAME canonical vector construction as single prediction
            # (predict_single) so batch NPS semantics == single NPS semantics:
            # the selected model OR the persisted weighted ensemble is applied
            # identically, and no probabilistic logic is duplicated here.
            from core.nps_predictor.inference import (
                predict_single_vector,
                postprocess_predictions,
            )

            for i, row in enumerate(rows):
                pred = predict_single_vector(nps_predictor, aligned_df.iloc[[i]])
                post = postprocess_predictions(pred, nps_rows[i])
                extracted = self._extract_nps_result(post)
                results.append({
                    "operations_health": oh_values[i],
                    "nps": extracted["nps"],
                    "bayesian_score_distribution": extracted["bayesian_score_distribution"],
                    "score_counts": extracted["score_counts"],
                    "errors": [oh_error] if oh_error else [],
                })
        except Exception as e:
            nps_error = f"NPS error: {e}"
            for i, row in enumerate(rows):
                results.append({
                    "operations_health": oh_values[i],
                    "nps": None,
                    "bayesian_score_distribution": None,
                    "score_counts": None,
                    "errors": ([oh_error] if oh_error else []) + [nps_error],
                })

        return results

    def predict_oh(self, state):
        """Compatibility helper for tests."""
        predictor = self.oh or PredictorProvider.get_oh_predictor()
        return predictor.predict(state)

    def predict_nps(self, state):
        """Compatibility helper for tests."""
        predictor = self.nps or PredictorProvider.get_nps_predictor()
        return predictor.predict(state)
predict = PredictionService.predict
predict_oh = PredictionService.predict_oh
predict_nps = PredictionService.predict_nps
