"""
ForecastOrchestrator – Recursive forecasting with ScenarioManager.
Orchestrates predictions, state evolution, and scenario application.
"""
import datetime
import logging
from typing import Dict, Any, Optional, List
from dataclasses import asdict

from ..base_engine import ForecastAIEngine
from ..models import (
    ForecastRequest, ForecastResponse, ForecastDay, ForecastResult,
    PredictionRequest
)
from ..prediction import PredictionService
from ..state import OperationalState, StateEvolutionEngine
from ..storage import HistoryStore
from ..validation import ForecastBacktester
from ..calibration import ForecastCalibration
from ..scenarios import ScenarioManager
from .confidence_engine import ConfidenceEngine
from .risk_engine import RiskEngine
from core.decision_intelligence.v3.integration.production_boundary import (
    ProductionDecisionBoundary,
)
from core.decision_intelligence.v3.integration.probabilistic_decision import (
    ProbabilisticDecisionService,
)
from core.decision_intelligence.v3.integration.probabilistic_adapter import (
    UniversalProbabilisticAdapter,
)
from core.decision_intelligence.v3.integration.decision_composer import (
    compose_decision_package,
)

logger = logging.getLogger(__name__)

class ForecastOrchestrator(ForecastAIEngine):
    def __init__(self, 
                 prediction_service: Optional[PredictionService] = None,
                 evolution_engine: Optional[StateEvolutionEngine] = None,
                 scenario_manager: Optional[ScenarioManager] = None):
        self.service = prediction_service or PredictionService()
        self.evolution = evolution_engine or StateEvolutionEngine()
        self.scenario_manager = scenario_manager or ScenarioManager()
        self.confidence_engine = ConfidenceEngine()
        self.risk_engine = RiskEngine()

        # Phase 3: separate actual and forecast memory
        self.actual_history = []
        self.forecast_history = []
        self.forecast_error_history = []
        self.history_store = HistoryStore()
        self.backtester = ForecastBacktester()
        self.calibration = ForecastCalibration()

        stored = self.history_store.load()

        self.forecast_history = stored.get(
            'forecast_history',
            self.forecast_history
        )

        self.actual_history = stored.get(
            'actual_history',
        self.actual_history
        )

        self.forecast_error_history = stored.get(
            'forecast_error_history',
        self.forecast_error_history
        )
    def execute(self, request: ForecastRequest) -> ForecastResponse:
        horizon = request.horizon or 1
        if horizon < 1:
            return self._error_response("Horizon must be at least 1.")

        state_dict = self._get_state_from_request(request)
        if state_dict is None:
            return self._error_response("No operational state provided in request.parameters['state']")

        # Explicit forecast cutoff (supports historical replay, backtesting,
        # and future-dated simulation). Falls back to today only when the
        # caller supplies none (normal production forecasting).
        explicit_cutoff = self._explicit_cutoff(request, state_dict)
        forecast_start = explicit_cutoff or datetime.date.today().isoformat()
        if explicit_cutoff is not None:
            state_dict = dict(state_dict)
            state_dict.setdefault("date", explicit_cutoff)

        # Capture the OBSERVED current state (caller-supplied) BEFORE the
        # recursive loop. The loop reassigns ``state_dict`` to evolved,
        # predicted states; the ADIE V3 handoff must consume only the
        # observed request state, never predicted recursive state.
        observed_state = dict(state_dict)

        current_state = OperationalState.from_dict(state_dict)
        timeline: List[ForecastDay] = []
        errors: List[str] = []
        warnings: List[str] = []

        forecast_history = []

        # Phase 3.2 recursive memory
        history_buffer = []

        start_time = datetime.datetime.now()
    
        for day_num in range(1, horizon + 1):
            try:
                # 1. Apply scenarios to current state (OperationalState)
                modified_state = self.scenario_manager.apply_scenarios(
                    current_state, 
                    day=day_num
                )

                # 2. Prepare prediction request
                # Recursive days (>=2) carry predicted state evolved from a
                # prior forecast. Mark it explicitly so consumers (e.g.
                # PredictionService) never treat predicted OH/NPS as observed
                # or known-at-cutoff data.
                state_dict = modified_state.to_dict()
                if day_num > 1:
                    state_dict["_predicted"] = True

                pred_req = PredictionRequest(
                    state=state_dict,
                    metadata={
                        "day": day_num,
                        "history_buffer": history_buffer.copy()
                    }
                )
                pred_result = self.service.predict(pred_req)

                forecast_payload = {
                    "timeline": [
                        {
                            "operations_health": pred_result.operations_health,
                            "nps": pred_result.nps,
                            "quality": modified_state.quality,
                            "competency": modified_state.competency,
                            "transfer": modified_state.transfer,
                            "release": modified_state.release,
                            "attendance": modified_state.attendance,
                        }
                    ]
                }

                confidence = None
                risk = None
                # Ensure confidence_obj is always bound (even when the
                # confidence engine reports failure on this day) so the risk
                # engine below never reads a stale/undefined value.
                confidence_obj = None

                confidence_response = self.confidence_engine.execute(
                    ForecastRequest(
                        operation="confidence",
                        parameters={
                            "forecast_result": forecast_payload
                        }
                    )
                )

                if confidence_response.success:
                    confidence = confidence_response.payload

                    # Phase 6: horizon confidence decay
                    if isinstance(confidence, dict):
                        horizon_factor = max(
                            0.70,
                            0.95 - ((day_num - 1) * 0.05)
                        )

                        if "overall_confidence" in confidence:
                            confidence["overall_confidence"] = round(
                                confidence["overall_confidence"] * horizon_factor,
                                4
                            )

                        forecast_conf = confidence.get(
                            "forecast_confidence"
                        )

                        if isinstance(forecast_conf, dict):
                            if "confidence_score" in forecast_conf:
                                forecast_conf["confidence_score"] = round(
                                    forecast_conf["confidence_score"]
                                    * horizon_factor,
                                    4
                                )

                        for analysis in confidence.get(
                            "analyses",
                            []
                        ):
                            if isinstance(analysis, dict):
                                if "confidence_score" in analysis:
                                    analysis["confidence_score"] = round(
                                        analysis["confidence_score"]
                                        * horizon_factor,
                                        4
                                    )

                        confidence["forecast_horizon_factor"] = round(
                            horizon_factor,
                            4
                        )

                        # Confidence CONTRACT: this value is a deterministic
                        # BUSINESS HEURISTIC (weighted metric sum + horizon
                        # decay), NOT calibrated statistical/model uncertainty.
                        # Explicitly stamp it so downstream consumers and the
                        # GUI never mislabel it as model-derived probability.
                        confidence["confidence_contract"] = {
                            "kind": "heuristic",
                            "basis": "weighted_component_metrics_with_horizon_decay",
                            "calibrated": False,
                            "statistical": False,
                        }

                    confidence_obj = self.confidence_engine.core.evaluate(
                        forecast_result=forecast_payload
                    )

                risk_response = self.risk_engine.execute(
                    ForecastRequest(
                        operation="risk",
                        parameters={
                            "forecast_result": forecast_payload,
                            "confidence_result": confidence_obj
                        }
                    )
                )

                if risk_response.success:
                    risk = risk_response.payload
                else:
                    risk = {
                        "overall_risk": 0.0,
                        "analyses": [],
                        "warnings": [],
                        "errors": [],
                        "status": "GREEN"
                    }


                if pred_result.errors:
                    errors.extend(pred_result.errors)
                    oh_val = None
                    nps_val = None
                else:
                    oh_val = pred_result.operations_health
                    nps_val = pred_result.nps

                # 3. Record forecast day
                forecast_date = (
                    datetime.date.fromisoformat(forecast_start)
                    + datetime.timedelta(days=day_num)
                )
                day = ForecastDay(
                    date=forecast_date.isoformat(),
                    operations_health=oh_val,
                    nps=nps_val,
                    quality=modified_state.quality,
                    competency=modified_state.competency,
                    transfer=modified_state.transfer,
                    release=modified_state.release,
                    attendance=modified_state.attendance,
                    confidence=confidence,
                    risk=risk,
                    notes=f"Scenario: {request.scenario or 'baseline'}",
                    bayesian_score_distribution=getattr(
                        pred_result, "bayesian_score_distribution", None
                    ),
                    score_counts=getattr(pred_result, "score_counts", None),
                )
                timeline.append(day)

                # Phase 4: persist forecast separately from actuals
                self.forecast_history.append({
                    "date": forecast_date.isoformat(),
                    "type": "forecast",
                    "quality": day.quality,
                    "competency": day.competency,
                    "attendance": day.attendance,
                    "release": day.release,
                    "transfer": day.transfer,
                    "operations_health": day.operations_health,
                    "nps": day.nps,
                })

                self.history_store.save({
                    'forecast_history': self.forecast_history,
                    'actual_history': self.actual_history,
                    'forecast_error_history': self.forecast_error_history
                })

                # 4. Evolve state for next iteration
                current_state = self.evolution.evolve(
                    current_state,
                    pred_result
                )

                next_state = current_state.to_dict()
                next_state["_predicted"] = True
                history_buffer.append(next_state)

            except Exception as e:
                errors.append(f"Day {day_num} error: {str(e)}")
                # Append placeholder day
                forecast_date = (
                    datetime.date.fromisoformat(forecast_start)
                    + datetime.timedelta(days=day_num)
                )
                day = ForecastDay(
                    date=forecast_date.isoformat(),
                    operations_health=None,
                    nps=None,
                    quality=current_state.quality,
                    competency=current_state.competency,
                    transfer=current_state.transfer,
                    release=current_state.release,
                    attendance=current_state.attendance,
                    confidence=None,
                    risk=None,
                    notes="Phase 6: prediction failed"
                )
                timeline.append(day)

        end_time = datetime.datetime.now()
        duration_sec = (end_time - start_time).total_seconds()

        result = ForecastResult(
            horizon=horizon,
            scenario=request.scenario or "baseline",
            start_date=forecast_start,
            end_date=(
                datetime.date.fromisoformat(forecast_start)
                + datetime.timedelta(days=horizon)
            ).isoformat(),
            timeline=timeline,
            summary={
                "total_days": horizon,
                "completed_days": len([d for d in timeline if d.operations_health is not None]),
                "execution_duration_sec": duration_sec
            }
        )

        # --- ADIE V3 handoff (canonical decision layer) ---
        # Forecast outputs reach the canonical V3 decision pipeline. Observed
        # inputs come only from the caller-supplied current state; forecast
        # timeline days are model-predicted outputs passed as scenarios.
        # A V3 failure must never break the forecast.
        try:
            v3_decision = self._build_adie_v3_decision(
                result,
                request,
                observed_state,
            )
        except Exception as exc:  # noqa: BLE001 - advisory handoff must not break forecast
            logger.warning("ADIE V3 handoff failed (forecast continues): %s", exc)
            v3_decision = {"status": "error", "error": str(exc)}

        return ForecastResponse(
            success=True,
            operation="forecast",
            engine="ForecastOrchestrator",
            timestamp=datetime.datetime.now().isoformat(),
            warnings=warnings,
            errors=errors,
            metadata={
                "phase": "6",
                "horizon": horizon,
                "duration_sec": duration_sec
            },
            payload={
                  "decision_intelligence": v3_decision,
                  **asdict(result),
                  "validation": (
                      self.backtester.evaluate(
                          self.forecast_error_history
                      )
                      if hasattr(self, "backtester")
                      else {}
                  ),
                  "calibration": (
                      self.calibration.summarize(
                          self.forecast_error_history
                      )
                      if hasattr(self, "calibration")
                      else {}
                  ),
                  "history_status": {
                      "forecast_records": len(
                          self.forecast_history
                      ),
                      "actual_records": len(
                          self.actual_history
                      ),
                      "error_records": len(
                          self.forecast_error_history
                      )
                  }
              }
        )


    def regenerate_forecast(
        self,
        horizon: int = 7
    ):
        """
        Phase 5:
        Rebuild future forecast after actual data update.

        Actual history is the source of truth.
        Forecast history is regenerated from latest actual state.
        """

        if not self.actual_history:
            raise ValueError("No actual history available")

        latest_actual = sorted(
            self.actual_history,
            key=lambda x: x.get("date", "")
        )[-1]

        state = {
            "quality": latest_actual.get("quality", 0),
            "competency": latest_actual.get("competency", 0),
            "attendance": latest_actual.get("attendance", 0),
            "release": latest_actual.get("release", 0),
            "transfer": latest_actual.get("transfer", 0),
            "operations_health": latest_actual.get(
                "operations_health"
            ),
            "nps": latest_actual.get("nps"),
        }

        self.forecast_history = []

        return self.execute(
            ForecastRequest(
                operation="forecast",
                horizon=horizon,
                parameters={
                    "state": state
                }
            )
        )

    def update_actual(self, actual_day: Dict[str, Any]):
        """
        Phase 3.1:
        Replace forecast with actual when real operational data arrives.

        Actual data always wins over forecast.
        """

        actual_date = actual_day.get("date")

        if actual_date is None:
            raise ValueError("Actual day requires date")

        # Phase 7: capture forecast before replacement
        matched_forecast = next(
            (
                f for f in self.forecast_history
                if f.get("date") == actual_date
            ),
            None
        )

        # remove matching forecast
        self.forecast_history = [
            f for f in self.forecast_history
            if f.get("date") != actual_date
        ]

        actual_record = {
            **actual_day,
            "type": "actual"
        }

        # Phase 7: forecast error calibration

        if matched_forecast:
            for metric in [
                "quality",
                "competency",
                "attendance",
                "release",
                "transfer",
                "operations_health",
                "nps",
            ]:
                predicted = matched_forecast.get(metric)
                actual = actual_record.get(metric)

                if predicted is not None and actual is not None:
                    self.forecast_error_history.append({
                        "date": actual_date,
                        "metric": metric,
                        "predicted": predicted,
                        "actual": actual,
                        "error": actual - predicted
                    })

        # append actual history
        self.actual_history.append(actual_record)

        # keep chronological order
        self.actual_history.sort(
            key=lambda x: x.get("date","")
        )


        if hasattr(self, "history_store"):
            self.history_store.save({
                "forecast_history": self.forecast_history,
                "actual_history": self.actual_history,
                "forecast_error_history": self.forecast_error_history
            })

        return actual_record

    def _get_state_from_request(self, request: ForecastRequest) -> Optional[Dict[str, Any]]:
        if request.parameters and 'state' in request.parameters:
            return request.parameters['state']
        return None

    @staticmethod
    def _explicit_cutoff(request, state_dict) -> Optional[str]:
        """Return an explicit forecast cutoff date if the caller supplied one.

        Priority: ``request.parameters["cutoff"]``, then ``state_dict["date"]``,
        then ``state_dict["cutoff"]``.  Returns ``None`` (meaning "use today",
        i.e. normal production forecasting) when no explicit cutoff is given.
        """
        params = request.parameters or {}
        for candidate in (
            params.get("cutoff"),
            (state_dict or {}).get("date"),
            (state_dict or {}).get("cutoff"),
        ):
            if candidate is None:
                continue
            try:
                from datetime import date, datetime
                return date.fromisoformat(str(candidate)).isoformat()
            except Exception:
                try:
                    from datetime import datetime
                    return datetime.fromisoformat(str(candidate)).isoformat()
                except Exception:
                    raise ValueError(
                        f"Invalid forecast cutoff: {candidate!r}. "
                        "Cutoff must be an ISO date/datetime."
                    )
        return None

    def _build_adie_v3_decision(
        self,
        result: ForecastResult,
        request: ForecastRequest,
        observed_state: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build the canonical ADIE V3 decision package from forecast outputs.

        Temporal/provenance safety:
          - ``observed_state`` (the caller-supplied current state at forecast
            start) is the ONLY source of observed inputs. Forecast timeline
            values are model-predicted outputs and are passed as named
            scenarios, never as observed/known-at-cutoff inputs.
          - The real forecast cutoff is the forecast start date (T).
          - Provenance stamps for observed inputs are the cutoff date itself
            (inputs known at or before T); forecast-day scenarios carry an
            explicit ``_predicted`` marker.
          - The same enforced ``ProductionDecisionBoundary`` used by the
            canonical V3 service is invoked here (identical ``validate()``
            then ``ProbabilisticDecisionService.analyze()``).
        """
        import math

        from core.common.temporal_contract import assert_known_at_cutoff

        if not observed_state:
            return {
                "status": "skipped",
                "reason": "no observed current state provided for ADIE V3",
            }

        # 1. Real forecast cutoff = forecast start date (T).
        cutoff = result.start_date or datetime.date.today().isoformat()

        # 2. Observed current-known inputs (normalized to [0,1] for the
        #    Bayesian engine). Only observed KPI fields are used.
        observed_metrics = {
            "operations_health": observed_state.get("operations_health")
            or observed_state.get("operational_health"),
            "nps": observed_state.get("nps"),
            "quality": observed_state.get("quality"),
            "competency": observed_state.get("competency"),
            "attendance": observed_state.get("attendance"),
            "release": observed_state.get("release"),
            "transfer": observed_state.get("transfer"),
        }
        observed_metrics = {
            key: value
            for key, value in observed_metrics.items()
            if value is not None
        }

        if not observed_metrics:
            return {
                "status": "skipped",
                "reason": "no observed metrics available for ADIE V3",
            }

        adapter = UniversalProbabilisticAdapter()

        observations: List[float] = []
        for metric, value in observed_metrics.items():
            try:
                normalized = adapter._normalize(
                    metric,
                    [float(value)],
                )[0]
            except (TypeError, ValueError):
                continue
            if math.isfinite(normalized):
                observations.append(normalized)

        if not observations:
            return {
                "status": "skipped",
                "reason": "no finite observed inputs for ADIE V3",
            }

        # Baseline = normalized observed operations_health (or first metric).
        baseline_value = observed_metrics.get("operations_health")
        if baseline_value is None:
            first_key = next(iter(observed_metrics))
            baseline_value = observed_metrics[first_key]
        try:
            baseline = adapter._normalize(
                "operations_health"
                if "operations_health" in observed_metrics
                else next(iter(observed_metrics)),
                [float(baseline_value)],
            )[0]
        except (TypeError, ValueError):
            baseline = observations[0]

        if not math.isfinite(baseline):
            baseline = observations[0]

        # 3. Forecast timeline days = model-predicted outputs as scenarios.
        #    Each carries an explicit predicted marker; none are observed.
        #    Per-day Forecast AI evidence (confidence, risk, OH deltas, NPS
        #    0..10 posterior distribution) is threaded through so ADIE can
        #    rank scenarios on real information instead of fabricating
        #    identical Bayesian/MC statistics (Phases 1, 2, 9).
        from core.decision_intelligence.v3.bayesian import (
            inference as bayes_utils,
        )

        observed_oh_raw = (
            observed_state.get("operations_health")
            or observed_state.get("operational_health")
        )

        scenarios: List[Dict[str, Any]] = []
        for index, day in enumerate(result.timeline, start=1):
            if day is None:
                continue

            oh = getattr(day, "operations_health", None)
            nps = getattr(day, "nps", None)

            # Per-day Forecast AI confidence / risk severity (when present).
            confidence_score = None
            conf_dict = getattr(day, "confidence", None)
            if isinstance(conf_dict, dict):
                confidence_score = conf_dict.get("overall_confidence")

            risk_severity = None
            risk_dict = getattr(day, "risk", None)
            if isinstance(risk_dict, dict):
                risk_severity = risk_dict.get("overall_risk")
            elif isinstance(risk_dict, list):
                # RiskResult stored as a list of analyses; no scalar severity.
                risk_severity = None

            # Day-over-day OH delta (real forecast arithmetic, not prediction).
            previous_oh = None
            if index > 1:
                previous_day = result.timeline[index - 2]
                if previous_day is not None:
                    previous_oh = getattr(previous_day, "operations_health", None)
            else:
                previous_oh = observed_oh_raw
            delta_oh = None
            if (
                oh is not None
                and previous_oh is not None
                and isinstance(oh, (int, float))
                and isinstance(previous_oh, (int, float))
                and math.isfinite(float(oh))
                and math.isfinite(float(previous_oh))
            ):
                delta_oh = float(oh) - float(previous_oh)

            # NPS 0..10 posterior distribution -> expected 0..10 survey score,
            # 0..10 score quantiles, and Monte Carlo NPS (-100..100) quantiles.
            #
            # Naming contract (unambiguous):
            #   expected_score : mean 0..10 survey score
            #   score_p05/p95  : 0..10 score quantiles
            #   nps_p05/p95    : -100..100 NPS quantiles (Monte Carlo)
            #   nps            : -100..100 NPS point forecast (from the model)
            distribution = (
                getattr(day, "bayesian_score_distribution", None)
                or getattr(day, "nps_distribution", None)
            )
            expected_score = None
            score_p05 = None
            score_p95 = None
            nps_p05 = None
            nps_p95 = None
            if distribution:
                try:
                    # 0..10 mean survey score.
                    expected_score = bayes_utils.expected_nps_from_distribution(distribution)
                    # 0..10 score quantiles.
                    score_p05, score_p95 = bayes_utils.nps_score_percentiles(distribution)
                    # -100..100 NPS quantiles via Monte Carlo.
                    nps_p05, nps_p95 = bayes_utils.nps_monte_carlo_percentiles(distribution)
                except Exception:
                    expected_score, score_p05, score_p95, nps_p05, nps_p95 = None, None, None, None, None
            expected_score = (
                float(expected_score)
                if expected_score is not None and math.isfinite(float(expected_score))
                else None
            )
            score_p05 = float(score_p05) if score_p05 is not None else None
            score_p95 = float(score_p95) if score_p95 is not None else None
            nps_p05 = float(nps_p05) if nps_p05 is not None else None
            nps_p95 = float(nps_p95) if nps_p95 is not None else None

            scenarios.append({
                "name": f"forecast_day_{index}",
                "date": getattr(day, "date", None),
                "_predicted": True,
                "operations_health": oh,
                "nps": nps,
                "quality": getattr(day, "quality", None),
                "competency": getattr(day, "competency", None),
                "attendance": getattr(day, "attendance", None),
                "release": getattr(day, "release", None),
                "transfer": getattr(day, "transfer", None),
                "confidence": confidence_score,
                "risk_severity": risk_severity,
                "delta_oh": delta_oh,
                "expected_score": expected_score,
                "score_p05": score_p05,
                "score_p95": score_p95,
                "nps_p05": nps_p05,
                "nps_p95": nps_p95,
                "bayesian_score_distribution": distribution,
            })


        if not scenarios:
            return {
                "status": "skipped",
                "reason": "forecast timeline empty; no ADIE V3 scenarios",
            }

        # 4. Provenance: observed inputs known at or before the cutoff.
        #    The cutoff is the forecast start date; the observed inputs are
        #    the caller's current state known as of that date.
        cutoff_ts = f"{cutoff}T00:00:00+00:00"
        provenance = [cutoff_ts] * len(observations)

        # Verify the provenance is temporally valid before handing off.
        for stamp in provenance:
            assert_known_at_cutoff(stamp, cutoff_ts, field_name="observed_input")

        metadata = {
            "provenance": provenance,
            "is_predicted": False,
            "treated_as_observed": True,
        }

        # 5. Invoke the same enforced boundary used by the canonical V3
        #    service (ProductionDecisionBoundary.validate + service.analyze).
        boundary = ProductionDecisionBoundary(
            decision_service=ProbabilisticDecisionService(),
        )
        boundary.validate(
            observations=observations,
            baseline=baseline,
            scenarios=scenarios,
            cutoff=cutoff_ts,
            metadata=metadata,
        )

        # 5. Build Forecast AI producer outputs (advisory, after forecast
        #    generation). Trend + sensitivity are always attempted; the
        #    recommendation/strategy sections are explicit (success or
        #    skipped-with-reason) and never fabricated.
        trend_output = self._build_trend_output(result)
        sensitivity_output = self._build_sensitivity_output(observed_state)
        recommendation_output = self._build_recommendation_output(request)
        strategy_output = self._build_strategy_output(recommendation_output)
        agreement = self._compute_agreement(recommendation_output)

        # 6. Targets (when supplied) drive per-metric target probability and
        #    KPI-specific recommendations. Never fabricated.
        params = request.parameters or {}
        targets = {}
        if params.get("target_oh") is not None:
            targets["target_oh"] = float(params["target_oh"])
        if params.get("target_nps") is not None:
            targets["target_nps"] = float(params["target_nps"])

        observed_baseline_oh = observed_metrics.get("operations_health")

        package = boundary.decision_service.analyze(
            scenarios=scenarios,
            observations=observations,
            baseline=baseline,
            targets=targets or None,
            sensitivity_output=sensitivity_output,
            observed=observed_baseline_oh,
            observed_metrics=sorted(observed_metrics.keys()),
            horizon=result.horizon,
        )

        # 7. Fold Forecast AI outputs into ONE canonical decision payload.
        return {
            "status": "success",
            "cutoff": cutoff_ts,
            "provenance": provenance,
            "observed_metrics": sorted(observed_metrics.keys()),
            "package": compose_decision_package(
                ProbabilisticDecisionService.to_dict(package),
                recommendation_output=recommendation_output,
                strategy_output=strategy_output,
                trend_output=trend_output,
                sensitivity_output=sensitivity_output,
                agreement=agreement,
                targets=targets or None,
                observed=observed_baseline_oh,
                observed_metrics=sorted(observed_metrics.keys()),
                horizon=result.horizon,
            ),
        }

    def _build_sensitivity_output(
        self,
        observed_state: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run the canonical Forecast AI SensitivityEngine over the observed
        state AFTER forecast generation.

        Sensitivity uses only the observed (non-predicted) state as input and
        never feeds back into prediction inputs. Its result is advisory and
        exposed to ADIE V3. On any failure it returns an empty dict (no
        fabricated output).
        """
        try:
            from ..sensitivity import SensitivityEngine

            if not observed_state:
                return {}

            state = dict(observed_state)
            # Drop non-KPI keys that the sensitivity generator does not expect.
            for key in ("_predicted", "date", "history_buffer", "metadata"):
                state.pop(key, None)

            result = SensitivityEngine().analyze(state)
            if not result.success:
                return {}

            from dataclasses import asdict

            return {
                "success": result.success,
                "analyses": [asdict(a) for a in result.analyses],
                "ranking": [asdict(r) for r in result.ranking],
                "warnings": result.warnings,
                "errors": result.errors,
                "metadata": result.metadata,
            }
        except Exception as exc:  # noqa: BLE001 - advisory, never break forecast
            logger.warning("Sensitivity analysis failed (forecast continues): %s", exc)
            return {}

    def _build_trend_output(
        self,
        result: ForecastResult,
    ) -> Dict[str, Any]:
        """Run the existing Forecast AI trend analysis over the forecast
        timeline (each KPI as a series). Returns its payload, or an empty
        dict when the timeline has no usable series."""
        try:
            from ..trends import TrendEngine, TrendSeries

            timeline = result.timeline or []
            if not timeline:
                return {}

            series_list: List[TrendSeries] = []
            for metric in (
                "operations_health",
                "nps",
                "quality",
                "competency",
                "attendance",
                "release",
                "transfer",
            ):
                values = [
                    getattr(day, metric, None)
                    for day in timeline
                    if getattr(day, metric, None) is not None
                ]
                timestamps = [
                    getattr(day, "date", None)
                    for day in timeline
                    if getattr(day, metric, None) is not None
                ]
                if values and timestamps:
                    series_list.append(TrendSeries(
                        metric=metric,
                        values=values,
                        timestamps=timestamps,
                    ))

            if not series_list:
                return {}

            trend_result = TrendEngine().analyze(series_list)
            return {
                "success": trend_result.success,
                "analyses": [
                    __import__("dataclasses").asdict(a)
                    for a in trend_result.analyses
                ],
                "warnings": trend_result.warnings,
                "errors": trend_result.errors,
                "metadata": trend_result.metadata,
            }
        except Exception as exc:  # noqa: BLE001 - advisory, never break forecast
            logger.warning("Trend analysis failed (forecast continues): %s", exc)
            return {}

    def _build_recommendation_output(
        self,
        request: ForecastRequest,
    ) -> Dict[str, Any]:
        """Run the Forecast AI recommendation engine ONLY when the request
        carries target_oh/target_nps and a state.

        On any failure an explicit ``skipped`` block with a specific reason is
        returned (never a silently-missing section, never fabricated recs).
        The optimizer reuses the forecast's own PredictionService so the same,
        already-loaded models are used and no duplicate model loading occurs.
        """
        try:
            from .recommendation_engine import RecommendationEngine
            from ..models import ForecastRequest as FR
            from ..optimization import ReverseOptimizer

            params = request.parameters or {}
            target_oh = params.get("target_oh")
            target_nps = params.get("target_nps")
            state = params.get("state")

            if target_oh is None and target_nps is None:
                return {
                    "status": "skipped",
                    "reason": "missing_target",
                    "recommendations": [],
                }
            if state is None:
                return {
                    "status": "skipped",
                    "reason": "missing_target",
                    "recommendations": [],
                }

            # Bound the advisory optimizer (timeout enforced in Phase 5).
            rec_params = dict(params)
            rec_params.setdefault("max_iterations", 25)
            rec_params.setdefault("timeout_seconds", 10)

            rec_req = FR(
                operation="recommend",
                parameters=rec_params,
            )
            engine = RecommendationEngine(
                optimizer_core=ReverseOptimizer(
                    prediction_service=self.service,
                )
            )
            response = engine.execute(rec_req)
            if not response.success:
                return {
                    "status": "skipped",
                    "reason": self._recommendation_failure_reason(response),
                    "success": False,
                    "recommendations": [],
                    "diagnostics": self._recommendation_diagnostics(
                        evaluated=True, reasons=[self._recommendation_failure_reason(response)],
                    ),
                }
            normalized = self._normalize_recommendation_payload(response.payload)
            if normalized is not None:
                return normalized
            return {
                "status": "skipped",
                "reason": "optimization_failed",
                "success": False,
                "recommendations": [],
                "diagnostics": self._recommendation_diagnostics(evaluated=True, reasons=["optimization_failed"]),
            }
        except Exception as exc:  # noqa: BLE001 - advisory, never break forecast
            logger.warning("Recommendation engine failed (forecast continues): %s", exc)
            return {
                "status": "skipped",
                "reason": "optimization_failed",
                "success": False,
                "recommendations": [],
                "diagnostics": self._recommendation_diagnostics(
                    evaluated=True,
                    reasons=[f"optimization_failed: {exc}"],
                ),
            }

    @staticmethod
    def _recommendation_diagnostics(
        evaluated: bool = False,
        reasons: list[str] | None = None,
        rules_considered: int = 0,
        rules_matched: int = 0,
        rules_rejected: list[str] | None = None,
        evidence_count: int = 0,
        final_count: int = 0,
    ) -> Dict[str, Any]:
        """Build the recommendation-path diagnostics surface.

        Tracks which engines/rules were considered, matched, or rejected and
        the resulting evidence/final counts. Purely descriptive — never alters
        the recommendation evidence itself.
        """
        return {
            "engines_evaluated": evaluated,
            "rules_considered": rules_considered,
            "rules_matched": rules_matched,
            "rules_rejected": list(rules_rejected or []),
            "reasons": list(reasons or []),
            "evidence_count": evidence_count,
            "final_recommendation_count": final_count,
        }

    @staticmethod
    def _normalize_recommendation_payload(
        payload: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Normalize the Forecast AI recommendation engine payload into the
        canonical flat shape consumed by ADIE decision_detail and agreement.

        The engine returns a NESTED block::

            {"optimization": {...},
             "recommendations": {"success": bool,
                                 "recommendations": [ {...rec...} ],
                                 "warnings": [...], "errors": [...],
                                 "metadata": {...}}}

        ADIE's evidence gate and detail builder (decision_evidence_sufficient,
        _build_top_recommendations) and the agreement/strategy consumers expect
        a FLAT shape::

            {"status": "success", "success": bool,
             "recommendations": [ {...rec...} ],   # flat list
             "warnings": [...], "errors": [...], "metadata": {...}}

        This normalizer flattens the nested list into the top-level
        ``recommendations`` key, preserving all genuine evidence. Returns None
        when there is no usable payload.
        """
        if not payload:
            return None
        rec_block = payload.get("recommendations") or {}
        recs = rec_block.get("recommendations") if isinstance(rec_block, dict) else None
        if not isinstance(recs, list):
            recs = payload.get("recommendations") if isinstance(payload.get("recommendations"), list) else []
        success = bool(rec_block.get("success")) if isinstance(rec_block, dict) else bool(recs)

        from ..recommendations import Priority as RecPriority

        # Normalize each recommendation to the canonical flat dict (resolve enum
        # values so downstream JSON/policy consumers never see Enum members).
        flat_recs = []
        for item in recs:
            if not isinstance(item, dict):
                continue
            norm = dict(item)
            prio = norm.get("priority")
            if hasattr(prio, "value"):
                norm["priority"] = prio.value
            cat = norm.get("category")
            if hasattr(cat, "value"):
                norm["category"] = cat.value
            diff = norm.get("difficulty")
            if hasattr(diff, "value"):
                norm["difficulty"] = diff.value
            flat_recs.append(norm)

        status = "success" if success else "skipped"
        warnings = rec_block.get("warnings") if isinstance(rec_block, dict) else []
        errors = rec_block.get("errors") if isinstance(rec_block, dict) else []
        metadata = rec_block.get("metadata") if isinstance(rec_block, dict) else {}

        diag = ForecastOrchestrator._recommendation_diagnostics(
            evaluated=True,
            rules_considered=5,  # the 5 operational KPI templates (quality/competency/attendance/transfer/release)
            rules_matched=len(flat_recs),
            rules_rejected=[],  # per-field rejections are recorded in engine metadata
            evidence_count=len(flat_recs),
            final_count=len(flat_recs),
        )

        return {
            "status": status,
            "success": success,
            "recommendations": flat_recs,
            "evidence_count": len(flat_recs),
            "final_recommendation_count": len(flat_recs),
            "warnings": warnings,
            "errors": errors,
            "metadata": metadata,
            "diagnostics": diag,
        }


    @staticmethod
    def _recommendation_failure_reason(response: Any) -> str:
        """Map a failed recommendation response to a specific reason."""
        metadata = getattr(response, "metadata", None) or {}
        reason = metadata.get("reason")
        if reason in {
            "optimization_timeout",
            "optimization_failed",
            "insufficient_feasible_actions",
            "missing_target",
            "invalid_input",
        }:
            return reason
        errors = getattr(response, "errors", None) or []
        text = " ".join(errors).lower()
        if any(k in text for k in ("timeout", "timed out")):
            return "optimization_timeout"
        if any(
            k in text
            for k in ("no solution within tolerance", "cannot generate recommendations")
        ):
            return "insufficient_feasible_actions"
        return "optimization_failed"

    @staticmethod
    def _compute_agreement(
        recommendation_output: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Conflict-aware recommendation agreement (Phase 10 / W6).

        Computed ONLY from real produced recommendations; returns None when no
        recommendations were generated (e.g. a skipped block). Never alters
        the forecast itself — it is advisory evidence for the ADIE decision.
        """
        if not recommendation_output:
            return None
        if recommendation_output.get("status") == "skipped":
            return None
        from core.decision_intelligence.v3.synthesis.decision_detail import (
            canonical_recommendation_list,
        )
        rec_data = canonical_recommendation_list(recommendation_output)
        if not rec_data or not recommendation_output.get("success", True):
            return None

        try:
            from ..recommendations import (
                Recommendation as Rec,
                Category as RecCategory,
                Priority as RecPriority,
                Difficulty as RecDifficulty,
            )
            from ..recommendations.conflicts import ConflictDetector
            from ..confidence.metrics import ConfidenceMetrics

            def _to_rec(item: Dict[str, Any]) -> Rec:
                cat = item.get("category")
                pri = item.get("priority")
                diff = item.get("difficulty")
                try:
                    cat_enum = RecCategory(cat) if not isinstance(cat, RecCategory) else cat
                except Exception:
                    cat_enum = RecCategory.GENERAL
                try:
                    pri_enum = RecPriority(pri) if not isinstance(pri, RecPriority) else pri
                except Exception:
                    pri_enum = RecPriority.MEDIUM
                try:
                    diff_enum = RecDifficulty(diff) if not isinstance(diff, RecDifficulty) else diff
                except Exception:
                    diff_enum = RecDifficulty.MEDIUM
                return Rec(
                    id=str(item.get("id", "")),
                    title=str(item.get("title", "")),
                    description=str(item.get("description", "")),
                    category=cat_enum,
                    priority=pri_enum,
                    difficulty=diff_enum,
                    confidence=item.get("confidence", 0.5),
                    actions=list(item.get("actions", []) or []),
                    reasoning=str(item.get("reasoning", "")),
                    metadata=dict(item.get("metadata", {}) or {}),
                    target_kpi=item.get("target_kpi"),
                    direction=item.get("direction"),
                    magnitude=item.get("magnitude"),
                )

            rec_objs = [
                _to_rec(r)
                for r in rec_data
                if isinstance(r, dict)
            ]
            if not rec_objs:
                return None

            detector = ConflictDetector()
            conflicts = detector.detect_conflicts(rec_objs)
            conflicts_text = [c[2] for c in conflicts]

            metric = ConfidenceMetrics()
            score = metric.recommendation_agreement(rec_objs)

            categories = {}
            for rec in rec_objs:
                cat = rec.category.value
                categories[cat] = categories.get(cat, 0) + 1
            if len(categories) <= 1:
                category_consistency = 1.0
            else:
                top = max(categories.values())
                total = len(rec_objs)
                penalty = max(0.0, 1.0 - (len(categories) - 1) * 0.1)
                category_consistency = max(
                    0.0, (top / total if total else 0.0) * max(0.5, penalty)
                )

            return {
                "score": round(float(score), 4),
                "category_consistency": round(float(category_consistency), 4),
                "conflicts": conflicts_text,
                "conflict_count": len(conflicts_text),
            }
        except Exception as exc:  # noqa: BLE001 - advisory
            logger.warning("Agreement computation failed (forecast continues): %s", exc)
            return None

    def _build_strategy_output(
        self,
        recommendation_output: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Run the Forecast AI strategy engine on an existing recommendation
        result. Returns an explicit ``skipped`` block (with reason) when the
        strategy cannot be generated; never fabricates strategies."""
        try:
            from .strategy_engine import StrategyEngine
            from ..models import ForecastRequest as FR

            if not recommendation_output:
                return {
                    "status": "skipped",
                    "reason": "missing_recommendation_result",
                    "strategies": [],
                }

            if recommendation_output.get("status") == "skipped":
                return {
                    "status": "skipped",
                    "reason": recommendation_output.get(
                        "reason", "missing_recommendation_result"
                    ),
                    "strategies": [],
                }

            from core.decision_intelligence.v3.synthesis.decision_detail import (
                canonical_recommendation_list,
            )
            rec_data = canonical_recommendation_list(recommendation_output)
            if not rec_data or not recommendation_output.get("success", True):
                return {
                    "status": "skipped",
                    "reason": "missing_recommendation_result",
                    "strategies": [],
                }

            strat_req = FR(
                operation="strategy",
                parameters={
                    "recommendation_result": {
                        "success": recommendation_output.get("success", True),
                        "recommendations": rec_data,
                    }
                },
            )
            response = StrategyEngine().execute(strat_req)
            if not response.success:
                return {
                    "status": "skipped",
                    "reason": "strategy_generation_failed",
                    "strategies": [],
                }
            return response.payload
        except Exception as exc:  # noqa: BLE001 - advisory, never break forecast
            logger.warning("Strategy engine failed (forecast continues): %s", exc)
            return {
                "status": "skipped",
                "reason": "strategy_generation_failed",
                "strategies": [],
            }

    def _error_response(self, message: str) -> ForecastResponse:
        return ForecastResponse(
            success=False,
            operation="forecast",
            engine="ForecastOrchestrator",
            timestamp=datetime.datetime.now().isoformat(),
            warnings=[],
            errors=[message],
            metadata={},
            payload=None
        )

# Module-level compatibility surface.
# Delegates to the existing implementation; no logic changed.
execute = ForecastOrchestrator.execute

# Module-level compatibility surface.
# Delegates to the existing implementation; no logic changed.
regenerate_forecast = ForecastOrchestrator.regenerate_forecast

# Module-level compatibility surface.
# Delegates to the existing implementation; no logic changed.
update_actual = ForecastOrchestrator.update_actual
