# AxiPulseAI V2.3 — ForecastAI Production Architecture

> **Specification vs Implementation.** The *V2.3 specification* describes
> desired operational behavior and validation constraints. *ForecastAI*
> implements the production forecasting behavior using **trained ML models**
> rather than a deterministic rule-based simulator. The historical V2.3 rule
> table (OH weights, events, complexity, call-volume multipliers, survey
> chain, etc.) is **not** an executable formula layer in the production
> engine; where those rules appear it is as **learned model behavior** or as
> GUI/presentation semantics — never as a separate rule-based simulator.

## 1. Production architecture

```
GUI / CLI
   │
   ▼
ForecastOrchestrator (core/forecast_ai/engines/forecast_orchestrator.py)
   │  per forecast day:
   ▼
ScenarioManager (core/forecast_ai/scenarios/manager.py)
   │  applies enabled scenario modifiers to OperationalState KPIs
   ▼
PredictionService (core/forecast_ai/prediction/service.py)
   │  builds OH feature row + NPS feature row
   ▼
OH / NPS ML predictors (core/operation_health_predictor, core/nps_predictor)
   │  trained models produce raw OH and NPS score distribution
   ▼
KPITransition (core/forecast_ai/state/transition.py)
   │  momentum (0.6) + hard-bound clamps on the 5 KPIs
   ▼
recursive forecast state (StateEvolutionEngine)
```

## 2. ML prediction flow

- Each forecast day calls `PredictionService.predict`, which constructs an
  **OH feature row** (`_build_oh_row`) and an **NPS feature row**
  (`_build_nps_row`) from the (scenario-modified) operational state.
- The OH row is scored by the trained `OperationalHealthPredictor`
  (CatBoost ensemble, model version 10.10). OH is returned **as-is without
  clipping** (raw excursions allowed).
- The NPS row is scored by the trained `NPSPredictor` (multi-output model on
  the 0–10 score distribution). NPS is derived from the predicted counts via
  `nps_from_score_counts` (`core/nps_predictor/bayesian_distribution.py`):
  `NPS = (promoters − detractors) / total × 100`, with promoters = scores
  9–10 and detractors = scores 0–6. This is the **explicit extraction
  formula**; the counts themselves are ML predictions.
- The NPS feature row deliberately uses the OH value **known at cutoff T**
  (`_known_oh_at_cutoff`), never a same-step T+1 OH forecast (no same-step
  circularity).

## 3. Feature inputs to the OH model

Built in `PredictionService._build_oh_row` + `core/operation_health_predictor/feature_engineering.py`:

- targets: `target_quality=87`, `target_competency=93`, `target_attendance=90`,
  `target_release_rate=60`, `target_transfer_rate=9`
- actuals: `actual_quality`, `actual_competency`, `actual_attendance`,
  `actual_release_rate`, `actual_transfer_rate`
- `total_calls_received` (default 2000), `total_release_calls`
- `operational_intelligence_factor` (0 in production rows)
- prior-day features: `quality/competency/release/transfer/attendance_previous_day`,
  `operations_health_previous_day`
- engineered: `release_score`, `transfer_score` (inverse), per-KPI `_gap`,
  cyclical date (`day_of_week_sin/cos`, `month_sin/cos`, `quarter`,
  `is_weekend`), optional `_lag1/_roll3/_roll7`, `issue_type_*` one-hots
- `date`

The OH **value** is the trained model's output — there is no explicit
`0.5·release + 0.15·transfer + 0.15·competency + 0.15·quality + 0.05·volume`
formula in the production engine.

## 4. Feature inputs to the NPS model

Built in `PredictionService._build_nps_row` + `core/nps_predictor/feature_engineering.py`:

- `operational_health` (prior-day OH known at cutoff T — **OH is an input to
  NPS**)
- `nps_previous_day` (**NPS uses prior-day NPS as an input**)
- `quality`, `competency`, `attendance`, `transfer`,
  `actual_release_rate`, `actual_transfer_rate`, `target_*` (87/93/90/9/60)
- `total_calls_received`, `total_release_calls`
- `total_surveys` (heuristic: released-calls × 0.10), `survey_rate`,
  `survey_confidence`
- prior-day: `quality/competency/release/transfer/attendance_previous_day`
- `business/member/operational_intelligence_factor` (0 in production rows)
- `release_gap`, `release_delta`, `quality_gap`, `competency_gap`,
  `attendance_gap`, `transfer_gap`
- `date` + cyclical date features
- optional provenance-guarded external factors: `seasonal_factor`,
  `weekday_factor`, `flu_factor`, `enrollment_factor`, `holiday_factor`,
  `random_factor` (each requires a `{factor}_known_at` timestamp)

## 5. KPITransition role

`core/forecast_ai/state/transition.py::KPITransition` applies momentum and
hard bounds to the five KPI state fields each day:

- `_move`: `0.6 × current + 0.4 × target` (autocorrelation 0.6)
- clamps: quality 60–100, competency 55–100, attendance 65–100,
  release 50–100, transfer 0–20

It does **not** modify OH or NPS (those are model outputs). Its effect feeds
the evolved state used by the next forecast day.

## 6. Scenario role

`core/forecast_ai/scenarios/{registry,manager,modifiers}.py`: enabled
scenarios apply `ADD`/`SET`/`MULTIPLY` modifiers to `OperationalState` KPI
fields (competency, quality, attendance, release, transfer) **before**
prediction. Disabled scenarios are never executed. OH/NPS are not modified
directly by scenarios.

## 7. Recursive forecasting behavior

`ForecastOrchestrator.execute` runs `horizon` days. Day 1 uses the supplied
observed state; subsequent days use the evolved **predicted** state (marked
`_predicted`) so predicted OH is never consumed as observed prior-period
input. The timeline stores each day's model OH/NPS and the scenario-modified
KPI values.

## 8. What V2.3 rules are explicit

These are explicitly coded (not learned):

- KPI hard bounds (rule 29) — `KPITransition`.
- Momentum coefficient 0.6 (rule 23) — `KPITransition`.
- No same-step OH circularity (rule 3) — `_known_oh_at_cutoff` + `_predicted`.
- NPS from survey counts (rule 26) and NPS ∈ [−100, 100] (rule 27) —
  `nps_from_score_counts`.
- KPI-met 95%/105% thresholds + ≥3-of-4 (rule 30) — `gui/contracts.kpi_met`
  and `gui/analytics/common.day_kpi_met` (presentation/analytics contract).
- OH does not add a direct point term to NPS (rule 28) — no additive formula;
  prior-day OH is a model feature (learned dependence).

## 9. What behavior is learned by ML

Because OH and NPS are trained models, the following are **learned**
relationships, not explicit formulas:

- OH weights / contribution (the 50/15/15/15/5 intuition) — the engineered
  `release_score` / `transfer_score` features feed the model, but the value is
  the model output.
- Survey→NPS probabilities — the NPS model outputs the score distribution.
- Call-volume, season, weekday, complexity, and handle-time effects that
  correlate with the model's output (via `total_calls_received`, cyclical
  date, and optional external factors) are learned, not simulated.
- Intelligence-factor effects (features are present but set to 0 in the
  production rows).

## 10. What V2.3 rules are historical / specification concepts

These are **not executable** in the production engine (they describe the
intended operational model, not the ML implementation):

- transfer → remaining calls → post-transfer release causal arithmetic
- complexity distribution sampling and complexity release/handle-time factors
- event sampling, the event effects table, and event precedence
- weekday call multipliers and final-7-day 50% reduction
- season multipliers
- experience factors
- rule-based survey sampling chain after release

Do not assume the ML engine implements these merely because a feature
correlates with them.

## 11. Known limitations

- Release is treated as a percentage of **total** calls in feature
  construction (`total_release_calls = calls × release / 100`); the spec's
  "remaining calls after transfer" semantics is not modeled.
- No event or complexity sampling exists in the prediction path.
- Call volume is a static/default feature (2000), not generated by weekday/
  season/final-week rules.
- The GUI labels OH/NPS as **ML-predicted** model outputs; any "formula" or
  "causal" language in the UI describes the intended/learned behavior, not an
  executable rule layer.
