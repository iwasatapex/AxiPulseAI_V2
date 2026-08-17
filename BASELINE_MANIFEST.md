# AxiPulseAI — Known-Good Baseline Manifest

**Locked at**: 2026-08-11 (14:44 UTC)
**Status**: BASELINE LOCKED — READY FOR NEXT FEATURE

This manifest records the verified, locked production baseline. Future agents
must verify against this before/after any change. If any item deviates, the
baseline is broken and must be investigated before proceeding.

---

## 1. Locked Runtime Architecture / Call Order (DO NOT CHANGE)

```
trained OH/NPS model artifacts
  -> PredictionService
  -> ForecastOrchestrator
     -> ScenarioManager
     -> ConfidenceEngine
     -> RiskEngine
     -> StateEvolutionEngine (recursive, _predicted=True on days >= 2)
  -> SensitivityEngine (BATCHED: 1 predict_batch call = 1 NPS model call for
     all experiments + fast per-row OH predict; verified numerically equivalent)
  -> ADIE V3
     -> ProductionDecisionBoundary.validate (cutoff + provenance)
     -> ProbabilisticDecisionService (Bayesian + exactly ONE Monte Carlo:
        core.monte_carlo)
     -> TrendEngine.analyze
     -> RecommendationEngine (only when request has target_oh/target_nps)
     -> StrategyEngine (only when recommendations succeed)
     -> compose_decision_package
  -> canonical payload["decision_intelligence"]
```

Invariants:
- ADIE V3 is advisor-only: it consumes Forecast AI outputs, never feeds back
  into predictors. Zero `.predict()` calls in `core/decision_intelligence/v3`.
- Observed state captured pre-loop; recursive forecast days are `_predicted=True`;
  predicted OH is never treated as observed (`_known_oh_at_cutoff` guard).
- Exactly ONE Monte Carlo: `core/monte_carlo`.
- ADIE V2 removed: no `decision_intelligence_v3` duplicate, no `/api/v1/adie/decision`;
  only `/api/v1/adie/v3/decision`.
- OH interference fix active: NPS rule component health-neutral (health_component=10.0);
  NPS None handling fixed (postprocess None-safe).

## 2. Trained Model Artifacts (source of truth — DO NOT MODIFY/RETRAIN)

| Artifact | Estimator | model_name | Features | History | Engine |
|----------|-----------|------------|---------:|--------:|--------|
| models/operation_health_predictor.joblib | CatBoostRegressor | CatBoost | 19 | 1289 | v10.10 |
| models/nps_predictor_model.pkl | MultiOutputRegressor(RandomForestRegressor) | RandomForest | 33 | 1289 | v2.1.0 |

Hashes (SHA-256):
- OH: fe36bdc0c98979a915c95c08e17bbd89b968b25f256bda1de797d213c7044e4e
- NPS: ed7cce81d6b46a4c1cfd7c1c8bae91fd9afc68c6e5c9db6f33bf09c3b40e50b3

Training metadata:
- OH: trained_at 2026-08-11T13:37:12, target_mean 91.96; algos: CatBoost(sel),
  LightGBM, XGBoost, ExtraTrees, RandomForest, HistGradientBoosting, GradientBoosting, MLP
- NPS: predict_mode=distribution, num_scores=11; algos: ExtraTrees,
  RandomForest(sel), HistGradientBoosting, GradientBoosting, MLP

## 3. Key Production Files (current known-good modifications)

| File | Known-good modification |
|------|-------------------------|
| core/nps_predictor/inference.py | P1 None-safe OH normalization (2 spots); OH-neutral rule component |
| core/forecast_ai/prediction/service.py | Added predict_batch(states) (batches NPS model predict; OH per-row) |
| core/forecast_ai/sensitivity/engine.py | analyze() uses predict_batch with per-state fallback |
| core/forecast_ai/engines/forecast_orchestrator.py | V3 canonical decision_intelligence; _build_sensitivity/trend/recommendation/strategy_output; V2 removed |
| core/forecast_ai/engines/strategy_engine.py | Rebuilds recommendation dicts into Recommendation objects (enum category/priority/difficulty) |
| core/nps_predictor/persistence.py | Original (n_jobs fix was reverted — NOT applied) |
| core/decision_intelligence/__init__.py | Exports V3 only; V2 modules/package deleted |
| api/main.py | Only /api/v1/adie/v3/decision ADIE route |

## 4. Performance Baseline (production venv, real artifacts)

| Run | Time |
|-----|-----:|
| Horizon 1 | ~5.5s |
| Horizon 5 | ~4.9–5.8s |

(Sensitivity batching reduced H5 from ~13s to ~5s.)

## 5. Test Commands & Expected Counts

```bash
cd /home/amteur/Documents/AxiPulseAI
PYTHONPATH=. venv/bin/python -m pytest \
  tests/project/core/forecast_ai/prediction/ \
  tests/project/core/forecast_ai/engines/ \
  tests/project/core/nps_predictor/ \
  tests/qa/adie_v3/ \
  tests/project/api/routes/ \
  tests/project/api/services/ \
  tests/temporal/ \
  tests/project/core/decision_intelligence/ \
  -q --tb=short
```

Expected ~120 passed, 0 failed. Temporal: 19. NPS: 16 (incl. OH-interference
regression). Forecast prediction+engines incl. predict_batch/handoff regression.
Known pre-existing (NOT collected): scenarios/sensitivity surface tests under
tests/project/core/forecast_ai/ excluded via pytest.ini norecursedirs.

## 6. Static Verification Gates

- Zero negative shift(-N) in core/nps_predictor + core/forecast_ai.
- Exactly one Monte Carlo directory: core/monte_carlo.
- App import: python -c "import api.main" -> APP_IMPORT_OK.
- Zero V2 references: no decision_intelligence import ADIE, no
  decision_intelligence.v2, no adie_routes, no build_decision_package.

## 7. Backup Snapshot

Full known-good backup:
/home/amteur/Documents/AxiPulseAI_BACKUPS/baseline_locked_20260811_144430/
(core/nps_predictor, core/forecast_ai, core/probabilistic,
core/decision_intelligence, api, tools).

Earlier per-fix backups (pre_x_<timestamp>) document each individual change.

## 8. Verified Smoke Checks (14/14 PASS)

1. H1, 2. H5, 3. all 5 days OH/NPS finite non-None, 4. sensitivity 5/5,
5. canonical decision_intelligence only, 6. one Monte Carlo, 7. target->recs+strategies,
8. stressed scenario, 9. _predicted markers, 10. cutoff+provenance,
11. no None/NaN/INF, 12. zero negative shifts, 13. app import, 14. runtime improved.

---

**Verification procedure**: load both model artifacts and compare SHA-256; run
the test command in §5 (expect ~120 passed); run one H5 forecast (expect ~5-6s,
all days finite); confirm static gates in §6. If any deviates, stop and report
the discrepancy — do not proceed with new features until baseline restored.
