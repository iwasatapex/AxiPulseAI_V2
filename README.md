# AxiPulseAI V2

Production-grade AI forecasting and decision platform for contact-center operations.

**Release:** `v2.0.0`  
**Status:** `READY`  
**Production Runtime:** Python `3.13.15`  
**Production Source Family:** `100k-10yr`

> Provenance note: the active canonical production generation
> (`models/production_generations/current`) is `100k-10yr`. Its manifest
> records a training-environment `python_version` of `3.14.4`; the models are
> validated to load and predict under the declared Python `3.13` runtime. CI
> enforces this provenance fail-closed against the active generation.

---

## Overview

AxiPulseAI V2 is an AI-driven operational forecasting and decision system built around a staged causal architecture.

Core flow:

```text
Call Volume
    ↓
Base Operational State
    ↓
Operations Health (OH)
    ↓
Competency / Quality / Complexity / Transfer / Release
    ↓
Handle Time / Survey Conversion / Sentiment Probabilities
    ↓
Transfer
    ↓
Remaining Calls
    ↓
Post-Transfer Release
    ↓
Released Calls
    ↓
Surveys
    ↓
0–10 Survey Scores
    ↓
Bayesian / Monte Carlo Survey-Score Analysis
    ↓
NPS
    ↓
Forecast
    ↓
Recommendation
    ↓
Evidence
    ↓
Risk
    ↓
Decision
    ↓
GUI
```

The architecture prevents downstream metrics from being used to improperly force upstream state.

---

## Release Status

```text
Version: v2.0.0
Status: READY
Runtime: Python 3.13.15
Source: 100k-10yr
```

Final production verification:

```text
Full pytest suite: 722 passed, 0 skipped
OH production inference: 93.03606462275141
NPS production inference: 84.0
NPS outputs: 11
NPS features: 34
OH features: 19
```

---

## Core Architecture

### No same-step OH circularity

OH is calculated from a BASE operational state and then applied downstream.

```text
BASE KPI state
      ↓
     OH
      ↓
OH-adjusted downstream metrics
```

OH must not be recomputed from KPI values it has already influenced within the same step.

### OH does not directly modify NPS

OH influences operational behavior upstream of survey generation. It is not a direct NPS score modifier.

### Transfer occurs once

Transfer is calculated before release. Transferred calls are removed from the remaining-call population. Post-transfer release is calculated only against remaining calls.

### NPS is outcome-based

NPS is calculated from promoter/passive/detractor survey outcomes rather than being directly forced from KPI values.

---

## KPI Contracts

| KPI | Target | Absolute Range |
|---|---:|---:|
| Quality | 87% | 60–100 |
| Competency | 93% | 55–100 |
| Attendance | 90% | 65–100 |
| Release | 60% | 50–100 |
| Transfer | 9% | 0–20 |

Hard constraints:

- Release target must never be below 50%.
- Transfer target must never exceed 20%.
- Quality target must never be below 60%.

KPI-met threshold:

- Quality/Competency/Release: actual >= 95% of target.
- Transfer: actual <= 105% of target.
- Overall KPI-met requires at least 3 of 4 checked KPIs.

---
## Probabilistic Survey-Score Uncertainty

Production NPS uncertainty is applied at the **individual survey-score level**, not to an already-computed scalar NPS.

The canonical production flow is:

```text
ML model
  ↓
0–10 survey-score probability distribution
  ↓
Bayesian posterior on the 0–10 survey-score distribution
  ↓
Monte Carlo / probabilistic analysis on survey scores
  ↓
integer survey-score counts
  ↓
Promoters / Passives / Detractors
  ↓
 NPS
 ```

## NPS Contract

Production NPS uses an 11-output model representing scores `0` through `10`.

```text
Detractors: 0–6
Passives:   7–8
Promoters:  9–10
```

Formula:

```text
NPS = (Promoters - Detractors) / Total Surveys * 100
```

Theoretical range:

```text
-100 to +100
```

Production verification:

```text
Features: 34
Outputs: 11
NPS: 84.0
```

---

## Production Models

### Canonical OH

```text
Path:       models/production_OH.pkl
Algorithm:  CatBoost
Features:   19
Outputs:    1
Trained:    True
Python:     3.13.15
Source:     100k-10yr
Role:       production
```

Production inference:

```text
93.03606462275141
```

### Canonical NPS

```text
Path:       models/production_NPS.pkl
Algorithm:  XGBoost
Features:   34
Outputs:    11
Trained:    True
Python:     3.13.15
Source:     100k-10yr
Role:       production
```

### Legacy compatibility artifact

```text
models/nps_predictor_model.pkl
```

This is a compatibility mirror and is explicitly `legacy: true`. It must never silently become canonical production.

### Stress artifact

```text
models/stess_test_NPS.pkl
```

This is a stress/test artifact and must never silently become production.

Production NPS and stress NPS are confirmed different.

---

## Production Entry Points

```text
API NPSService()
    → models/production_NPS.pkl

ForecastAI create_oh_predictor()
    → models/production_OH.pkl

ForecastAI create_nps_predictor()
    → models/production_NPS.pkl

load_model_pair("production")
    → canonical production OH/NPS pair
```

Legacy loading is explicit/opt-in.

Test, stress, smoke, and legacy artifacts cannot silently become production.

### Model-family selection

Model families are discovered by scanning `models/` for complete `<family>_OH.pkl`
+ `<family>_NPS.pkl` pairs (`core/forecast_ai/prediction/model_selector.list_model_families`).

The GUI model selector passes the chosen family into the service layer, which
activates it on `PredictorProvider`. Predictors are then created by name:
`create_oh_predictor("<family>")` loads `models/<family>_OH.pkl` and
`create_nps_predictor("<family>")` loads `models/<family>_NPS.pkl`
(`core/forecast_ai/prediction/predictor_config`). Changing the selected family
therefore changes which artifact files are loaded for inference.

Currently available families:

- `100k-10yr` — the named source pair (`models/100k-10yr_OH.pkl`,
  `models/100k-10yr_NPS.pkl`).
- `production` — `models/production_OH.pkl` / `models/production_NPS.pkl`, the
  promoted canonical pair. It is a byte-identical/promoted copy of the
  `100k-10yr` artifacts (same SHA-256 in the manifest), so selecting either
  family runs the same fitted models.

---

## Production Artifact Integrity

Production loading is fail-closed.

Validation includes:

- SHA-256 against manifest;
- manifest entry;
- production role;
- non-legacy status;
- non-empty provenance;
- loadability;
- trained state;
- feature metadata;
- feature count;
- output dimensionality;
- NPS 11-output contract.

A corrupt or incompatible production artifact must fail rather than silently falling back.

### Fallback behavior

Fallback predictors are **never** used silently when production artifacts are missing.
If no canonical artifact can be loaded, the default behavior is to **raise**
(`FileNotFoundError` from `predictor_config.create_oh_predictor()` /
`create_nps_predictor()`, or `RuntimeError` when the loaded model is not trained).
The trained model is always called; an absent model surfaces as an error, never as
a deterministic substitute.

The only fallback predictors (`_FallbackOHPredictor` / `_FallbackNPSPredictor`) are
explicit degraded-mode objects that compute a hard-coded formula. They are reachable
**only** when all of the following hold:

- no canonical production artifact exists for the family;
- no legacy mirror exists;
- the environment variable `AXIPULSE_ALLOW_FALLBACK_MODE` is set to
  `1`, `true`, or `yes` (see `predictor_config._allow_fallback()`).

Without that opt-in flag, a missing model raises instead of degrading.

---

## Production Promotion

Production models are promoted through the hardened production registry.

The registry uses immutable generation directories and an atomic active-generation pointer.

The intended invariant is:

```text
reader sees old complete generation
OR
reader sees new complete generation
```

Candidate validation requires:

- loadable model;
- trained model;
- production-safe role;
- valid metadata;
- valid provenance;
- compatible structure.

Test/stress/smoke/legacy-only candidates cannot become canonical production.

---

## Temporal Integrity

The system enforces the cutoff/target-time contract.

Where required:

```text
feature_time < target_time
```

Future information must never enter training or inference features.

Provenance distinguishes:

```text
observed
predicted
```

values.

Recursive OH→NPS prediction preserves the observed/predicted distinction.

`enrollment_factor` has a narrow explicit cutoff-known exemption and is not a general future-data bypass.

Repeated dates are handled through trajectory semantics/identifiers.

Rolling-origin CV preserves chronological order.

---

## Call Volume

Baseline:

```text
2,000 calls/day
```

Weekday multipliers:

| Day | Multiplier |
|---|---:|
| Monday | 1.20 |
| Tuesday | 1.15 |
| Wednesday | 1.00 |
| Thursday | 0.85 |
| Friday | 0.75 |
| Saturday | 0.00 |
| Sunday | 0.00 |

Final seven simulation days use a 50% call-volume reduction.

Season modifiers:

| Season | Multiplier |
|---|---:|
| NORMAL | 1.00 |
| OEP | 1.20 |
| AEP | 1.50 |
| BENEFIT_ACTIVATION | 1.30 |

Queue pressure is not an independent driver.

---

## Complexity

Complexity levels:

```text
Low
Medium
High
Very High
Critical
```

Normal distribution:

```text
Low:       30%
Medium:    30%
High:      20%
Very High: 10%
Critical:  event-driven
```

Critical complexity should remain rare under normal conditions and increase primarily under severe events/stress.

Handle times:

| Complexity | Handle Time |
|---|---:|
| Low | 4 min |
| Medium | 8 min |
| High | 15 min |
| Very High | 25 min |
| Critical | 45 min |

Handle time affects probability behavior; it does not directly add/subtract NPS points.

---

## Event System

Exactly one event is selected per simulation day.

| Event | Probability |
|---|---:|
| NORMAL | 40% |
| PHARMACY_DELAY | 10% |
| PROVIDER_UPDATE | 20% |
| CLAIMS_BACKLOG | 10% |
| SYSTEM_SLOWDOWN | 3% |
| CORE_OUTAGE | 2% |
| CMS_CHANGE | 5% |
| TRAINING | 10% |

Event precedence:

```text
1. Hard business constraints
2. Event effects
3. OH effects
4. Intelligence effects
5. Complexity effects
6. Momentum/recovery
7. Random variation
```

---

## Momentum and Recovery

Default autocorrelation:

```text
0.6
```

Conceptually:

```text
current_state =
    0.6 * previous_state
    +
    0.4 * current_base_state
    +
    controlled_noise
```

Momentum prevents unrealistic daily jumps and supports gradual recovery after severe events.

---

## Agent Intelligence

Agent attributes include:

- operational intelligence;
- business intelligence;
- member intelligence;
- experience;
- fatigue;
- availability;
- shift information;
- training status;
- knowledge-category proficiency.

Experience competency factors:

| Experience | Factor |
|---|---:|
| <4 weeks | 0.95 |
| 4–8 weeks | 0.97 |
| 8–16 weeks | 0.99 |
| >16 weeks | 1.00 |

---

## Resource Safety

The system considers:

- RAM;
- CPU;
- subprocess isolation;
- timeouts;
- memory ceilings;
- final-fit feasibility;
- GPU availability;
- GPU/CPU fallback.

A model cannot win CV if it cannot safely perform its final fit.

### OH 1M-row training

Final verification:

```text
Training rows: approximately 1M
Selection sample: 10,000 rows
Sampled dates remain aligned with sampled rows
```

Infeasible candidates are excluded before final selection.

### NPS

```text
Serial subprocess CV
500-row selection sample
8 candidates
2 folds
```

Infeasible candidates are excluded and XGBoost was selected.

---

## Confidence Contract

Confidence is heuristic unless genuinely statistically calibrated.

Current contract:

```text
kind = heuristic
calibrated = false
statistical = false
```

Horizon decay is deterministic.

The GUI labels heuristic confidence as non-statistical.

---

## Risk Contract

Risk factors identify their source:

```text
business_rule
model
derived
```

Business-rule severity is not presented as model evidence.

Model-derived risk requires model-output provenance.

When evidence is insufficient, the system abstains rather than fabricating evidence.

---

## Forecast / Decision Pipeline

```text
Model
  ↓
Forecast
  ↓
Recommendation
  ↓
Evidence
  ↓
Agreement
  ↓
Risk
  ↓
Decision
  ↓
GUI
```

The system must not:

- fabricate evidence;
- use stale forecast fields;
- silently fallback;
- produce contradictory decisions;
- present business rules as model predictions.

### Forecast uses trained models

Production Forecast drives every horizon day from the **trained model artifacts**;
it does not manufacture OH/NPS from deterministic formulas.

- **OH** predictions come from the trained **CatBoostRegressor** (loaded from
  `models/<family>_OH.pkl`; `core/operation_health_predictor.inference` →
  `model.predict`).
- **NPS** predictions come from the trained **XGBoost-based MultiOutputRegressor**
  (loaded from `models/<family>_NPS.pkl`; 11 survey-score outputs) which feeds the
  canonical Bayesian / Monte-Carlo survey-score analysis → NPS.
- Forecast is **recursive**: each day's evolved state and features feed the next
  day's prediction (`prediction[t] → state[t] → features[t+1] → prediction[t+1]`).
- **Scenario changes perturb the input KPI state only**; they do not directly
  manufacture OH/NPS outputs. OH/NPS always come from the model.

---

## Reverse Optimization

`gui.services.reverse_optimize_canonical()` is the canonical **OH/NPS reverse
optimization** entry point. The GUI reverse path calls it and it drives the
canonical `core.forecast_ai.engines.reverse_optimizer.ReverseOptimizer`, which
generates operational states and evaluates each through the canonical
`PredictionService` (joint OH + NPS from the same generated state). It never
calls TargetStateEngine for this OH/NPS reverse path.

TargetStateEngine remains available only for the separate **multi-KPI Target State**
feature via `gui.services.find_target_state()`; it is not part of the canonical
OH/NPS reverse optimization path.

---

## GUI

The GUI supports explicit model-family selection.

It provides:

- stale-state invalidation when family changes;
- production/test isolation;
- human-readable recommendation;
- human-readable risk;
- heuristic confidence labeling;
- correct selected-algorithm metrics;
- technical JSON only in technical-detail areas.

---

## API

The API defaults to canonical production models.

```text
NPSService()
→ production_NPS.pkl
```

Model-loading failures fail closed.

Legacy models are explicit/opt-in only.

---

## Testing

Important behavioral test areas:

- production lifecycle;
- production artifact isolation;
- real NPS inference;
- temporal provenance;
- confidence/risk contracts;
- decision quality;
- forecast orchestration;
- GUI state/leaderboard;
- API NPS;
- training lifecycle;
- feature alignment;
- resource safety;
- end-to-end integration.

Final Python 3.13 release result:

```text
722 passed
0 skipped
```

Compilation:

```text
clean
```

---

## Installation

Create the Python 3.13 environment:

```bash
uv venv --python 3.13 .venv313
```

Activate:

```bash
source .venv313/bin/activate
```

Verify:

```bash
python --version
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## Running Tests

Full suite:

```bash
python -m pytest -q
```

Real NPS inference:

```bash
python -m pytest -q tests/project/core/nps_predictor/test_nps_real_inference.py
```

Production isolation:

```bash
python -m pytest -q tests/project/core/forecast_ai/prediction/test_production_artifact_isolation.py
```

Production lifecycle:

```bash
python -m pytest -q tests/project/core/forecast_ai/prediction/test_production_lifecycle.py
```

Training/features:

```bash
python -m pytest -q   core/nps_predictor/tests/test_training.py   core/nps_predictor/tests/test_features.py
```

Compile:

```bash
python -m compileall -q api core gui tests
```

---

## Model Hash Verification

Generate hashes:

```bash
sha256sum   models/production_OH.pkl   models/production_NPS.pkl   models/nps_predictor_model.pkl   models/stess_test_NPS.pkl
```

Verify production NPS differs from stress NPS:

```bash
cmp -s   models/production_NPS.pkl   models/stess_test_NPS.pkl   && echo "ERROR: identical"   || echo "OK: production NPS differs from stress NPS"
```

---

## Backup

Final backup directory:

```text
/home/amteur/AxiPulseAI_FINAL_BACKUPS
```

Create a final backup:

```bash
mkdir -p /home/amteur/AxiPulseAI_FINAL_BACKUPS && tar   --exclude='.git'   --exclude='.venv313'   --exclude='__pycache__'   --exclude='*.pyc'   --exclude='.pytest_cache'   --exclude='.coverage'   --exclude='htmlcov'   -czf "/home/amteur/AxiPulseAI_FINAL_BACKUPS/AxiPulseAI_V2_FINAL_$(date +%Y%m%d_%H%M%S).tar.gz" .
```

List backups:

```bash
ls -lh /home/amteur/AxiPulseAI_FINAL_BACKUPS/
```

Verify latest backup:

```bash
tar -tzf "$(ls -t /home/amteur/AxiPulseAI_FINAL_BACKUPS/AxiPulseAI_V2_FINAL_*.tar.gz | head -1)"   >/dev/null && echo "BACKUP VERIFIED"
```

Checksum:

```bash
sha256sum "$(ls -t /home/amteur/AxiPulseAI_FINAL_BACKUPS/AxiPulseAI_V2_FINAL_*.tar.gz | head -1)"
```

---

## Git Release

Check repository:

```bash
git status
git branch --show-current
git remote -v
git log -1 --oneline
```

Large training dataset:

```text
training/1mil-10yr.csv
```

is approximately 403.89 MB and must not be committed directly to GitHub.

Add it to `.gitignore`:

```bash
printf '
# Large training datasets
training/1mil-10yr.csv
' >> .gitignore
```

Remove it from Git tracking while keeping the local file:

```bash
git rm --cached -- training/1mil-10yr.csv
```

Verify:

```bash
git ls-files training/1mil-10yr.csv
```

Expected:

```text
(no output)
```

Commit release:

```bash
git add -A
git commit -m "Release AxiPulseAI V2"
```

Create tag:

```bash
git tag -a v2.0.0 -m "AxiPulseAI V2 production release"
```

Push:

```bash
git push -u origin master
git push origin v2.0.0
```

Verify:

```bash
git status
git tag --points-at HEAD
git ls-remote --heads origin master
git ls-remote --tags origin v2.0.0
```

---

## Release Freeze

After `v2.0.0`:

- do not modify production models casually;
- do not retrain without a new release candidate;
- do not change runtime requirements without compatibility validation;
- do not manually replace production artifacts;
- do not promote stress/test models;
- preserve the final backup;
- preserve the release manifest and artifact hashes.

Any production model change should follow:

```text
candidate
  ↓
training
  ↓
CV
  ↓
resource validation
  ↓
artifact validation
  ↓
temporal validation
  ↓
production promotion
  ↓
fresh-process inference
  ↓
full test suite
  ↓
new release
```

---

## Known Non-Blocking Risks

### GPU VRAM estimation

GPU VRAM footprint estimation remains conservative/incomplete.

### Permutation importance

Post-fit permutation importance currently uses:

```text
n_jobs=-1
```

These are documented resource-governance improvements and are not release-blocking defects for V2.

---

## Final Release Checklist

```text
[x] Production OH verified
[x] Production NPS verified
[x] Production manifest verified
[x] Production != stress artifact
[x] API uses canonical production NPS
[x] ForecastAI uses canonical production OH/NPS
[x] Legacy artifact isolated
[x] Test/stress artifacts isolated
[x] Production integrity fail-closed
[x] Production promotion atomic
[x] NPS 11-output contract verified
[x] NPS bucket semantics verified
[x] NPS calculation verified
[x] Feature ordering verified
[x] Feature dtype validation verified
[x] Temporal cutoff verified
[x] Future leakage checks verified
[x] Provenance verified
[x] Recursive OH→NPS verified
[x] Repeated-date semantics verified
[x] Rolling-origin CV verified
[x] OH 1M-row sampling verified
[x] Final-fit feasibility verified
[x] NPS resource safety verified
[x] Confidence contract verified
[x] Risk source attribution verified
[x] Forecast→decision contract verified
[x] GUI model isolation verified
[x] GUI leaderboard verified
[x] Real artifact inference verified
[x] Persistence/reload verified
[x] Python 3.13.15 compatibility verified
[x] Full test suite: 722 passed / 0 skipped
[x] py_compile clean
[x] Final backup created
[x] Large training CSV excluded from Git
[x] Release commit created
[x] v2.0.0 tag created
[x] Release pushed
```

---

## Final Release Identity

```text
========================================
          AXIPULSEAI V2
========================================

Version:
v2.0.0

Status:
READY

Runtime:
Python 3.13.15

Production Source:
100k-10yr

OH:
CatBoost
19 features
1 output
93.03606462275141

NPS:
XGBoost
34 features
11 outputs
NPS = 84.0

Tests:
722 passed
0 skipped

Production:
FAIL-CLOSED

Promotion:
ATOMIC

Temporal Leakage:
NO CONFIRMED DEFECT

Production/Test Isolation:
VERIFIED

Persistence/Reload:
VERIFIED

Final Verdict:
READY

========================================
```

---

## License / Ownership

Add the project's actual license and ownership terms here if applicable.

Do not invent a license that is not present in the repository.
