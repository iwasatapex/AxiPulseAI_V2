# AxiPulseAI V2

Production-grade AI forecasting and decision platform for contact-center operations.

**Release:** `v2.0.0`  
**Status:** `READY`  
**Production Runtime:** Python `3.13.15`  
**Production Source Family:** `1mil-10yr`

---

## 1. Overview

AxiPulseAI V2 is an AI-driven operational forecasting and decision system designed around a staged causal architecture.

The system separates:

- operational state generation;
- Operations Health (OH);
- NPS prediction;
- forecasting;
- recommendations;
- evidence;
- risk;
- decisioning;
- GUI presentation.

The architecture is designed to prevent downstream predictions from being used to incorrectly force upstream state.

The central causal flow is:

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
Promoters / Passives / Detractors
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
3. Core Architecture

AxiPulseAI uses a staged causal architecture.

Core principles
No same-step OH circularity

OH is calculated from a BASE operational state.

OH is then applied downstream.

OH must not be recomputed from KPI values that OH already influenced within the same step.

BASE KPI state
      ↓
     OH
      ↓
OH-adjusted downstream metrics
OH does not directly modify NPS

NPS is generated from actual survey outcomes / modeled survey outcomes.

OH influences operational behavior upstream of survey generation, but it is not used as a direct NPS score modifier.

Transfer occurs once

Transfer is calculated before release.

Transferred calls are removed from the remaining-call population.

Post-transfer release is calculated only against remaining calls.

Transfer must never be applied twice.

NPS is outcome-based

NPS is calculated from:

Promoters
Passives
Detractors

It is not directly forced from KPI values.

4. Operations Health

Operations Health (OH) is the operational master variable.

OH scale
Nominal range: 0–100

Raw OH may temporarily exceed the presentation bounds.

Reported/presentation OH is bounded, while raw OH remains available for validation.

OH components

Nominal weights:

Release       50%
Transfer      15% inverse
Competency    15%
Quality       15%
Call Volume    5%

OH is calculated from the BASE state before OH effects are applied downstream.

5. KPI Contracts

Default targets:

KPI	Target	Absolute Range
Quality	87%	60–100
Competency	93%	55–100
Attendance	90%	65–100
Release	60%	50–100
Transfer	9%	0–20

Important hard constraints:

Release target must never be below 50%.
Transfer target must never exceed 20%.
Quality target must never be below 60%.
KPI-met threshold

A KPI is considered met when it reaches at least 95% of target.

For Transfer, because lower is better:

actual transfer <= 105% of target

Overall KPI-met status requires at least:

3 of 4

checked KPIs:

Quality
Competency
Release
Transfer
6. NPS Model Contract

Production NPS uses an 11-output model.

The outputs represent:

0
1
2
3
4
5
6
7
8
9
10
NPS buckets
Detractors: 0–6
Passives:   7–8
Promoters:  9–10
NPS formula
NPS = (Promoters - Detractors) / Total Surveys * 100

Theoretical range:

-100 to +100

The production model must preserve the 11-output contract.

7. Production AI Models
Canonical OH model
Path:
models/production_OH.pkl


Algorithm:
CatBoost


Features:
19


Outputs:
1


Trained:
True


Python:
3.13.15


Source:
1mil-10yr


Role:
production

Production inference:

OH = 93.03606462275141
Canonical NPS model
Path:
models/production_NPS.pkl


Algorithm:
XGBoost


Features:
34


Outputs:
11


Trained:
True


Python:
3.13.15


Source:
1mil-10yr


Role:
production

Production inference:

NPS = 84.0
Legacy NPS artifact
models/nps_predictor_model.pkl

This is a compatibility mirror of the canonical production NPS artifact.

It is explicitly:

legacy: true

It must never silently become the canonical production model.

Stress NPS artifact
models/stess_test_NPS.pkl

This is a stress/test artifact.

It must never silently become the production NPS model.

The production and stress NPS artifacts are confirmed different.

8. Production Model Entry Points

Canonical production paths:

API NPSService()
    ↓
models/production_NPS.pkl
ForecastAI create_oh_predictor()
    ↓
models/production_OH.pkl
ForecastAI create_nps_predictor()
    ↓
models/production_NPS.pkl
load_model_pair("production")
    ↓
canonical production OH/NPS pair

Legacy models are opt-in only.

Test, stress, smoke, and legacy artifacts cannot silently become production.

9. Production Artifact Integrity

Production loading is fail-closed.

The production artifact verification contract validates:

SHA-256;
manifest entry;
production role;
non-legacy status;
provenance;
model loadability;
trained state;
feature metadata;
feature count;
output dimensionality;
NPS 11-output contract.

A corrupted or incompatible production artifact must fail rather than silently falling back to another model.

10. Production Promotion

Production models are promoted through the hardened production registry.

Production activation uses immutable generation directories.

Conceptually:

models/
    production_generations/
        <generation-id>/
            production_OH.pkl
            production_NPS.pkl
            legacy artifacts
            manifest.json


        current

The active generation is switched through an atomic pointer operation.

The intended invariant is:

reader sees old complete generation
OR
reader sees new complete generation

A reader must never see a partially updated OH/NPS/manifest combination.

Candidate validation requires:

loadable model;
trained model;
production-safe role;
valid metadata;
valid provenance;
compatible structure.

Test/stress/smoke/legacy-only candidates cannot be promoted as canonical production.

11. Model Manifest

The production manifest records artifact integrity and provenance.

Production metadata includes, where available:

filename;
SHA-256;
source;
role;
legacy status;
algorithm;
feature metadata;
output metadata;
runtime information.

The manifest must never contain fabricated hashes or fabricated provenance.

Production provenance must be non-empty.

12. Temporal Integrity

Temporal correctness is a release requirement.

The system enforces the cutoff/target-time contract.

Where required:

feature_time < target_time

Future information must never enter training or inference features.

Provenance

The system distinguishes:

observed
predicted

values.

Predicted OH must not be presented as observed OH.

Recursive OH → NPS

Recursive prediction is handled explicitly.

The system must preserve the distinction between:

observed OH

and:

predicted OH

through the recursive NPS path.

enrollment_factor

enrollment_factor has a narrow, explicit cutoff-known exemption.

It must not become a general future-data bypass.

Repeated dates

Repeated dates are supported through trajectory semantics/trajectory identifiers rather than assuming every date is globally unique.

Rolling-origin CV

Temporal CV preserves chronological order and prevents future observations from entering earlier training folds.

13. Call Volume

Baseline:

2,000 calls/day

Weekday multipliers:

Day	Multiplier
Monday	1.20
Tuesday	1.15
Wednesday	1.00
Thursday	0.85
Friday	0.75
Saturday	0.00
Sunday	0.00

Final seven simulation days:

50% call-volume reduction

Season modifiers:

Season	Multiplier
NORMAL	1.00
OEP	1.20
AEP	1.50
BENEFIT_ACTIVATION	1.30

Queue pressure is not an independent driver.

Workload is represented by:

Call Volume
+
OH volume component
14. Complexity

Complexity levels:

Low
Medium
High
Very High
Critical

Normal-day distribution:

Low:       30%
Medium:    30%
High:      20%
Very High: 10%
Critical:  event-driven

Critical complexity should remain rare under normal conditions and should increase primarily due to severe events/stress conditions.

Handle times
Complexity	Handle Time
Low	4 min
Medium	8 min
High	15 min
Very High	25 min
Critical	45 min

Long handle time affects probability behavior.

It does not directly add or subtract NPS points.

15. Event System

Exactly one event is selected per simulation day.

Default event probabilities:

Event	Probability
NORMAL	40%
PHARMACY_DELAY	10%
PROVIDER_UPDATE	20%
CLAIMS_BACKLOG	10%
SYSTEM_SLOWDOWN	3%
CORE_OUTAGE	2%
CMS_CHANGE	5%
TRAINING	10%

Event effects are additive metric effects on their own scales.

Event precedence:

1. Hard business constraints
2. Event effects
3. OH effects
4. Intelligence effects
5. Complexity effects
6. Momentum/recovery
7. Random variation
16. Momentum and Recovery

Momentum uses autocorrelation to prevent unrealistic daily jumps.

Default:

autocorrelation = 0.6

Conceptually:

current_state =
    0.6 * previous_state
    +
    0.4 * current_base_state
    +
    controlled_noise

Momentum applies to relevant KPI/OH/sentiment state variables.

Severe-event damage recovers gradually rather than disappearing immediately.

17. Agent Intelligence

Backend agent attributes include:

operational intelligence;
business intelligence;
member intelligence;
experience;
fatigue;
availability;
shift information;
training status;
knowledge-category proficiency.

Intelligence effects are baseline effects, not permanent hard penalties.

Higher intelligence reduces negative effects and improves effectiveness.

Experience competency factors:

Experience	Factor
<4 weeks	0.95
4–8 weeks	0.97
8–16 weeks	0.99
>16 weeks	1.00
18. Resource Safety

AxiPulseAI V2 includes resource-aware model selection.

The system considers:

RAM;
CPU;
subprocess isolation;
timeout;
memory ceilings;
final-fit feasibility;
GPU availability;
GPU/CPU fallback.

A model cannot win CV if it cannot safely perform its final fit.

OH 1M-row training

The 1M-row OH path uses a configured selection sample.

Final verification:

Training rows: approximately 1M
Selection sample: 10,000 rows
Sampled dates remain aligned with sampled rows

Infeasible candidates are excluded before final model selection.

Example:

ExtraTrees
RandomForest
HistGB
GradBoosting

were excluded when estimated memory exceeded the configured budget.

CatBoost was selected.

NPS

NPS CV uses:

serial subprocess CV
500-row selection sample
8 candidates
2 folds

Infeasible candidates are excluded.

XGBoost was selected for production.

19. Confidence Contract

Confidence must not be represented as calibrated statistical probability unless it has actually been calibrated.

Current confidence contract:

kind = heuristic
calibrated = false
statistical = false

Horizon decay is deterministic.

GUI wording explicitly identifies heuristic confidence as non-statistical.

20. Risk Contract

Risk factors identify their source kind:

business_rule
model
derived

Business-rule severity must not be presented as model-derived evidence.

Model-derived risk requires appropriate model-output provenance.

Evidence gating prevents contradictory or insufficient-evidence states.

When evidence is insufficient, the system should:

ABSTAIN

rather than fabricate confidence or evidence.

21. Forecast / Decision Pipeline

The production decision chain is:

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

Each stage must preserve consistent payload semantics.

The system must not:

fabricate evidence;
use stale forecast fields;
silently fallback to another model;
produce a decision contradictory to the evidence;
present business rules as model predictions.
22. GUI

The GUI supports explicit model-family selection.

The GUI must:

invalidate stale state when the selected family changes;
prevent silent model fallback;
display human-readable recommendations;
display human-readable risk;
display confidence with appropriate heuristic labeling;
show correct selected-algorithm metrics;
isolate production/test models;
isolate legacy artifacts.

Raw JSON is restricted to technical-details areas.

Production users should not be presented with raw internal dictionaries as the primary UI.

23. API

The API uses canonical production artifacts by default.

Default:

NPSService()
→ production_NPS.pkl

Legacy loading is explicit.

Model-loading failures must fail closed.

The API must never silently turn:

model load failure

into:

plausible-looking prediction
24. Repository Structure

Important areas include:

api/
    API services


core/
    ForecastAI
    OH
    NPS
    risk
    decision
    temporal logic


gui/
    GUI views


models/
    production models
    manifests
    production generations
    test/stress artifacts


tests/
    integration
    behavioral
    production isolation
    temporal
    resource
    GUI
25. Testing

The release contains behavioral tests rather than relying only on test counts.

Important test areas:

Production lifecycle
Production artifact isolation
Real NPS inference
Temporal provenance
Confidence/risk contracts
Decision quality
Forecast orchestration
GUI state/leaderboard
API NPS
Training lifecycle
Feature alignment
Resource safety
End-to-end integration

Final Python 3.13 release test result:

701 passed
0 skipped

Compilation:

py_compile / compileall clean
26. Real Artifact Verification

Production artifacts must be tested as real serialized artifacts.

Do not replace production artifact tests with toy models or mocks.

Real NPS verification confirms:

34 features
11 outputs
NPS = 84.0

Real OH verification confirms:

19 features
OH = 93.03606462275141
27. Production Runtime

The supported runtime is:

Python >=3.13,<3.14

The final production artifacts were rebuilt and verified under:

Python 3.13.15

Key release dependency versions:

numpy       2.4.6
pandas      3.0.5
scikit-learn 1.9.0
catboost    1.2.10
xgboost     3.3.0
joblib      1.5.3
pyarrow     24.0.0
scipy       1.18.0
lightgbm    4.7.0
streamlit   1.61.1
pytest      9.1.1

The exact dependency source/lock configuration in the repository remains authoritative.

28. Installation

Create the environment:

uv venv --python 3.13 .venv313

Activate:

source .venv313/bin/activate

Verify:

python --version

Install dependencies:

python -m pip install -r requirements.txt
29. Running Tests

Full suite:

python -m pytest -q

Real NPS:

python -m pytest -q tests/project/core/nps_predictor/test_nps_real_inference.py

Production isolation:

python -m pytest -q tests/project/core/forecast_ai/prediction/test_production_artifact_isolation.py

Production lifecycle:

python -m pytest -q tests/project/core/forecast_ai/prediction/test_production_lifecycle.py

Training/features:

python -m pytest -q \
  core/nps_predictor/tests/test_training.py \
  core/nps_predictor/tests/test_features.py

Compile:

python -m compileall -q api core gui tests
30. Model Hash Verification

Generate hashes:

sha256sum \
  models/production_OH.pkl \
  models/production_NPS.pkl \
  models/nps_predictor_model.pkl \
  models/stess_test_NPS.pkl

Verify production and stress NPS are different:

cmp -s \
  models/production_NPS.pkl \
  models/stess_test_NPS.pkl \
  && echo "ERROR: identical" \
  || echo "OK: production NPS differs from stress NPS"
31. Backup Procedure

Final backup location:

/home/amteur/AxiPulseAI_FINAL_BACKUPS

Create a backup:

mkdir -p /home/amteur/AxiPulseAI_FINAL_BACKUPS && \
tar \
  --exclude='.git' \
  --exclude='.venv313' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  --exclude='.coverage' \
  --exclude='htmlcov' \
  -czf "/home/amteur/AxiPulseAI_FINAL_BACKUPS/AxiPulseAI_V2_FINAL_$(date +%Y%m%d_%H%M%S).tar.gz" .

List backups:

ls -lh /home/amteur/AxiPulseAI_FINAL_BACKUPS/

Verify the latest backup:

tar -tzf "$(ls -t /home/amteur/AxiPulseAI_FINAL_BACKUPS/AxiPulseAI_V2_FINAL_*.tar.gz | head -1)" \
  >/dev/null \
  && echo "BACKUP VERIFIED"

Generate backup checksum:

sha256sum "$(ls -t /home/amteur/AxiPulseAI_FINAL_BACKUPS/AxiPulseAI_V2_FINAL_*.tar.gz | head -1)"
32. Git Configuration

Check repository:

git status
git branch --show-current
git remote -v
git log -1 --oneline

Expected release branch:

master

Expected remote:

https://github.com/iwasatapex/AxiPulseAI_V2.git
33. Large Training Dataset

The following training dataset is approximately 403.89 MB:

training/1mil-10yr.csv

GitHub's normal Git file limit is 100 MB.

The dataset must not be committed directly to Git.

Add it to .gitignore:

printf '\n# Large training datasets\ntraining/1mil-10yr.csv\n' >> .gitignore

Remove it from Git tracking while keeping it locally:

git rm --cached -- training/1mil-10yr.csv

Verify:

git ls-files training/1mil-10yr.csv

Expected:

(no output)

Verify the local file remains:

ls -lh training/1mil-10yr.csv
34. Release Commit

Check:

git status

Commit:

git add -A
git commit -m "Release AxiPulseAI V2"

Create the release tag:

git tag -a v2.0.0 -m "AxiPulseAI V2 production release"

Verify:

git log -1 --oneline
git tag --points-at HEAD
git show --stat v2.0.0
35. Push Release

Push master:

git push -u origin master

Push tag:

git push origin v2.0.0

Verify remote:

git ls-remote --heads origin
git ls-remote --tags origin v2.0.0
36. Post-Release Verification

After pushing:

git status

Expected:

nothing to commit, working tree clean

Verify release tag:

git tag --points-at HEAD

Verify commit:

git log -1 --oneline

Verify remote branch:

git ls-remote --heads origin master
37. Release Freeze

After v2.0.0 is released:

Do not modify production models casually.

Do not retrain production models without creating a new release candidate.

Do not change:

Python runtime;
dependency versions;
model algorithms;
model feature schema;
production artifact roles;
temporal contracts;
NPS semantics;
OH semantics;

without a new audit.

Any production model change requires:

new candidate
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
38. Known Non-Blocking Risks

The final release audit identified two non-blocking resource improvements:

GPU VRAM estimation

GPU VRAM footprint estimation is conservative/incomplete.

This is documented and does not currently block release.

Permutation importance parallelism

Post-fit permutation importance currently uses:

n_jobs=-1

This can consume substantial CPU resources.

It is a resource-governance improvement for a future release, not a confirmed production blocker for V2.

39. Final Release Checklist
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
[x] Full test suite: 701 passed / 0 skipped
[x] py_compile clean
[x] Final backup created
[x] Large training CSV excluded from Git
[x] Release commit created
[x] v2.0.0 tag created
[x] Release pushed
40. Final Release Identity
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
1mil-10yr


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
701 passed
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
License / Ownership

Add the project's actual license and ownership terms here if applicable.

Do not invent a license that is not present in the repository.
