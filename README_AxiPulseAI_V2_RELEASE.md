# AxiPulseAI V2 — Production Release README

## Release

**Version:** v2.0.0  
**Status:** READY  
**Release runtime:** Python 3.13.15  
**Production source family:** 1mil-10yr

This README records the commands used to audit, verify, back up, clean, freeze, and publish the AxiPulseAI V2 production release.

---

# 1. Enter the repository

```bash
cd ~/Documents/AxiPulseAI_V2
```

Check the repository:

```bash
pwd
git status
git branch --show-current
git remote -v
git log -1 --oneline
```

Expected final state:

```text
On branch master
nothing to commit, working tree clean
```

---

# 2. Python 3.13 release environment

The authoritative production artifact runtime is:

```text
Python 3.13.15
```

Create/use the isolated environment as appropriate:

```bash
uv venv --python 3.13 .venv313
source .venv313/bin/activate
```

Verify:

```bash
python --version
python -m pip --version
```

Install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

Verify important packages:

```bash
python -m pip show numpy pandas scikit-learn catboost xgboost joblib pyarrow scipy lightgbm streamlit pytest
```

---

# 3. Run the test suite

Full suite:

```bash
python -m pytest -q
```

The final Python 3.13 release verification reported:

```text
701 passed
0 skipped
```

Run real NPS inference tests:

```bash
python -m pytest -q tests/project/core/nps_predictor/test_nps_real_inference.py
```

Production artifact isolation:

```bash
python -m pytest -q tests/project/core/forecast_ai/prediction/test_production_artifact_isolation.py
```

Production lifecycle:

```bash
python -m pytest -q tests/project/core/forecast_ai/prediction/test_production_lifecycle.py
```

Training/features:

```bash
python -m pytest -q core/nps_predictor/tests/test_training.py core/nps_predictor/tests/test_features.py
```

Compile changed Python files:

```bash
python -m compileall -q core api gui tests
```

---

# 4. Verify production model paths

Check canonical files:

```bash
ls -lh models/production_OH.pkl
ls -lh models/production_NPS.pkl
ls -lh models/manifest.json
```

Check the production generation structure:

```bash
find models/production_generations -maxdepth 2 -type f -o -type l | sort
```

Check canonical links:

```bash
ls -l models/production_OH.pkl
ls -l models/production_NPS.pkl
ls -l models/manifest.json
```

Verify stress and production NPS are different:

```bash
sha256sum models/production_NPS.pkl models/stess_test_NPS.pkl
```

---

# 5. Verify production artifact hashes

Current release artifact hashes:

```text
production_OH.pkl
0814ac59…

production_NPS.pkl
e91ee9c8…

nps_predictor_model.pkl
e91ee9c8…

stess_test_NPS.pkl
0ee5b6a0…
```

Generate the actual full hashes at any time:

```bash
sha256sum \
  models/production_OH.pkl \
  models/production_NPS.pkl \
  models/nps_predictor_model.pkl \
  models/stess_test_NPS.pkl
```

Verify the production NPS is not the stress artifact:

```bash
cmp -s models/production_NPS.pkl models/stess_test_NPS.pkl \
  && echo "ERROR: identical" \
  || echo "OK: production NPS differs from stress NPS"
```

---

# 6. Verify production entrypoints

Canonical API NPS model:

```text
api/services/nps_service.py
→ models/production_NPS.pkl
```

Canonical ForecastAI models:

```text
create_oh_predictor()
→ models/production_OH.pkl

create_nps_predictor()
→ models/production_NPS.pkl

load_model_pair("production")
→ canonical production OH/NPS pair
```

Search all model-loading locations:

```bash
grep -RInE \
  'joblib\.load|pickle\.load|load_model|DEFAULT_MODEL_PATH|production_OH|production_NPS|nps_predictor_model|stess_test' \
  api core gui --exclude-dir='__pycache__'
```

---

# 7. Production artifact integrity

The production loader must fail closed on:

- missing manifest;
- missing artifact entry;
- SHA-256 mismatch;
- wrong role;
- legacy artifact;
- empty provenance;
- corrupt artifact;
- unloadable model;
- untrained model;
- incompatible feature schema;
- incorrect feature count;
- incorrect output count;
- incompatible runtime.

Do not bypass these checks.

---

# 8. Fresh-process production inference

OH:

```bash
python -c "
from core.forecast_ai.prediction.predictor_config import create_oh_predictor
p = create_oh_predictor()
print(type(p))
"
```

NPS:

```bash
python -c "
from core.forecast_ai.prediction.predictor_config import create_nps_predictor
p = create_nps_predictor()
print(type(p))
"
```

API NPS service:

```bash
python -c "
from api.services.nps_service import NPSService
s = NPSService()
print(s)
"
```

The final release verification recorded:

```text
OH prediction: 93.03606462275141
NPS outputs: 11
NPS: 84.0
NPS features: 34
OH features: 19
```

---

# 9. NPS contract

NPS has 11 outputs representing scores:

```text
0 1 2 3 4 5 6 7 8 9 10
```

Buckets:

```text
Detractors = 0–6
Passives   = 7–8
Promoters  = 9–10
```

Formula:

```text
NPS = (promoters - detractors) / total * 100
```

Production inference must preserve this contract.

---

# 10. Backup the final repository

Backup destination:

```text
/home/amteur/AxiPulseAI_FINAL_BACKUPS
```

Create a compressed backup:

```bash
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
```

List backups:

```bash
ls -lh /home/amteur/AxiPulseAI_FINAL_BACKUPS/
```

Verify the latest archive:

```bash
tar -tzf "$(ls -t /home/amteur/AxiPulseAI_FINAL_BACKUPS/AxiPulseAI_V2_FINAL_*.tar.gz | head -1)" >/dev/null \
  && echo "BACKUP VERIFIED"
```

Generate a checksum:

```bash
sha256sum "$(ls -t /home/amteur/AxiPulseAI_FINAL_BACKUPS/AxiPulseAI_V2_FINAL_*.tar.gz | head -1)"
```

---

# 11. Remove the large training dataset from Git

GitHub rejects ordinary Git blobs larger than 100 MB.

The release training dataset:

```text
training/1mil-10yr.csv
```

is approximately 403.89 MB and must not be committed to the GitHub repository.

Keep the dataset locally if required for training/reproducibility, but exclude it from Git.

Add it to `.gitignore`:

```bash
printf '\n# Large training datasets\ntraining/1mil-10yr.csv\n' >> .gitignore
```

Remove it from Git tracking while keeping the local file:

```bash
git rm --cached -- training/1mil-10yr.csv
```

Verify it is no longer tracked:

```bash
git ls-files training/1mil-10yr.csv
```

Expected result:

```text
(no output)
```

Check that the local file still exists:

```bash
ls -lh training/1mil-10yr.csv
```

---

# 12. Commit the release

Check status:

```bash
git status
```

Commit:

```bash
git add -A
git commit -m "Release AxiPulseAI V2"
```

Create/update the release tag:

```bash
git tag -f v2.0.0
```

Verify:

```bash
git log -1 --oneline
git tag --list
git show --stat v2.0.0
```

---

# 13. Push to GitHub

Check the remote:

```bash
git remote -v
```

Expected:

```text
origin  https://github.com/iwasatapex/AxiPulseAI_V2.git
```

Push master:

```bash
git push -u origin master
```

Push the release tag:

```bash
git push -f origin v2.0.0
```

Do NOT use `--force` on `master` unless there is a deliberate repository-history recovery procedure.

---

# 14. If GitHub rejects a large file

If GitHub reports:

```text
GH001: Large files detected
File ... exceeds GitHub's file size limit
```

first determine whether the file is in the current commit:

```bash
git ls-files | grep 'training/1mil-10yr.csv'
```

If it is tracked:

```bash
git rm --cached -- training/1mil-10yr.csv
git add .gitignore
git commit --amend --no-edit
```

If the large file exists in an unpublished local commit history, rewrite the unpublished release commit carefully before pushing.

Do not use force-push against an established shared branch without first confirming repository state.

---

# 15. Final Git verification

After a successful push:

```bash
git status
```

Expected:

```text
nothing to commit, working tree clean
```

Verify commit:

```bash
git log -1 --oneline
```

Verify tag:

```bash
git tag --points-at HEAD
```

Verify remote:

```bash
git ls-remote --heads origin
git ls-remote --tags origin v2.0.0
```

---

# 16. Release documentation

Final release record:

```text
AxiPulseAI_V2_PRODUCTION_RELEASE_RECORD.txt
```

Release status:

```text
READY
```

Runtime:

```text
Python 3.13.15
```

Final reported Python 3.13 test gate:

```text
701 passed
0 skipped
```

Production inference:

```text
OH = 93.03606462275141
NPS = 84.0
NPS outputs = 11
```

---

# 17. Post-release rule

After `v2.0.0` is pushed:

- do not modify production models casually;
- do not retrain without creating a new release candidate;
- do not change runtime requirements without a new compatibility audit;
- do not replace production artifacts manually;
- do not delete the final backup;
- do not promote stress/test artifacts to production;
- preserve the release manifest and artifact hashes.

Any production model change should result in a new release/version and a fresh audit.

---

# 18. Quick release checklist

```text
[ ] Repository clean
[ ] Python 3.13.15 verified
[ ] Production OH verified
[ ] Production NPS verified
[ ] Manifest verified
[ ] Production != stress artifact
[ ] API uses production NPS
[ ] ForecastAI uses production OH/NPS
[ ] NPS 11-output contract verified
[ ] NPS bucket calculation verified
[ ] Temporal tests pass
[ ] Resource tests pass
[ ] GUI/decision tests pass
[ ] Full Python 3.13 test suite passes
[ ] Backup created
[ ] Backup checksum recorded
[ ] Large training CSV excluded from Git
[ ] .gitignore updated
[ ] Release commit created
[ ] v2.0.0 tag created
[ ] master pushed
[ ] v2.0.0 tag pushed
[ ] GitHub remote verified
[ ] Final release record preserved
```

---

## Final Release State

```text
AxiPulseAI V2
Version: v2.0.0
Status: READY
Runtime: Python 3.13.15
Production source: 1mil-10yr
Full test gate: 701 passed / 0 skipped
Production OH: 19 features
Production NPS: 34 features / 11 outputs
Production NPS: 84.0
```
