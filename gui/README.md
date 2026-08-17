# AxiPulseAI V2 GUI

A polished, interactive desktop/web GUI built with **Streamlit** on top of the
existing AxiPulseAI V2 services. The GUI is a **thin presentation layer** — it
delegates all model, training, prediction, forecast, and ADIE work to the
canonical V2 engines. No business logic is duplicated here.

## Architecture

```
gui/app.py            Streamlit entry point + sidebar navigation
gui/services.py       Thin service layer → delegates to canonical V2 services
gui/state.py          Session-scoped state (active model family, recent results)
gui/contracts.py      Canonical V2.3 KPI bounds / NPS range / dataset loader
gui/charts.py         Plotly chart builders (presentation only)
gui/views/            One module per page (Dashboard, Train, Models, …)
gui/components.py     Shared UI helpers (cards, pills, raw-JSON, errors)
gui/run.sh            Convenience launcher
```

### Model flow (session-scoped, request-serialized)

The active model family is a **per-session** preference stored in Streamlit's
session state — one browser session's selection never leaks into another.

* **Selection is session-scoped.** `gui.state` writes/reads the active family
  from `st.session_state` (a separate store per browser session).
* **Activation is request-scoped and serialized.** The canonical
  `PredictorProvider` is a process-global singleton, so the GUI activates the
  *explicit* family under a lock for each `predict`/`forecast`/ADIE request.
  Concurrent requests therefore can never cross model families (A always uses
  A, B always uses B), and a slow forecast in one session cannot change the
  model another session is using.
* **Training completion affects only the initiating session.** Training itself
  never touches session state; the Train view activates the freshly trained
  family on its own main thread, so session A training family C never changes
  session B's selection.
* **Models live in the canonical models directory.** Training and discovery
  resolve to `{repo}/models` via absolute paths — independent of the working
  directory the GUI is launched from.

The GUI is a **thin presentation layer** — it delegates all model, training,
prediction, forecast, and ADIE work to the canonical V2 engines. No business
logic is duplicated here.

> **Production model.** ForecastAI uses **trained ML models** for OH and NPS
> (not a rule-based simulator). See
> [`docs/FORECASTAI_ARCHITECTURE.md`](../docs/FORECASTAI_ARCHITECTURE.md) for
> the exact feature inputs, ML prediction flow, KPITransition/Scenario roles,
> and which V2.3 rules are explicit vs. learned vs. historical spec concepts.


## Start the GUI

From the V2 project root (`/home/amteur/Documents/AxiPulseAI_V2`), always use the
project virtualenv (`.venv`) — a system Python that has `streamlit` but not
`pydantic` fails with `ModuleNotFoundError: No module named 'pydantic'`.

```bash
# Preferred: activate the reproducible venv, then run streamlit.
source .venv/bin/activate
python -m streamlit run gui/app.py

# Or skip activation by calling the venv interpreter directly:
.venv/bin/python -m streamlit run gui/app.py

# Or use the convenience launcher (it prefers .venv/bin/python):
./gui/run.sh
```

Open http://localhost:8501

## Pages

- **Dashboard** – active family, model availability, latest prediction/forecast.
- **Train** – lists every file in `training/`, trains OH+NPS from one dataset,
  shows progress/logs and final metrics; re-training replaces the family pair.
- **Models** – lists complete OH+NPS families with metadata; select the active
  family explicitly.
- **Predict** – explicit family selection, input form, real V2 prediction with
  OH + NPS + confidence/risk when available.
- **Forecast** – horizon (1/3/5/7/custom), scenario selector, runs the real
  `ForecastOrchestrator`, interactive OH/NPS timeline (predicted days labelled
  as predicted), risk/confidence/sensitivity.
- **Target State** – multi-target reverse optimization via the canonical
  `TargetStateEngine`; enter one or more targets and get the recommended
  operational state, council consensus, distance, and model leaderboards.
- **Reverse Optimizer** – pick a single metric (OH or NPS), set a target, and
  get the KPIs that get you closest to it (also via the `TargetStateEngine`).
- **ADIE Decision** – canonical V3 decision package rendered in human-friendly
  sections, with a collapsible **Raw JSON** view. Missing sections show
  "Unavailable"; nothing is fabricated.
- **Settings** – paths, system info, theme guidance.

## Tests

```bash
.venv/bin/python -m pytest tests/test_gui_services.py -v
.venv/bin/python -m pytest tests/ -q   # full regression
```

## Theme

Dark/light is toggled via the Streamlit theme selector or `.streamlit/config.toml`
in the project root. The app adapts automatically via CSS variables.
