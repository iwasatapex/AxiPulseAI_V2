#!/usr/bin/env python3

from dataclasses import asdict, is_dataclass

from core.forecast_ai.engines.forecast_orchestrator import ForecastOrchestrator
from core.forecast_ai.engines.risk_engine import RiskEngine
from core.forecast_ai.models import ForecastRequest
from core.forecast_ai.prediction import (
    PredictorProvider,
    ModelPairError,
    select_model_family,
    list_model_families,
)


WIDTH = 108


def banner(text):
    print("═" * WIDTH)
    print(text.center(WIDTH))
    print("═" * WIDTH)


def get_default_state():
    return {
        "quality": 87.0,
        "competency": 93.0,
        "attendance": 90.0,
        "transfer": 9.0,
        "release": 60.0,
        "operations_health": 95.0,
        "nps": 82.0,
        "total_surveys": 100,
        "survey_rate": 5.0,
    }


def prompt_state(title, source):
    defaults = get_default_state()

    print()
    print(title)
    print("─" * WIDTH)
    print("Press Enter to use the default value.")
    print()

    def ask(label, key):
        raw = input(f"{label} [{defaults[key]}]: ").strip()
        if not raw:
            return defaults[key]
        try:
            return float(raw)
        except ValueError:
            print(f"Invalid value for {label}. Using {defaults[key]}.")
            return defaults[key]

    state = {
        "quality": ask("Quality %", "quality"),
        "competency": ask("Competency %", "competency"),
        "attendance": ask("Attendance %", "attendance"),
        "release": ask("Release Rate %", "release"),
        "transfer": ask("Transfer Rate %", "transfer"),
        "operations_health": ask("Operational Health %", "operations_health"),
        "nps": ask("NPS", "nps"),
        "total_surveys": defaults["total_surveys"],
        "survey_rate": defaults["survey_rate"],
        "state_source": source,
    }

    return state


def get_starting_state():
    print()
    print("FORECAST STARTING STATE")
    print("─" * WIDTH)
    print()
    print("How should Forecast AI initialize the forecast?")
    print()
    print("1 - Use Target State Engine (TSE) output")
    print("2 - Use yesterday's actual KPIs")
    print()

    while True:
        choice = input("Choice: ").strip()

        if choice == "1":
            return prompt_state(
                "TARGET STATE ENGINE OUTPUT",
                "tse",
            )

        if choice == "2":
            return prompt_state(
                "YESTERDAY'S ACTUAL KPIs",
                "yesterday_actual",
            )

        print("Invalid choice. Enter 1 or 2.")


def as_dict(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return value
    return value.__dict__ if hasattr(value, "__dict__") else {}


def get_timeline(result):
    if not result or not result.payload:
        return []

    timeline = result.payload.get("timeline", [])

    return [
        as_dict(day)
        for day in timeline
    ]


def fmt_pct(value):
    if value is None:
        return "—"
    return f"{float(value):.1f}%"


def fmt_nps(value):
    if value is None:
        return "—"
    return f"{float(value):.1f}"


def fmt_calls(value):
    if value is None:
        return "—"
    return f"{float(value):,.0f}"


def print_forecast_table(result, title, scenario):
    timeline = get_timeline(result)

    print()
    print(f"▸ {title}")
    print("─" * WIDTH)
    print(f"  Scenario : {scenario}")
    print(f"  Horizon  : {len(timeline)} days")
    print()

    header = (
        f"{'Day':<12}"
        f"{'Quality':>10}"
        f"{'Competency':>12}"
        f"{'Attendance':>12}"
        f"{'Release':>10}"
        f"{'Transfer':>10}"
        f"{'OH':>10}"
        f"{'NPS':>9}"
    )

    print(header)
    print("─" * WIDTH)

    for index, day in enumerate(timeline, 1):
        date = day.get("date", f"Day {index:02d}")

        print(
            f"{date:<12}"
            f"{fmt_pct(day.get('quality')):>10}"
            f"{fmt_pct(day.get('competency')):>12}"
            f"{fmt_pct(day.get('attendance')):>12}"
            f"{fmt_pct(day.get('release')):>10}"
            f"{fmt_pct(day.get('transfer')):>10}"
            f"{fmt_pct(day.get('operations_health')):>10}"
            f"{fmt_nps(day.get('nps')):>9}"
        )

    print("─" * WIDTH)

    if not timeline:
        print("  No forecast timeline returned.")
        return

    avg = lambda key: (
        sum(
            float(d[key])
            for d in timeline
            if d.get(key) is not None
        )
        / len(
            [
                d for d in timeline
                if d.get(key) is not None
            ]
        )
        if any(d.get(key) is not None for d in timeline)
        else None
    )

    print(
        f"{'AVERAGE':<12}"
        f"{fmt_pct(avg('quality')):>10}"
        f"{fmt_pct(avg('competency')):>12}"
        f"{fmt_pct(avg('attendance')):>12}"
        f"{fmt_pct(avg('release')):>10}"
        f"{fmt_pct(avg('transfer')):>10}"
        f"{fmt_pct(avg('operations_health')):>10}"
        f"{fmt_nps(avg('nps')):>9}"
    )

    print("─" * WIDTH)

    first = timeline[0]
    last = timeline[-1]

    print()
    print("FORECAST SUMMARY")
    print("─" * WIDTH)
    print(
        f"  Starting OH : {fmt_pct(first.get('operations_health'))}"
        f"    Ending OH : {fmt_pct(last.get('operations_health'))}"
    )
    print(
        f"  Starting NPS: {fmt_nps(first.get('nps'))}"
        f"      Ending NPS: {fmt_nps(last.get('nps'))}"
    )

    if result.payload:
        summary = result.payload.get("summary", {})
        if isinstance(summary, dict):
            for key in ("risk_level", "trend", "confidence"):
                if key in summary:
                    print(f"  {key.replace('_', ' ').title():<13}: {summary[key]}")

    print("─" * WIDTH)


def select_model_for_prediction():
    """Prompt the user to select a model family for prediction.

    Lists every complete OH+NPS pair in ``models/`` and loads the
    selected pair into ``PredictorProvider`` so that
    ``ForecastOrchestrator`` / ``PredictionService`` use the chosen
    models.

    If no families exist, the user is told to train first.
    """
    families = list_model_families()

    if not families:
        print()
        print("─" * WIDTH)
        print("No model families available for prediction.")
        print("  Train models first (python train_all_ai.py)")
        print("  Models must be saved as models/{family}_OH.pkl")
        print("  and models/{family}_NPS.pkl")
        print("─" * WIDTH)
        return False

    family = select_model_family()

    try:
        PredictorProvider.load_pair(family)
    except ModelPairError as e:
        print()
        print("─" * WIDTH)
        print(f"Model pair error: {e}")
        print("─" * WIDTH)
        return False
    except FileNotFoundError as e:
        print()
        print("─" * WIDTH)
        print(f"Model file not found: {e}")
        print("─" * WIDTH)
        return False

    return True


def execute_forecast(horizon, scenario, state):
    request = ForecastRequest(
        operation="forecast",
        horizon=horizon,
        scenario=scenario,
        parameters={
            "state": state,
            # Phase 1 rule: only yesterday's actual KPIs are eligible
            # to become actual learning/history input. TSE is forecast
            # initialization only and is never treated as actual history.
            "learning_source": "yesterday_actual",
        }
    )

    engine = ForecastOrchestrator()
    return engine.execute(request)


def get_horizon():
    while True:
        try:
            value = input("\nForecast horizon days:\n> ").strip()

            if not value:
                return 7

            value = int(value)

            if value < 1:
                print("Horizon must be at least 1 day.")
                continue

            return value

        except ValueError:
            print("Enter a valid number of days.")


def get_scenario():
    allowed = {
        "baseline",
        "aep",
        "oep",
        "training",
        "staffing_shortage",
    }

    while True:
        value = input(
            "Scenario (baseline/aep/oep/training/staffing_shortage):\n> "
        ).strip().lower()

        if not value:
            return "baseline"

        if value in allowed:
            return value

        print("Invalid scenario.")


def run_forecast():
    print()
    print("YESTERDAY'S ACTUAL KPIs")
    print("─" * WIDTH)
    print("Enter yesterday's actual values. Press Enter for defaults.")
    print()

    state = {
        "quality": float(input("Quality % [87.0]: ").strip() or "87.0"),
        "competency": float(input("Competency % [93.0]: ").strip() or "93.0"),
        "attendance": float(input("Attendance % [90.0]: ").strip() or "90.0"),
        "release": float(input("Release Rate % [60.0]: ").strip() or "60.0"),
        "transfer": float(input("Transfer Rate % [9.0]: ").strip() or "9.0"),
        "operations_health": float(input("Operational Health % [95.0]: ").strip() or "95.0"),
        "nps": float(input("NPS [82.0]: ").strip() or "82.0"),
    }

    # Yesterday's actual KPIs are the only learning data.
    actual_state = {
        **state,
        "date": "yesterday",
    }

    print()
    print("─" * WIDTH)
    print("TARGET STATE ENGINE")
    print("─" * WIDTH)
    print()
    print("Use TSE output as the forecast starting state?")
    print("1 - Yes, use TSE output")
    print("2 - No, use yesterday's actual KPIs")
    print()

    choice = input("Choice: ").strip()

    if choice == "1":
        print()
        print("TARGET STATE ENGINE OUTPUT")
        print("─" * WIDTH)

        tse_state = get_starting_state()

        if tse_state:
            state = tse_state

        print()
        print("Forecast starting state: TSE output")

    elif choice == "2":
        print()
        print("Forecast starting state: Yesterday's actual KPIs")
        print("The forecast will learn only from these actual KPI inputs.")

    else:
        print()
        print("Invalid choice. Using yesterday's actual KPIs.")
        print("Forecast starting state: Yesterday's actual KPIs")

    print()
    print("Recording yesterday's actual KPIs in ForecastAI...")
    
    try:
        learning_engine = ForecastOrchestrator()
        recorded_actual = learning_engine.update_actual(actual_state)

        # Use the actual record as the forecast starting state.
        # This does not alter ForecastAI formulas; it only prevents
        # forecast-generated OH/NPS from replacing the supplied actuals
        # before the forecast begins.
        state = {
            "quality": recorded_actual.get("quality", state["quality"]),
            "competency": recorded_actual.get("competency", state["competency"]),
            "attendance": recorded_actual.get("attendance", state["attendance"]),
            "release": recorded_actual.get("release", state["release"]),
            "transfer": recorded_actual.get("transfer", state["transfer"]),
            "operations_health": recorded_actual.get(
                "operations_health",
                state["operations_health"],
            ),
            "nps": recorded_actual.get("nps", state["nps"]),
        }

        print("Actual KPI learning input recorded.")
    except Exception as exc:
        print()
        print("ACTUAL KPI LEARNING ERROR")
        print("─" * WIDTH)
        print(f"  {exc}")
        return

    print()
    horizon = get_horizon()
    scenario = "baseline"

    result = execute_forecast(
        horizon,
        scenario,
        state,
    )

    if not result.success:
        print()
        print("FORECAST ERROR")
        print("─" * WIDTH)
        for error in result.errors:
            print(f"  {error}")
        return

    print_forecast_table(
        result,
        "FORECAST — OH / NPS",
        scenario,
    )

def run_scenario():
    print()
    print("▸ SCENARIO FORECAST")
    print("─" * WIDTH)

    state = get_starting_state()
    horizon = get_horizon()
    scenario = get_scenario()

    result = execute_forecast(horizon, scenario, state)

    if not result.success:
        print("\nSCENARIO FORECAST ERROR")
        print("─" * WIDTH)
        for error in result.errors:
            print(f"  {error}")
        return

    print_forecast_table(
        result,
        "SCENARIO FORECAST",
        scenario,
    )


def run_risk():
    print()
    print("▸ RISK ANALYSIS")
    print("─" * WIDTH)

    state = get_starting_state()
    horizon = get_horizon()
    scenario = get_scenario()

    forecast_result = execute_forecast(
        horizon,
        scenario,
        state,
    )

    if not forecast_result.success:
        print("\nUnable to perform risk analysis.")
        for error in forecast_result.errors:
            print(f"  {error}")
        return

    # Use the existing ForecastAI RiskEngine.
    risk_engine = RiskEngine()

    risk_request = ForecastRequest(
        operation="risk",
        parameters={
            "forecast_result": forecast_result.payload,
        },
    )

    risk_result = risk_engine.execute(risk_request)

    if not risk_result.success:
        print("\nRISK ANALYSIS ERROR")
        print("─" * WIDTH)

        for error in risk_result.errors:
            print(f"  {error}")

        return

    payload = risk_result.payload or {}

    print()
    print(f"  Scenario : {scenario}")
    print(f"  Horizon  : {horizon} days")
    print()

    print("RISK OVERVIEW")
    print("─" * WIDTH)

    overall_risk = payload.get("overall_risk")

    if overall_risk is not None:
        if isinstance(overall_risk, float):
            print(f"  Overall Risk : {overall_risk:.1%}")
        else:
            print(f"  Overall Risk : {overall_risk}")
    else:
        print("  Overall Risk : —")

    analyses = payload.get("analyses", [])

    if analyses:
        print()
        print("RISK COMPONENTS")
        print("─" * WIDTH)

        print(
            f"{'Component':<28}"
            f"{'Risk':>12}"
            f"{'Classification':>20}"
        )
        print("─" * WIDTH)

        for analysis in analyses:
            component = analysis.get("component", "Unknown")
            risk = analysis.get("overall_risk")
            classification = analysis.get(
                "classification",
                "—",
            )

            if isinstance(risk, float):
                risk_text = f"{risk:.1%}"
            else:
                risk_text = str(risk or "—")

            print(
                f"{component:<28}"
                f"{risk_text:>12}"
                f"{classification:>20}"
            )

        print("─" * WIDTH)

        print()
        print("RISK FACTORS")
        print("─" * WIDTH)

        factor_count = 0

        for analysis in analyses:
            factors = analysis.get("risk_factors", [])

            for factor in factors:
                name = factor.get("name", "Unknown")
                score = factor.get("risk_score")
                reason = factor.get("reason", "—")
                mitigation = factor.get("mitigation", "—")

                if isinstance(score, float):
                    score_text = f"{score:.2f}"
                else:
                    score_text = str(score or "—")

                print(f"  {name}  [{score_text}]")
                print(f"    Reason    : {reason}")
                print(f"    Mitigation: {mitigation}")
                print()

                factor_count += 1

                if factor_count >= 10:
                    break

            if factor_count >= 10:
                break

    warnings = payload.get("warnings", [])

    if warnings:
        print("WARNINGS")
        print("─" * WIDTH)

        for warning in warnings:
            print(f"  • {warning}")

        print("─" * WIDTH)


def main():
    # Model-family selection must happen before any forecast.  The selected
    # pair is loaded into PredictorProvider so that ForecastOrchestrator /
    # PredictionService use exactly the chosen OH+NPS models.  The user is
    # never given a silent default.
    selected_family = None

    while True:
        print()
        banner("AXIPULSEAI FORECAST AI")

        if selected_family:
            print(f"Model family : {selected_family}")
        else:
            print("Model family : (not selected)")

        print(
            f"""
1 - Select model family
2 - Forecast OH/NPS
3 - Scenario forecast
4 - Risk analysis
5 - Exit
"""
        )

        choice = input("Choice: ").strip()

        if choice == "1":
            if select_model_for_prediction():
                selected_family = PredictorProvider.get_model_family()
            continue

        if choice in ("2", "3", "4"):
            # PREDICT path: the user must have selected a model pair.
            # Never silently fall back to another model.
            if selected_family is None:
                print()
                print("─" * WIDTH)
                print("Please select a model family first (option 1).")
                print("─" * WIDTH)
                continue

        if choice == "2":
            run_forecast()

        elif choice == "3":
            run_scenario()

        elif choice == "4":
            run_risk()

        elif choice == "5":
            print("Exit")
            break

        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()

# Module-level compatibility surface
get_state = get_starting_state
