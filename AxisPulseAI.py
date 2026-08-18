#!/usr/bin/env python3
"""
AxisPulseAI — unified launcher for training, prediction, and (later) forecasting.

This wraps the existing, working scripts instead of reimplementing their
internals, so behavior stays identical to running them directly:
  - train_all_ai.py   (training)
  - predict_cli.py     (OH / NPS prediction via CLI subcommands)
  - predict_manual.py  (interactive NPS prediction)
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

OH_MODEL = "models/operation_health_predictor.joblib"
NPS_MODEL = "models/nps_predictor_model.pkl"


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"


def banner(text):
    width = 80
    print(f"{C.CYAN}{C.BOLD}{'═'*width}{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}{text.center(width)}{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}{'═'*width}{C.RESET}")


def section(text):
    print(f"\n{C.MAGENTA}{C.BOLD}▸ {text}{C.RESET}")
    print(f"{C.DIM}{'─'*80}{C.RESET}")



def ask(name, default):
    """Read a numeric value with a default."""
    value = input(f"{name} [{default}]: ").strip()
    return float(value) if value else default


def run(cmd):
    """Run a subprocess, streaming its output live, from the project root."""
    print(f"{C.DIM}$ {' '.join(cmd)}{C.RESET}\n")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"{C.RED}⚠️  Command exited with code {result.returncode}{C.RESET}")


def model_exists(path):
    return (PROJECT_ROOT / path).exists()


def do_train():
    section("TRAIN")
    run([sys.executable, "train_all_ai.py"])


def do_predict():
    section("PREDICT")
    print(f" {C.GREEN}1{C.RESET} - Predict Operational Health (predict_cli.py)")
    print(f" {C.GREEN}2{C.RESET} - Predict NPS, combined engine (predict_cli.py)")
    print(f" {C.GREEN}3{C.RESET} - Back to main menu")

    choice = input(f"\n{C.BOLD}Choice: {C.RESET}").strip()

    if choice == "1":
        if not model_exists(OH_MODEL):
            print(f"{C.RED}❌ OH model not found at {OH_MODEL}. Train first.{C.RESET}")
            return
        run([sys.executable, "predict_cli.py", "predict", "--model", OH_MODEL])

    elif choice == "2":
        _run_universal_probabilistic_prediction()


    elif choice == "3":
        if not model_exists(NPS_MODEL):
            print(f"{C.RED}❌ NPS model not found at {NPS_MODEL}. Train first.{C.RESET}")
            return
        run([sys.executable, "predict_manual.py"])

    elif choice == "4":
        return

    else:
        print(f"{C.YELLOW}Unrecognized choice.{C.RESET}")


def do_forecast():
    section("FORECAST")

    run([sys.executable, "forecast_cli.py"])


def do_reverse():
    section("REVERSE OPTIMIZATION")

    try:
        from core.forecast_ai.engines.reverse_optimizer import ReverseOptimizer

        print("Target OH (blank skip):")
        target_oh = input("> ").strip()

        print("Target NPS (blank skip):")
        target_nps = input("> ").strip()

        print("Deadline days:")
        deadline = input("> ").strip()

        from core.forecast_ai.models import ForecastRequest

        engine = ReverseOptimizer()

        request = ForecastRequest(
            operation="reverse_optimize",
            parameters={
                "target_oh": float(target_oh) if target_oh else None,
                "target_nps": float(target_nps) if target_nps else None,
                "deadline_days": int(deadline) if deadline else 30,
                "priority": "balanced",
                "state": __import__("pandas").read_csv(
                    "training/training.csv"
                ).tail(1).pipe(
                    lambda df: {
                        "operational_health": float(df.iloc[0]["operational_health"]),
                        "quality": float(df.iloc[0]["actual_quality"]),
                        "competency": float(df.iloc[0]["actual_competency"]),
                        "attendance": float(df.iloc[0]["actual_attendance"]),
                        "release": float(df.iloc[0]["actual_release_rate"]),
                        "transfer": float(df.iloc[0]["actual_transfer_rate"]),
                    }
                )
            }
        )

        print()
        print("Running reverse optimizer...")
        print("-" * 80)

        result = engine.execute(request)

        if result.payload:
            best = result.payload.get("best_solution")
            print()
            print("REVERSE OPTIMIZATION RESULT")
            print("=" * 80)

            if best is None:
                print("No valid operational state could be evaluated.")
                return

            if result.success:
                print("Status: Target reached within the configured tolerance.")
            else:
                print("Status: Target is not reachable with the current model and allowed changes.")
                print("Showing the closest operational state found instead.")

            print()
            print(f"Closest predicted OH:  {best['predicted_operations_health']:.2f}")
            print(f"Closest predicted NPS: {best['predicted_nps']:.2f}")
            print(f"Remaining target gap:  {best['distance_to_target']:.2f}")

            print()
            print("Recommended changes:")
            changes = best.get("state_changes", {})
            meaningful_changes = [
                (name, value)
                for name, value in changes.items()
                if value != 0
            ]
            if meaningful_changes:
                for name, value in meaningful_changes:
                    sign = "+" if value > 0 else ""
                    print(f" - {name.title()}: {sign}{value:g}")
            else:
                print(" - No beneficial change was found within the configured search limits.")

            print()
            print("Closest operational state:")
            for name, value in best.get("state", {}).items():
                print(f" - {name.title()}: {value:.2f}")

            if result.errors:
                print()
                print("Note:", result.errors[0])

    except Exception as e:
        print(f"{C.RED}Reverse optimizer failed:{C.RESET}")
        print(e)



def do_target_state():
    section("TARGET STATE ENGINE")

    print(f"{C.CYAN}🧠 AxiPulseAI Target State Engine{C.RESET}")

    target_oh = input("Target OH (blank skip): ").strip()
    target_nps = input("Target NPS (blank skip): ").strip()

    try:
        from core.target_state_engine import TargetStateEngine

        engine = TargetStateEngine()

        print()
        print("Searching required operational state...")
        print("-" * 80)

        targets = {}
        if target_oh:
            targets["operational_health"] = float(target_oh)
        if target_nps:
            targets["nps"] = float(target_nps)
        if not targets:
            print("Enter at least one target before starting the search.")
            return

        result = engine.find_target_state(targets=targets)

        if result:
            print()
            print("TARGET STATE RESULT")
            print("=" * 80)

            consensus = result["consensus"]
            print("Predicted OH:", round(consensus["oh"], 3))
            print("Predicted NPS:", round(consensus["nps"], 3))

            print()
            print("Recommended operational state:")

            for k, v in result["recommended_state"].items():
                print(f"{k.replace('_', ' ').title():20}: {v:.2f}")

            print()
            print("Distance:", round(result["distance"],3))

        else:
            print("No reachable target state found")

    except Exception as e:
        print(f"{C.RED}Target State Engine failed:{C.RESET}")
        print(e)


def do_surprise():
    section("SURPRISE")

    print(f"{C.YELLOW}🧠 ADIE Decision Intelligence (V3){C.RESET}")
    print("-" * 80)

    try:
        from core.decision_intelligence.v3.integration.probabilistic_decision import (
            ProbabilisticDecisionService,
        )

        service = ProbabilisticDecisionService()

        print("Enter operational state:")
        print()

        def ask(name, default):
            value = input(f"{name} [{default}]: ").strip()
            return float(value) if value else default

        oh = ask("Operations Health", 82)
        competency = ask("Competency", 88)
        quality = ask("Quality", 85)

        observations = [
            min(1.0, max(0.0, oh / 100.0)),
            min(1.0, max(0.0, quality / 100.0)),
            min(1.0, max(0.0, competency / 100.0)),
        ]

        print()
        print("Generating ADIE decision...")
        print("-" * 80)

        package = service.analyze(
            scenarios=[{"name": "current_state", "expected": oh / 100.0}],
            observations=observations,
            baseline=oh / 100.0,
            samples=5000,
        )

        print()
        print("=" * 80)
        print("ADIE DECISION RESULT")
        print("=" * 80)

        print("Recommendation:", package.recommendation)
        print("Risk:", package.risk)
        print("Probability:", f"{package.probability:.3f}")
        print("Confidence:", f"{package.confidence:.3f}")
        print("Expected:", f"{package.expected:.3f}")
        print("Downside (p05):", f"{package.downside:.3f}")
        print("Upside (p95):", f"{package.upside:.3f}")

        print("=" * 80)

    except Exception as e:
        print(f"{C.RED}ADIE failed:{C.RESET}")
        print(e)

    print(
        f"{C.DIM}Future AI module reserved.{C.RESET}"
    )


def main_menu():
    while True:
        banner("🧠  AXISPULSEAI  🧠")
        print(f" {C.GREEN}1{C.RESET} - Train")
        print(f" {C.GREEN}2{C.RESET} - Predict")
        print(f" {C.GREEN}3{C.RESET} - Forecast OH/NPS")
        print(f" {C.GREEN}4{C.RESET} - Reverse Optimization")
        print(f" {C.GREEN}5{C.RESET} - Target State Engine")
        print(f" {C.GREEN}6{C.RESET} - 👩 ADIE Decision Intelligence")
        print(f" {C.GREEN}7{C.RESET} - Exit")

        choice = input(f"\n{C.BOLD}Choice: {C.RESET}").strip()

        if choice == "1":
            do_train()
        elif choice == "2":
            _run_universal_probabilistic_prediction()
        elif choice == "3":
            do_forecast()

        elif choice == "4":
            do_reverse()

        elif choice == "5":
            do_target_state()

        elif choice == "6":
            do_surprise()

        elif choice == "7":
            print(f"{C.CYAN}Goodbye.{C.RESET}")
            break
        else:
            print(f"{C.YELLOW}Unrecognized choice.{C.RESET}")




def _run_universal_probabilistic_prediction():
    """
    AxiPulseAI universal production prediction surface.

    Uses the existing production prediction pipeline.
    Does not train, mutate data, replace models, or modify
    predictor logic.
    """
    from core.forecast_ai.prediction import predict_production

    print()
    print("=" * 70)
    print(" AXIPULSEAI — UNIVERSAL PROBABILISTIC PREDICTION")
    print("=" * 70)

    def ask_float(label, default):
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return float(default)
        return float(raw)

    print()
    print("Enter today's operational state.")
    print("Press Enter to use the displayed default.")
    print()

    quality = ask_float("Quality", 87.0)
    competency = ask_float("Competency", 93.0)
    attendance = ask_float("Attendance", 90.0)
    release = ask_float("Release Rate", 60.0)
    transfer = ask_float("Transfer Rate", 9.0)
    calls = ask_float("Total Calls", 2000.0)

    state = {
        "quality": quality,
        "competency": competency,
        "attendance": attendance,
        "release": release,
        "transfer": transfer,
        "calls": int(calls),
        "operations_health": 90.0,
        "nps": 82.0,
    }

    print()
    print("Running existing production prediction pipeline...")
    print("Bayesian + Monte Carlo + uncertainty enabled.")
    print()

    result = predict_production(
        state,
        metadata={
            "runtime_trial": True,
            "source": "AxisPulseAI",
        },
        operations_health_uncertainty=0.05,
        simulations=1000,
        seed=0,
    )

    prediction = result.prediction

    def field(obj, *names, default=None):
        for name in names:
            value = getattr(obj, name, None)
            if value is not None:
                return value
        return default

    def display(title, envelope, precision=2):
        probabilistic = envelope.probabilistic
        bayesian = getattr(probabilistic, "bayesian", None)
        monte_carlo = getattr(probabilistic, "monte_carlo", None)
    
        def value(obj, name, default=None):
            if obj is None:
                return default

            # getattr() requires a string attribute name.
            # Runtime Monte Carlo fields are named attributes.
            if not isinstance(name, str):
                return default

            if isinstance(obj, dict):
                return obj.get(name, default)

            return getattr(obj, name, default)
    
        def first(obj, names, default=None):
            for name in names:
                result = value(obj, name, None)
                if result is not None:
                    return result
    
            # Some Monte Carlo results expose percentiles as a mapping.
            percentiles = value(obj, "percentiles", None)
            if isinstance(percentiles, dict):
                for name in names:
                    if name in percentiles:
                        return percentiles[name]
    
            return default
    
        prediction = value(envelope, "prediction")
        most_likely = value(probabilistic, "most_likely")
        expected = value(probabilistic, "expected_value")
        uncertainty = value(probabilistic, "uncertainty")
    
        posterior_mean = first(
            bayesian,
            ("posterior_mean", "mean"),
            0.5,
        )
    
        lower = first(
            bayesian,
            (
                "credible_interval_lower",
                "lower",
                "lower_bound",
            ),
            posterior_mean,
        )
    
        upper = first(
            bayesian,
            (
                "credible_interval_upper",
                "upper",
                "upper_bound",
            ),
            posterior_mean,
        )
    
        simulations = first(
            monte_carlo,
            ("simulations", "n_simulations", "num_simulations"),
            1000,
        )
    
        p05 = first(
            monte_carlo,
            (
                "p05",
                "P05",
                "p_05",
                "q05",
                "q_05",
                "percentile_05",
                "percentile05",
                "lower_5",
                "lower_5_percentile",
                "percentile_5",
            ),
            None,
        )
    
        p50 = first(
            monte_carlo,
            (
                "p50",
                "P50",
                "p_50",
                "q50",
                "q_50",
                "percentile_50",
                "percentile50",
                "median",
            ),
            most_likely,
        )
    
        p95 = first(
            monte_carlo,
            (
                "p95",
                "P95",
                "p_95",
                "q95",
                "q_95",
                "percentile_95",
                "percentile95",
                "upper_95",
                "upper_95_percentile",
                "percentile_95",
            ),
            None,
        )
    
        print()
        print("=" * 72)
        print(title.upper())
        print("=" * 72)
    
        print()
        print(f"Prediction       : {float(prediction):.2f}")
        print("  What AxiPulseAI expects to happen.")
    
        print(f"Most likely      : {float(most_likely):.2f}")
        print("  The outcome most likely to occur.")
    
        print(f"Expected value   : {float(expected):.2f}")
        print("  The average outcome across possible results.")
    
        print(f"Uncertainty      : {float(uncertainty):.4f}")
        print("  How much the prediction may vary.")
    
        print()
        print("BAYESIAN ANALYSIS")
        print("  What it does: Updates the prediction using available evidence.")
        print(f"  Posterior probability : {float(posterior_mean):.4f}")
        print(f"  Likely range          : {float(lower):.4f} → {float(upper):.4f}")
    
        print()
        print("MONTE CARLO ANALYSIS")
        print("  What it does: Tests many possible scenarios to estimate the range.")
        print(f"  Scenarios tested      : {int(simulations):,}")
    
        if p05 is not None:
            print(f"  Lower-end scenario    : {float(p05):.2f}")
    
        if p50 is not None:
            print(f"  Middle scenario       : {float(p50):.2f}")
    
        if p95 is not None:
            print(f"  Upper-end scenario    : {float(p95):.2f}")
    
        print("  Lower-end = P05 | Middle = P50 | Upper-end = P95")

    print("=" * 70)
    display(
        "Operations Health",
        prediction.operations_health,
        4,
    )

    _show_nps_distribution(prediction)

    input("\nPress Enter to return to AxiPulseAI...")


def _show_nps_distribution(prediction):
    """
    CLI-only presentation of the existing 0-10 NPS distribution.

    Does not:
    - recalculate NPS
    - modify Bayesian inference
    - modify Monte Carlo
    - modify the 0-10 distribution
    - optimize the scalar NPS
    """

    distribution = getattr(
        prediction,
        "bayesian_score_distribution",
        None,
    )

    score_counts = getattr(
        prediction,
        "score_counts",
        None,
    )

    def get_score(mapping, score, default=0.0):
        if not isinstance(mapping, dict):
            return default

        key = f"score_{score}"

        if key in mapping:
            return mapping[key]

        if str(score) in mapping:
            return mapping[str(score)]

        if score in mapping:
            return mapping[score]

        return default

    counts = [
        int(get_score(score_counts, score, 0))
        for score in range(11)
    ]

    probabilities = [
        float(get_score(distribution, score, 0.0))
        for score in range(11)
    ]

    total_surveys = sum(counts)

    if total_surveys > 0:
        count_percentages = [
            (count / total_surveys) * 100.0
            for count in counts
        ]
    else:
        count_percentages = [0.0] * 11

    bayesian_total = sum(probabilities)

    if bayesian_total > 0:
        probabilities = [
            (value / bayesian_total) * 100.0
            for value in probabilities
        ]
    else:
        probabilities = [0.0] * 11

    detractors = sum(counts[0:7])
    passives = sum(counts[7:9])
    promoters = sum(counts[9:11])

    detractor_pct = (
        detractors / total_surveys * 100.0
        if total_surveys
        else 0.0
    )

    passive_pct = (
        passives / total_surveys * 100.0
        if total_surveys
        else 0.0
    )

    promoter_pct = (
        promoters / total_surveys * 100.0
        if total_surveys
        else 0.0
    )

    nps_envelope = getattr(
        prediction,
        "nps",
        None,
    )

    nps_value = getattr(
        nps_envelope,
        "prediction",
        None,
    )

    if nps_value is None:
        nps_value = (
            ((promoters - detractors) / total_surveys) * 100.0
            if total_surveys
            else 0.0
        )

    print()
    print("=" * 72)
    print("NPS — 0-10 SCORE DISTRIBUTION")
    print("=" * 72)
    print()
    print(
        f"{'Score':>7} | "
        f"{'Responses':>10} | "
        f"{'Response %':>11} | "
        f"{'Bayesian %':>11}"
    )
    print("-" * 72)

    for score in range(11):
        print(
            f"{score:>7} | "
            f"{counts[score]:>10} | "
            f"{count_percentages[score]:>10.2f}% | "
            f"{probabilities[score]:>10.2f}%"
        )

    print("-" * 72)
    print(
        f"{'TOTAL':>7} | "
        f"{total_surveys:>10} | "
        f"{100.00:>10.2f}% | "
        f"{sum(probabilities):>10.2f}%"
    )

    print()
    print("NPS CLASSIFICATION")
    print("-" * 72)
    print(
        f"Promoters   (9-10) : "
        f"{promoters:>6}  ({promoter_pct:>6.2f}%)"
    )
    print(
        f"Passives    (7-8)  : "
        f"{passives:>6}  ({passive_pct:>6.2f}%)"
    )
    print(
        f"Detractors   (0-6) : "
        f"{detractors:>6}  ({detractor_pct:>6.2f}%)"
    )
    print("-" * 72)
    print(
        f"Total Surveys      : "
        f"{total_surveys:>6}"
    )
    print(
        f"NPS                : "
        f"{float(nps_value):>6.2f}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main_menu()
