C = {}

from pathlib import Path
import pandas as pd

from core.operation_health_predictor import OperationalHealthPredictor
from core.nps_predictor import NPSPredictor
from core.forecast_ai.prediction.model_selector import (
    list_training_files,
    list_model_families,
)


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


BASE = Path("training")

# Globals populated by scan_files() (kept for backward-compat surface)
oh_files = []
nps_files = []


# ============================================================
# UNIVERSAL TRAINING SCHEMA
# Order follows simulator causal flow
# ============================================================


# ------------------------------------------------------------
# 1. TARGETS
# ------------------------------------------------------------

TARGET_COLUMNS = {
    "target_quality",
    "target_competency",
    "target_attendance",
    "target_release_rate",
    "target_transfer_rate",
}


# ------------------------------------------------------------
# 2. ACTUAL OPERATIONAL RESULTS
# ------------------------------------------------------------

ACTUAL_COLUMNS = {
    "operational_health",

    "actual_quality",
    "actual_competency",
    "actual_attendance",

    "actual_release_rate",
    "actual_transfer_rate",
}


# ------------------------------------------------------------
# 3. AGENT / INTELLIGENCE FACTORS
# ------------------------------------------------------------

FACTOR_COLUMNS = {
    "operational_intelligence_factor",
    "business_intelligence_factor",
    "member_intelligence_factor",
}


# ------------------------------------------------------------
# 4. WORKLOAD / CALL VOLUME
# ------------------------------------------------------------

CALL_COLUMNS = {
    "total_calls_received",
}


# ------------------------------------------------------------
# 5. SURVEY + NPS DISTRIBUTION
# ------------------------------------------------------------

SURVEY_COLUMNS = {
    "total_surveys",
    "survey_rate",

    "promoters",
    "passives",
    "detractors",

    "score_0",
    "score_1",
    "score_2",
    "score_3",
    "score_4",
    "score_5",
    "score_6",
    "score_7",
    "score_8",
    "score_9",
    "score_10",
}


# ============================================================
# ENGINE REQUIREMENTS
# ============================================================

OH_REQUIRED = (
    TARGET_COLUMNS
    | ACTUAL_COLUMNS
    | FACTOR_COLUMNS
    | CALL_COLUMNS
)


NPS_REQUIRED = (
    SURVEY_COLUMNS
)



def scan_files():
    """Scan ALL training files for OH and NPS column compatibility.

    Uses ``list_training_files()`` to discover every file in the
    training directory, then checks the header row of each against
    ``OH_REQUIRED`` and ``NPS_REQUIRED``.

    Populates the module-level ``oh_files`` and ``nps_files`` lists
    (kept for backward compatibility) and returns ``(oh_files, nps_files)``.
    """
    global oh_files, nps_files
    oh_files = []
    nps_files = []

    for f in list_training_files():
        try:
            df = pd.read_csv(f, nrows=5)
            cols = set(df.columns)
            if OH_REQUIRED.issubset(cols):
                oh_files.append(f)
            if NPS_REQUIRED.issubset(cols):
                nps_files.append(f)
        except Exception:
            pass

    return oh_files, nps_files


def describe_dataset(path):
    """Print row count and date range for a dataset file."""
    try:
        df = pd.read_csv(path, usecols=lambda c: c == "date")
        n_rows = len(df)
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"], errors="coerce")
            dates = dates.dropna()
            if len(dates):
                print(f"     {C.DIM}rows: {n_rows:,}   date range: {dates.min().date()} → {dates.max().date()}{C.RESET}")
            else:
                print(f"     {C.DIM}rows: {n_rows:,}   date range: unavailable{C.RESET}")
        else:
            print(f"     {C.DIM}rows: {n_rows:,}{C.RESET}")
    except Exception as e:
        print(f"     {C.YELLOW}(could not read dataset summary: {e}){C.RESET}")


def confirm(prompt):
    ans = input(f"{prompt} [y/n]: ").strip().lower()
    return ans in ("y", "yes")


# Module-level: initialise globals only (no side-effects at import time)
oh_files, nps_files = [], []


def main():
    """V2 training workflow.

    Lists ALL files in the ``training/`` directory, lets the user choose
    ONE dataset, and trains **both** OH and NPS models from that same
    file.  Output models are saved using the V2 naming convention:

        models/{stem}_OH.pkl
        models/{stem}_NPS.pkl

    where ``{stem}`` is the selected training filename stem.  If the
    same dataset is trained again, the existing pair is **replaced**
    (no duplicate versions are created).
    """
    # ----------------------------------------------------------
    # Step 1 — List ALL training files
    # ----------------------------------------------------------
    files = list_training_files()

    if not files:
        print(f"\n{C.RED}❌ No training files found in {BASE}/{C.RESET}")
        return

    # ----------------------------------------------------------
    # Step 2 — Show filename + extension, let user choose ONE
    # ----------------------------------------------------------
    section("AVAILABLE TRAINING DATASETS")
    print(f"{C.BOLD}Filename + Extension:{C.RESET}")
    print("─" * 70)
    for i, f in enumerate(files):
        print(f"  {C.GREEN}[{i}]{C.RESET}  {f.name}")
        describe_dataset(f)
    print("─" * 70)

    while True:
        raw = input(
            f"\n{C.BOLD}Select a dataset (0-{len(files) - 1}): {C.RESET}"
        ).strip()
        try:
            idx = int(raw)
        except ValueError:
            print("Enter a valid number.")
            continue
        if 0 <= idx < len(files):
            selected = files[idx]
            break
        print(f"Enter a number between 0 and {len(files) - 1}.")

    # ----------------------------------------------------------
    # Validate the selected dataset has both OH + NPS columns
    # ----------------------------------------------------------
    try:
        df = pd.read_csv(selected, nrows=5)
        cols = set(df.columns)
    except Exception as e:
        print(f"\n{C.RED}❌ Cannot read dataset: {e}{C.RESET}")
        return

    oh_ok = OH_REQUIRED.issubset(cols)
    nps_ok = NPS_REQUIRED.issubset(cols)

    if not oh_ok:
        print(f"\n{C.RED}❌ Selected dataset lacks OH-required columns.{C.RESET}")
        return
    if not nps_ok:
        print(f"\n{C.RED}❌ Selected dataset lacks NPS-required columns.{C.RESET}")
        return

    # ----------------------------------------------------------
    # Step 3 — Model family = selected filename stem
    # ----------------------------------------------------------
    model_family = selected.stem
    oh_path = f"models/{model_family}_OH.pkl"
    nps_path = f"models/{model_family}_NPS.pkl"

    print(f"\n{C.CYAN}Training dataset : {selected.name}{C.RESET}")
    print(f"{C.CYAN}Model family      : {model_family}{C.RESET}")
    print(f"{C.CYAN}Output (OH)       : {oh_path}{C.RESET}")
    print(f"{C.CYAN}Output (NPS)      : {nps_path}{C.RESET}")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # The same dataset trains BOTH models.  joblib.dump overwrites the
    # target path, so re-training the same dataset replaces the pair
    # automatically — no duplicate versions are created.
    if not confirm("Proceed with training both models?"):
        print(f"{C.YELLOW}Training cancelled.{C.RESET}")
        return

    # ----------------------------------------------------------
    # Step 4 — Train OH
    # ----------------------------------------------------------
    section("[1/2] TRAINING OH")
    print(f"Dataset: {C.CYAN}{selected}{C.RESET}")

    oh = OperationalHealthPredictor()
    oh.train(str(selected))
    oh.save_model(oh_path)
    print(f"{C.GREEN}✅ OH training complete (saved to {oh_path}){C.RESET}")

    # ----------------------------------------------------------
    # Step 5 — Train NPS (same dataset)
    # ----------------------------------------------------------
    section("[2/2] TRAINING NPS")
    print(f"Dataset: {C.CYAN}{selected}{C.RESET}")

    nps = NPSPredictor()
    nps.train(str(selected))
    nps.save_model(nps_path)
    print(f"{C.GREEN}✅ NPS training complete (saved to {nps_path}){C.RESET}")

    print()


def show_leaderboard():
    """Display the model-family leaderboard.

    Lists every complete model family (those with both _OH.pkl and
    _NPS.pkl) found in the models/ directory.
    """
    import joblib

    print()
    banner("🏆 AXIPULSEAI MODEL LEADERBOARD")

    families = list_model_families()

    if not families:
        print("\n" + C.YELLOW + "No complete model pairs found." + C.RESET)
        print(C.DIM + "Train models first to see them here." + C.RESET)
        return

    print("\n" + C.BOLD + "Complete model families:" + C.RESET)
    print("-" * 70)

    for fam in families:
        oh_file = "models/" + fam + "_OH.pkl"
        nps_file = "models/" + fam + "_NPS.pkl"

        print("\n" + C.GREEN + "▸ " + fam + C.RESET)
        print("  OH  : " + oh_file)
        print("  NPS : " + nps_file)

        for label, f_path in [("OH", oh_file), ("NPS", nps_file)]:
            try:
                data = joblib.load(f_path)
                print("  " + label + " model        : " + str(data.get("model_name", "unknown")))
                perf = data.get("algorithm_performance", {})
                if perf:
                    print("  Algorithm perf   :")
                    for model, metrics in perf.items():
                        print("    -> " + str(model) + ": " + str(metrics))
                else:
                    print("  Algorithm perf   : (none)")
                print("  Features         : " + str(len(data.get("feature_names", []))))
                print("  Training days    : " + str(data.get("history_days")))
            except Exception as e:
                print("  " + label + " model load error: " + str(e))

    print("-" * 70)


if __name__ == "__main__":
    main()
    show_leaderboard()
    banner("✅  DONE  ✅")
