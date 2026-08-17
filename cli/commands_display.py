import shutil

def cprint(text=""):
    """Print text centered in the terminal."""
    width = shutil.get_terminal_size().columns
    print(text.center(width))

def left_print(text=""):
    """Print text left-aligned."""
    print(text)

# ---- Dashboard rendering ----
def render_full_dashboard(ops_predictor, nps_predictor, defaults, ops_score, nps_pred, nps_score, row):
    """Print the full AxiPulseAI dashboard."""
    width = shutil.get_terminal_size().columns

    # ---- Header (centered) ----
    cprint("╔" + "═"*66 + "╗")
    cprint("║                           AxiPulseAI                               ║")
    cprint("║                Healthcare CX Intelligence Engine                   ║")
    cprint("╚" + "═"*66 + "╝")
    print()

    # ---- System Status (left-aligned) ----
    left_print("SYSTEM STATUS")
    left_print("──────────────────────────────────────────────────────────────────────")
    left_print(f"  Health Engine      : {ops_predictor.model_name} v{ops_predictor.metadata.get('engine_version', '10.10')}")
    left_print(f"  NPS Engine         : {nps_predictor.model_name} v{nps_predictor.metadata.get('engine_version', '2.1.0')}")
    left_print(f"  Training Records   : {ops_predictor.history_days:,}")
    left_print(f"  Status             : ONLINE")
    left_print(f"  Date               : {datetime.now().strftime('%Y-%m-%d')}")
    left_print("")
    left_print("════════════════════════════════════════════════════════════════════════")

    # ---- Target Profile (left-aligned) ----
    left_print("")
    left_print("TARGET PROFILE")
    left_print("──────────────────────────────────────────────────────────────────────")
    left_print(f"  Quality            : {defaults.get('target_quality', 70):.1f}%")
    left_print(f"  Competency         : {defaults.get('target_competency', 70):.1f}%")
    left_print(f"  Attendance         : {defaults.get('target_attendance', 70):.1f}%")
    left_print(f"  Release Rate       : {defaults.get('target_release_rate', 55):.1f}%")
    left_print(f"  Transfer Rate      : {defaults.get('target_transfer_rate', 12):.1f}%")
    left_print("")
    left_print("════════════════════════════════════════════════════════════════════════")

    # ---- Today's Performance (left-aligned) ----
    left_print("")
    left_print("TODAY'S PERFORMANCE")
    left_print("──────────────────────────────────────────────────────────────────────")
    left_print(f"  Quality            : {row.get('actual_quality', 0):.1f}%")
    left_print(f"  Competency         : {row.get('actual_competency', 0):.1f}%")
    left_print(f"  Attendance         : {row.get('actual_attendance', 0):.1f}%")
    left_print(f"  Release Rate       : {row.get('actual_release_rate', 0):.1f}%")
    left_print(f"  Transfer Rate      : {row.get('actual_transfer_rate', 0):.1f}%")
    left_print(f"  Calls Handled      : {row.get('total_calls_received', 0):,}")
    left_print("")
    left_print("════════════════════════════════════════════════════════════════════════")

    # ---- Forecast (branding and numbers centered, details left) ----
    left_print("")
    cprint("   TOMORROW'S NPS FORECAST")
    left_print("")
    left_print("  Operational Health Score")
    cprint(f"                    {ops_score:.2f}%")
    left_print("")
    left_print("  Predicted NPS")
    cprint(f"                    {nps_score:.1f}%")
    left_print("")
    left_print("──────────────────────────────────────────────────────────────────────")
    left_print("")
    left_print("")
    # Format with thousands separators
    promoters = int(nps_pred['promoters'])
    passives = int(nps_pred['passives'])
    detractors = int(nps_pred['detractors'])
    left_print(f"      Promoters     : {promoters:,}      CI: —")
    left_print(f"      Passives      : {passives:,}      CI: —")
    left_print(f"      Detractors    : {detractors:,}      CI: —")
    left_print("")
    left_print("──────────────────────────────────────────────────────────────────────")
    left_print("")
    cprint("Powered by AxiPulseAI • Version 2.1")
    print()
