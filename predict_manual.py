from core.nps_predictor import NPSPredictor
from core.forecast_ai.prediction.model_selector import MODELS_DIR, NPS_LEGACY


def main():
    # Locate the canonical legacy NPS model relative to the project root
    # (portable — independent of the current working directory).
    model_path = str(MODELS_DIR / NPS_LEGACY)
    p = NPSPredictor()
    p.load_model(model_path)
    print("\n=== AxiPulseAI NPS Manual Prediction ===\n")

    row={
    "call_volume": float(input("Call volume: ")),
    "quality": float(input("Quality %: ")),
    "competency": float(input("Competency %: ")),
    "release": float(input("Release %: ")),
    "transfer": float(input("Transfer %: ")),
    "attendance": float(input("Attendance %: ")),
    "complexity": input("Complexity (Low/Medium/High/Very High/Critical): "),
    "event": input("Event: "),
}

    print("\nPredicting...\n")

    result=p.predict(row)

    print("=== RESULT ===")
    for k,v in result.items():
        print(k,":",v)


if __name__ == "__main__":
    main()
