#!/usr/bin/env python3

import itertools
import joblib
import pandas as pd
from pathlib import Path


MODEL = Path("models/operation_health_predictor.joblib")
DATA = Path("training/training.csv")


def load_latest_state():

    df = pd.read_csv(DATA)
    row = df.tail(1).iloc[0]

    return {
        "quality": float(row["actual_quality"]),
        "competency": float(row["actual_competency"]),
        "attendance": float(row["actual_attendance"]),
        "release": float(row["actual_release_rate"]),
        "transfer": float(row["actual_transfer_rate"]),
        "total_calls_received": float(row["total_calls_received"]),
        "operational_health": float(row["operational_health"]),
    }


def load_model():

    model = joblib.load(MODEL)

    # handle wrapped models
    if isinstance(model, dict):
        for k in [
            "model",
            "predictor",
            "pipeline",
            "estimator"
        ]:
            if k in model:
                return model[k]

    return model


def predict(model,state):

    df=pd.DataFrame([state])

    return float(model.predict(df)[0])


def solve_for(target: float) -> dict:
    """Reverse-optimise the KPIs that drive a target Operational Health.

    Returns a dict with the best KPI combination found, the predicted OH and
    the search distance. Reusable from the GUI and the CLI (``solve``).
    """
    model=load_model()

    base=load_latest_state()

    best=None
    distance_best=999999


    print("\nSearching KPI combinations...\n")


    for q,c,r,t,a in itertools.product(
        range(60,101,2),
        range(55,101,2),
        range(50,101,2),
        range(0,21,2),
        range(65,101,2)
    ):

        state=base.copy()

        state.update({
            "quality":q,
            "competency":c,
            "release":r,
            "transfer":t,
            "attendance":a
        })


        try:
            oh=predict(model,state)
        except Exception:
            continue


        distance=abs(oh-target)


        if distance < distance_best:

            distance_best=distance

            best={
                "predicted_OH":oh,
                "quality":q,
                "competency":c,
                "release":r,
                "transfer":t,
                "attendance":a,
                "calls":state["total_calls_received"]
            }

    if best is None:
        return {"found": False, "distance": None, "error": "No solution found"}

    result = dict(best)
    result["found"] = True
    result["distance"] = round(distance_best, 3)
    result["target"] = target
    return result


def solve():

    target=float(
        input("Target OH: ")
    )

    best = solve_for(target)

    print("==============================")
    print("REQUIRED ACTUALS")
    print("==============================")

    if best.get("found"):
        for k,v in best.items():
            print(f"{k:25}: {v}")
    else:
        print("No solution found")


if __name__=="__main__":
    solve()
