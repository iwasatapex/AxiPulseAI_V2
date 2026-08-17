import pandas as pd
from pathlib import Path
from datetime import datetime


SIM_FILE = Path("training/healthcare_cx_data.csv")


def main():

    TRAIN_FILE = Path("training/training.csv")


    print("="*70)
    print("AXIPULSEAI TRAINING DATA INGESTION")
    print("="*70)


    if not SIM_FILE.exists():
        raise FileNotFoundError(SIM_FILE)


    sim = pd.read_csv(SIM_FILE)

    print("Simulation rows:", len(sim))


    # Convert simulator schema -> universal training schema

    sim["actual_quality"] = sim["quality_kpi"]
    sim["actual_competency"] = sim["competency"]
    sim["actual_attendance"] = sim["attendance"]
    sim["actual_transfer_rate"] = sim["transfer_rate"]

    sim["target_transfer_rate"] = sim["target_transfer"]


    # Keep only model fields

    columns = [
        "date",
        "operational_health",

        "actual_quality",
        "actual_competency",
        "actual_attendance",
        "actual_transfer_rate",

        "target_quality",
        "target_competency",
        "target_attendance",
        "target_transfer_rate",

        "operational_intelligence_factor",

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
    ]


    # score columns may not exist in simulator output
    if not all(c in sim.columns for c in columns):

        print("Missing columns from simulation:")
        for c in columns:
            if c not in sim.columns:
                print(" -",c)

        raise SystemExit(
            "Simulation output cannot be appended"
        )


    new = sim[columns].copy()


    # Add source marker
    new["ingested_at"] = datetime.now().isoformat()


    if TRAIN_FILE.exists():

        old = pd.read_csv(TRAIN_FILE)

        print("Existing training rows:",len(old))

        combined = pd.concat(
            [old,new],
            ignore_index=True
        )

    else:

        combined=new


    # remove exact duplicates
    before=len(combined)

    combined.drop_duplicates(
        subset=[
            "date",
            "operational_health",
            "actual_quality",
            "actual_competency"
        ],
        inplace=True
    )

    after=len(combined)


    combined.to_csv(
        TRAIN_FILE,
        index=False
    )


    print()
    print("✅ TRAINING DATA UPDATED")
    print("-------------------------")
    print("Added rows:",len(new))
    print("Removed duplicates:",before-after)
    print("Total rows:",len(combined))
    print("Saved:",TRAIN_FILE)


if __name__ == "__main__":
    main()
