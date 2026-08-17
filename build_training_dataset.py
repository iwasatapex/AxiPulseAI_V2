import pandas as pd
from pathlib import Path

SOURCE = Path("training/healthcare_cx_data.csv")
TARGET = Path("training/training.csv")

if __name__ == "__main__":
    df = pd.read_csv(SOURCE)

    # Create ML targets
    if "promoters" in df and "total_surveys" in df:
        scores = []
        for _, r in df.iterrows():
            total = int(r["total_surveys"])
            p = int(r["promoters"])
            pa = int(r["passives"])
            d = int(r["detractors"])

            row = [0]*11

            # Existing simulator score buckets if available
            if all(f"score_{i}" in df.columns for i in range(11)):
                scores.append([r[f"score_{i}"] for i in range(11)])
            else:
                # fallback distribution reconstruction
                row[10] = p
                row[8] = pa
                row[5] = d
                scores.append(row)

        for i in range(11):
            df[f"score_{i}"] = [x[i] for x in scores]


    # Remove analysis-only columns
    remove = [
        "promoter_pct",
        "passive_pct",
        "detractor_pct",
        "nps_today",
        "avg_nps",
    ]

    df = df.drop(columns=remove, errors="ignore")


    # append
    if TARGET.exists():
        old = pd.read_csv(TARGET)
        df = pd.concat([old, df], ignore_index=True)

    df.to_csv(TARGET,index=False)

    print("TRAINING DATA READY")
    print("Rows:",len(df))
    print("Columns:",len(df.columns))
    print("Saved:",TARGET)
