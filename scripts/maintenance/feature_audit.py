import pandas as pd
import os

FILE="training/healthcare_cx_data.csv"

if __name__ == "__main__":
    df=pd.read_csv(FILE)

    print("="*80)
    print("AXIPULSEAI FEATURE COMPLETENESS AUDIT")
    print("="*80)

    print("\nDATASET")
    print("-"*80)
    print("Rows:",len(df))
    print("Columns:",len(df.columns))


    # ----------------------------
    # Required feature groups
    # ----------------------------

    groups = {

    "Operational Core":[
    "calls",
    "complexity",
    "transfer_rate",
    "release_rate",
    "quality_kpi",
    "competency",
    "attendance",
    "operational_health"
    ],

    "Outcome / CX":[
    "nps_value",
    "promoters",
    "passives",
    "detractors",
    "total_surveys",
    "survey_rate"
    ],

    "Events":[
    "event",
    "last_week"
    ],

    "Temporal":[
    "day",
    "date",
    "weekday"
    ],

    "Agent":[
    "agent_id",
    "experience_weeks",
    "training_completed",
    "operational_intelligence",
    "business_intelligence",
    "member_intelligence",
    "fatigue",
    "availability"
    ],

    "Capacity / Queue":[
    "available_agents",
    "available_capacity",
    "workload_capacity_ratio",
    "queue_length",
    "wait_time",
    "sla",
    "occupancy",
    "abandon_rate"
    ],

    "ML Time-Series":[
    "previous_day_OH",
    "rolling_7_day_OH",
    "rolling_14_day_release",
    "rolling_30_day_quality"
    ]

    }


    for name,cols in groups.items():

        print("\n"+name)
        print("-"*40)

        for c in cols:
            if c in df.columns:
                print("✅",c)
            else:
                print("❌",c)


    # Missing values

    print("\nMISSING VALUE SUMMARY")
    print("-"*80)

    missing=df.isna().mean()*100

    for c,v in missing.sort_values(ascending=False).head(20).items():
        print(f"{c:35} {v:.2f}%")


    # Duplicate

    print("\nQUALITY")
    print("-"*80)

    print("Duplicate rows:",df.duplicated().sum())


    # Numeric feature count

    print("\nFEATURE TYPES")
    print("-"*80)

    print(df.dtypes.value_counts())


    print("\nAUDIT COMPLETE")
