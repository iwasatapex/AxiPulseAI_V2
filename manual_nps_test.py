from api.services.nps_service import NPSService


def ask(label, default, minimum=None, maximum=None):
    while True:
        value=input(f"  {label} [{default}]: ").strip()

        if value.lower()=="q":
            exit()

        if value=="":
            return default

        try:
            value=type(default)(value)

            if minimum is not None and value < minimum:
                print(f"  ❌ Minimum {minimum}")
                continue

            if maximum is not None and value > maximum:
                print(f"  ❌ Maximum {maximum}")
                continue

            return value

        except Exception: print("  ❌ Invalid input")


def section(title):
    print("\n"+"="*45)
    print(title)
    print("="*45)


def main():
    print("""
    ╔══════════════════════════════════════╗
    ║        AxiPulseAI NPS TEST           ║
    ╚══════════════════════════════════════╝

    Enter values or press ENTER for defaults.
    Type q anytime to exit.
    """)


    data={
    "date":"2026-08-02"
    }


    section("TARGET KPI INPUTS")

    data.update({
    "target_quality":ask("Quality Target %",87,0,100),
    "target_competency":ask("Competency Target %",93,0,100),
    "target_attendance":ask("Attendance Target %",90,0,100),
    "target_release_rate":ask("Release Target %",60,0,100),
    "target_transfer_rate":ask("Transfer Target %",9,0,20),
    })


    section("ACTUAL KPI INPUTS")

    data.update({
    "actual_quality":ask("Quality Actual %",85,0,100),
    "actual_competency":ask("Competency Actual %",90,0,100),
    "actual_attendance":ask("Attendance Actual %",92,0,100),
    "actual_release_rate":ask("Release Actual %",58,0,100),
    "actual_transfer_rate":ask("Transfer Actual %",11,0,20),
    })


    section("OPERATIONAL INPUTS")

    data.update({
    "operational_health":ask("Operational Health",88,0,120),
    "business_intelligence_factor":ask("Business Intelligence",0.7,0,1),
    "member_intelligence_factor":ask("Member Intelligence",0.7,0,1),
    "total_calls_received":ask("Total Calls",2400,1,20000),
    })


    section("KPI GAP ANALYSIS")

    kpis=[
        ("Quality","target_quality","actual_quality"),
        ("Competency","target_competency","actual_competency"),
        ("Attendance","target_attendance","actual_attendance"),
        ("Release","target_release_rate","actual_release_rate"),
        ("Transfer","target_transfer_rate","actual_transfer_rate"),
    ]

    for name,target,actual in kpis:
        gap=data[target]-data[actual]

        # Transfer is better when lower
        if name=="Transfer":
            status="✅ Better" if gap > 0 else ("⚠️ Worse" if gap < 0 else "➖ Met")
        else:
            status="✅ Better" if gap < 0 else ("⚠️ Worse" if gap > 0 else "➖ Met")

        print(f"{name:12} Target:{data[target]:6} Actual:{data[actual]:6} Gap:{gap:7.2f} {status}")


    confirm=input("\nRun prediction? (Y/n): ").strip().lower()

    if confirm=="n":
        print("Cancelled")
        exit()


    print("\nPredicting...\n")

    service=NPSService()
    result=service.predict(data)


    section("NPS RESULT")

    print(f"NPS              : {result['nps']}")
    print(f"Confidence       : {result['confidence']}")
    print(f"Promoters        : {result['promoters']}")
    print(f"Passives         : {result['passives']}")
    print(f"Detractors       : {result['detractors']}")
    print(f"Prediction Range : {result['prediction_interval']}")


if __name__ == "__main__":
    main()
