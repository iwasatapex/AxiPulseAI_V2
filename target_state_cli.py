#!/usr/bin/env python3

from core.target_state_engine.engine import TargetStateEngine


def banner(text):
    print("=" * 70)
    print(text.center(70))
    print("=" * 70)


def divider(char="─", width=80):
    print(char * width)


def main():

    banner("AxiPulseAI Target State Engine")

    targets = {}

    print("Enter targets (blank = ignore)\n")

    oh = input("Target OH: ").strip()
    nps = input("Target NPS: ").strip()
    release = input("Target Release: ").strip()
    transfer = input("Target Transfer: ").strip()
    quality = input("Target Quality: ").strip()
    competency = input("Target Competency: ").strip()
    attendance = input("Target Attendance: ").strip()


    if oh:
        targets["operational_health"] = float(oh)

    if nps:
        targets["nps"] = float(nps)

    if release:
        targets["release"] = float(release)

    if transfer:
        targets["transfer"] = float(transfer)

    if quality:
        targets["quality"] = float(quality)

    if competency:
        targets["competency"] = float(competency)

    if attendance:
        targets["attendance"] = float(attendance)


    print()
    print("=" * 70)
    print("SEARCHING REQUIRED OPERATIONAL STATE")
    print("=" * 70)


    engine = TargetStateEngine()

    result = engine.find_target_state(
        targets=targets
    )

    import numpy as np


    def divider():
        print("═" * 70)


    divider()
    print("🧠 AxiPulseAI TARGET STATE ENGINE")
    divider()


    print("\nTARGET REQUEST\n")

    for k,v in result.get("targets",{}).items():
        print(
            f"  {k.replace('_',' ').title():<25}: {v}"
        )


    print("\nRECOMMENDED OPERATIONAL STATE\n")

    for k,v in result.get(
        "recommended_state",
        {}
    ).items():

        print(
            f"  {k.title():<25}: {v}"
        )


    divider()


    oh_board = result.get(
        "leaderboards",
        {}
    ).get("OH", [])


    nps_board = result.get(
        "leaderboards",
        {}
    ).get("NPS", [])


    divider()

    print(
        "🧠 AXIPULSE AI MODEL COUNCIL"
    )

    divider()


    print(
        "MODEL PERFORMANCE BOARD"
    )

    print(
        "-" * 66
    )

    print(
        f"{'Model':<24}"
        f"{'OH':>10}"
        f"{'NPS':>10}"
        f"{'OH Confidence':>16}"
        f"{'NPS Confidence':>16}"
        f"{'State':>10}"
    )


    nps_map={
        x.get("model"):x
        for x in nps_board
    }


    for row in oh_board:

        nps=nps_map.get(
            row.get("model"),
            {}
        )

        status = (
            "OUTLIER"
            if row.get("model") in [
                x.get("model")
                for x in engine.council.analyze(
                    oh_board
                ).get("outliers", [])
            ]
            else "NORMAL"
        )

        print(
            f"{row.get('model','Unknown'):<24}"
            f"{row.get('prediction',0):>10.3f}"
            f"{nps.get('prediction',0):>10.3f}"
            f"{str(round(row.get('confidence',0),1))+'%':>16}"
            f"{str(round(nps.get('confidence',0),1))+'%':>16}"
            f"{status:>10}"
        )


    divider()

    print(
        "COUNCIL CONSENSUS"
    )

    print(
        "-" * 66
    )


    for metric,board in (
        ("OH",oh_board),
        ("NPS",nps_board)
    ):

        analysis=engine.council.analyze(
            board
        )

        print(
            f"{metric:<8}"
            f" Consensus {analysis['consensus']:<8.3f}"
            f" Spread ±{analysis['clean_spread']:<8.3f}"
            f" Agreement {analysis['agreement']:<6.2f}%"
            f" {analysis['health']}"
        )

    divider()


    print("NEURAL COUNCIL VERDICT")
    divider()

    total=sum(
        len(x)
        for x in result.get(
            "leaderboards",
            {}
        ).values()
    )

    outliers=[]

    for board in result.get(
        "leaderboards",
        {}
    ).values():

        for row in board:

            if row.get("status") == "OUTLIER":

                outliers.append(
                    row.get("model","Unknown")
                )


    valid_models = total - len(outliers)

    print(
        f"AI Minds Consulted : {total}"
    )

    print(
        f"Valid Models       : {valid_models}"
    )

    print(
        "Outliers           : "
        +
        (
            ", ".join(outliers)
            if outliers
            else "None"
        )
    )

    print(
        "Council Health     : GREEN"
    )
    divider()







if __name__ == "__main__":
    main()
