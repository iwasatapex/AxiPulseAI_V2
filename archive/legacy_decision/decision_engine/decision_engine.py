"""
AxiPulseAI Decision Intelligence Layer v1

Converts:
Forecast + Target State
into:
Business actions and expected impact
"""


class DecisionEngine:


    def __init__(self):

        self.rules = [

            {
                "name":
                "Competency Improvement",

                "condition":
                lambda c:
                c["competency_gap"] < 0,

                "action":
                "Increase coaching and knowledge reinforcement",

                "impact":
                "Improves quality and reduces transfers"
            },


            {
                "name":
                "Transfer Reduction",

                "condition":
                lambda c:
                c["transfer"] > 12,

                "action":
                "Review escalation paths and agent guidance",

                "impact":
                "Improves resolution efficiency"
            },


            {
                "name":
                "Quality Recovery",

                "condition":
                lambda c:
                c["quality"] < 85,

                "action":
                "Deploy targeted quality calibration",

                "impact":
                "Improves customer outcomes"
            },


            {
                "name":
                "Attendance Risk",

                "condition":
                lambda c:
                c["attendance"] < 85,

                "action":
                "Review staffing availability",

                "impact":
                "Protects operational health"
            }

        ]


    def analyze(
        self,
        current,
        target
    ):


        gaps={

            "quality_gap":
            target.get("quality",0)
            -
            current.get("quality",0),


            "competency_gap":
            target.get("competency",0)
            -
            current.get("competency",0),


            "transfer":
            current.get("transfer",0),


            "quality":
            current.get("quality",0),


            "attendance":
            current.get("attendance",0)

        }


        actions=[]


        for rule in self.rules:

            if rule["condition"](gaps):

                actions.append({

                    "decision":
                    rule["name"],

                    "action":
                    rule["action"],

                    "expected_effect":
                    rule["impact"]

                })


        return {

            "current_state":
            current,


            "target_state":
            target,


            "recommended_actions":
            actions,


            "priority":
            len(actions)

        }


def analyze(current, target):
    return DecisionEngine().analyze(current, target)
