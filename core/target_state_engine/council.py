"""
AxiPulseAI Model Council

Responsible for:
- Multi-model prediction
- NPS distribution conversion
- Consensus
- Outlier detection
- Confidence scoring
"""


import numpy as np

from core.target_state_engine.nps_adapter import convert_output


class ModelCouncil:


    def __init__(
        self,
        oh_bundle,
        nps_bundle
    ):

        self.oh_bundle = oh_bundle
        self.nps_bundle = nps_bundle



    def _models(self, bundle):

        models = bundle.get(
            "all_models",
            {}
        )

        if not models:

            models = {
                bundle["model_name"]:
                bundle["model"]
            }

        return models



    def _confidence(
        self,
        name,
        bundle
    ):

        perf = (
            bundle
            .get(
                "algorithm_performance",
                {}
            )
            .get(
                name,
                1
            )
        )

        try:

            if isinstance(
                perf,
                dict
            ):

                error=float(
                    list(
                        perf.values()
                    )[0]
                )

            else:

                error=float(perf)


            return round(
                max(
                    0,
                    min(
                        99.9,
                        100-error*10
                    )
                ),
                2
            )

        except Exception:
            return 0



    def predict(
        self,
        oh_features,
        nps_features
    ):


        result={

            "OH":[],
            "NPS":[]

        }


        for name,model in self._models(
            self.oh_bundle
        ).items():

            try:

                pred=float(
                    np.asarray(
                        model.predict(
                            oh_features
                        )
                    )
                    .flatten()[0]
                )


                result["OH"].append({

                    "model":name,

                    "prediction":pred,

                    "confidence":
                    self._confidence(
                        name,
                        self.oh_bundle
                    )

                })


            except Exception as e:

                result["OH"].append({

                    "model":name,

                    "error":str(e)

                })



        for name,model in self._models(
            self.nps_bundle
        ).items():

            try:

                raw=model.predict(
                    nps_features
                )

                pred=float(
                    convert_output(
                        raw
                    )[0]
                )


                result["NPS"].append({

                    "model":name,

                    "prediction":pred,

                    "confidence":
                    self._confidence(
                        name,
                        self.nps_bundle
                    )

                })


            except Exception as e:

                result["NPS"].append({

                    "model":name,

                    "error":str(e)

                })


        return result





    def analyze(
        self,
        board,
        lower=0,
        upper=100
    ):
        """
        ANIC consensus intelligence.
        """

        values=[]

        for row in board:

            if "prediction" in row:

                values.append(
                    float(
                        row["prediction"]
                    )
                )


        if not values:

            return {
                "consensus": None,
                "agreement": 0,
                "outliers": []
            }


        median=float(
            np.median(values)
        )

        spread=float(
            np.std(values)
        )


        outliers=[]
        clean_values=[]


        threshold=max(
            10,
            spread * 3
        )


        for row in board:

            if "prediction" not in row:
                continue


            value=float(
                row["prediction"]
            )


            if (
                value < lower
                or value > upper
                or abs(value-median) > threshold
            ):

                row["status"]="OUTLIER"

                outliers.append(
                    {
                        "model":row["model"],
                        "prediction":value
                    }
                )

            else:

                row["status"]="NORMAL"

                clean_values.append(
                    value
                )


        if clean_values:

            consensus=float(
                np.mean(clean_values)
            )

            clean_spread=float(
                np.std(clean_values)
            )

            clean_median=float(
                np.median(clean_values)
            )

            agreement=max(
                0,
                min(
                    100,
                    100 -
                    (
                        clean_spread /
                        max(1, clean_median)
                        *
                        100
                    )
                )
            )

        else:

            consensus=median
            clean_spread=spread
            agreement=0


        return {

            "consensus":
            round(
                consensus,
                3
            ),

            "median":
            round(
                median,
                3
            ),

            "spread":
            round(
                spread,
                3
            ),

            "clean_spread":
            round(
                clean_spread,
                3
            ),

            "agreement":
            round(
                agreement,
                2
            ),

            "outliers":
            outliers,

            "health":
            (
                "GREEN"
                if agreement > 95
                else
                "YELLOW"
            )

        }


    def weighted_consensus(
        self,
        board,
        bundle
    ):
        """
        Weighted AI consensus.

        Weight comes from model
        historical performance.
        """

        values=[]
        weights=[]


        performance=bundle.get(
            "algorithm_performance",
            {}
        )


        for row in board:

            if (
                "prediction" not in row
                or
                row.get("status")
                ==
                "OUTLIER"
            ):
                continue


            name=row["model"]


            perf=performance.get(
                name,
                1
            )


            try:

                if isinstance(
                    perf,
                    dict
                ):

                    error=float(
                        list(
                            perf.values()
                        )[0]
                    )

                else:

                    error=float(perf)


                # lower error = higher weight
                weight=1/(error+0.001)


            except Exception:
                weight = 1


            values.append(
                float(
                    row["prediction"]
                )
            )

            weights.append(
                weight
            )


            row["weight"]=round(
                weight,
                4
            )


        if not values:

            return None


        return round(
            np.average(
                values,
                weights=weights
            ),
            3
        )


    def summarize(
        self,
        board
    ):

        values=[

            x["prediction"]

            for x in board

            if "prediction" in x

        ]


        if not values:

            return {}


        mean=float(
            np.mean(values)
        )


        spread=float(
            np.std(values)
        )


        return {

            "average":
            round(mean,3),

            "spread":
            round(spread,3),

            "models":
            len(values)

        }


# ---------------------------------------------------------------------------
# Module-level surface API (backward compatibility)
# ---------------------------------------------------------------------------
# Each helper builds a ModelCouncil from the production OH/NPS bundles and
# delegates to the corresponding method, so the module-level surface mirrors
# the class without duplicating any council mathematics.
def _council_from_bundles(oh_bundle, nps_bundle):
    return ModelCouncil(oh_bundle, nps_bundle)


def predict(oh_bundle, nps_bundle, oh_features, nps_features):
    """Compatibility alias for ``ModelCouncil.predict``."""
    return _council_from_bundles(oh_bundle, nps_bundle).predict(
        oh_features,
        nps_features,
    )


def analyze(oh_bundle, nps_bundle, board, lower=0, upper=100):
    """Compatibility alias for ``ModelCouncil.analyze``."""
    return _council_from_bundles(oh_bundle, nps_bundle).analyze(
        board,
        lower=lower,
        upper=upper,
    )


def weighted_consensus(oh_bundle, nps_bundle, board):
    """Compatibility alias for ``ModelCouncil.weighted_consensus``."""
    return _council_from_bundles(oh_bundle, nps_bundle).weighted_consensus(
        board,
        nps_bundle,
    )


def summarize(oh_bundle, nps_bundle, board):
    """Compatibility alias for ``ModelCouncil.summarize``."""
    return _council_from_bundles(oh_bundle, nps_bundle).summarize(board)

