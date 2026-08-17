
"""
AxiPulseAI Target State Engine v2

Model Council Architecture:
- Uses all available trained models
- Multi-target reverse optimization
- Prediction leaderboard
- Confidence scoring
- Consensus scoring
"""

import json
import numpy as np
import joblib
from pathlib import Path

from core.forecast_ai.prediction.model_selector import MODELS_DIR, OH_LEGACY, NPS_LEGACY


class TargetStateEngine:

    def __init__(self, oh_bundle=None, nps_bundle=None):

        # Resolve the legacy model artifacts relative to the project root so
        # the engine is portable (independent of the current working directory).
        self.oh_file = MODELS_DIR / OH_LEGACY

        self.nps_file = MODELS_DIR / NPS_LEGACY

        # The caller may inject the exact model bundles to use (e.g. the GUI's
        # explicitly chosen model family). When no bundles are given the
        # legacy model files are loaded — unchanged backward-compatible
        # behavior. Injection only replaces WHERE models come from, never any
        # council/simulation mathematics.
        if oh_bundle is None or nps_bundle is None:
            oh_bundle = joblib.load(str(self.oh_file))
            nps_bundle = joblib.load(str(self.nps_file))

        self.oh_bundle = oh_bundle
        self.nps_bundle = nps_bundle

        from core.target_state_engine.council import ModelCouncil

        self.council = ModelCouncil(
            self.oh_bundle,
            self.nps_bundle
        )


    def _models(self, bundle):

        models = bundle.get(
            "all_models",
            {}
        )

        if not models:
            models={
                bundle["model_name"]:
                bundle["model"]
            }

        return models


    def _confidence(self, error):

        # lower error = higher confidence

        return round(
            max(
                0,
                min(
                    99.9,
                    100 - error*10
                )
            ),
            2
        )



    def _build_features(self, state, bundle):

        import pandas as pd

        features = bundle.get(
            "feature_names",
            []
        )

        row = {}

        # KPI mapping
        mapping = {
            "actual_quality": state.get("quality",90),
            "actual_competency": state.get("competency",94),
            "actual_attendance": state.get("attendance",90),
            "actual_release_rate": state.get("release",60),
            "actual_transfer_rate": state.get("transfer",9),
        }


        for k,v in mapping.items():
            row[k]=v


        # gaps
        row["quality_gap"] = row["actual_quality"] - 87
        row["competency_gap"] = row["actual_competency"] - 93
        row["attendance_gap"] = row["actual_attendance"] - 90
        row["release_gap"] = row["actual_release_rate"] - 60
        row["transfer_gap"] = row["actual_transfer_rate"] - 9


        # volume defaults
        row["total_calls_received"]=2000
        row["total_release_calls"]=1200


        # intelligence
        row["operational_intelligence_factor"]=-0.01
        row["business_intelligence_factor"]=0.0
        row["member_intelligence_factor"]=0.0


        # calendar defaults
        calendar={
            "day_of_week_sin":0,
            "day_of_week_cos":1,
            "month_sin":0,
            "month_cos":1,
            "quarter":1,
            "is_weekend":0
        }

        row.update(calendar)


        # NPS survey defaults
        row.update({
            "total_surveys":100,
            "survey_rate":5,
            "promoters":80,
            "passives":10,
            "detractors":10
        })


        df=pd.DataFrame([row])


        # exact model schema
        for f in features:
            if f not in df.columns:
                df[f]=0


        return df[features]




    def _build_feature_batch(
        self,
        states,
        bundle
    ):
        """
        Vectorized feature generation.
        Builds one dataframe for many states.
        """

        import pandas as pd


        rows=[]


        for state in states:

            row={

                "actual_quality":
                state.get(
                    "quality",
                    90
                ),

                "actual_competency":
                state.get(
                    "competency",
                    94
                ),

                "actual_attendance":
                state.get(
                    "attendance",
                    90
                ),

                "actual_release_rate":
                state.get(
                    "release",
                    60
                ),

                "actual_transfer_rate":
                state.get(
                    "transfer",
                    9
                )
            }


            row["quality_gap"] = (
                row["actual_quality"] - 87
            )

            row["competency_gap"] = (
                row["actual_competency"] - 93
            )

            row["attendance_gap"] = (
                row["actual_attendance"] - 90
            )

            row["release_gap"] = (
                row["actual_release_rate"] - 60
            )

            row["transfer_gap"] = (
                row["actual_transfer_rate"] - 9
            )


            row.update({

                "total_calls_received":2000,
                "total_release_calls":1200,

                "operational_intelligence_factor":-0.01,
                "business_intelligence_factor":0.0,
                "member_intelligence_factor":0.0,

                "day_of_week_sin":0,
                "day_of_week_cos":1,
                "month_sin":0,
                "month_cos":1,
                "quarter":1,
                "is_weekend":0,

                "total_surveys":100,
                "survey_rate":5,
                "promoters":80,
                "passives":10,
                "detractors":10
            })


            rows.append(row)


        df=pd.DataFrame(rows)


        features=bundle.get(
            "feature_names",
            []
        )


        for f in features:

            if f not in df.columns:
                df[f]=0


        return df[features]


    def _leaderboard(
        self,
        models,
        features,
        target
    ):

        results=[]

        for name,model in models.items():

            try:

                prediction=float(
                    np.asarray(
                        model.predict(features)
                    ).flatten()[0]
                )


                perf = (
                    self.oh_bundle
                    .get(
                        "algorithm_performance",
                        {}
                    )
                    .get(name,1)
                )


                if isinstance(perf,dict):

                    error=float(
                        list(
                            perf.values()
                        )[0]
                    )

                else:
                    error=float(perf)


                confidence=self._confidence(
                    error
                )


                results.append(
                    {
                    "model":name,
                    "prediction":round(
                        prediction,
                        3
                    ),
                    "confidence":confidence
                    }
                )

            except Exception as e:

                results.append(
                    {
                    "model":name,
                    "error":str(e)
                    }
                )


        return sorted(
            results,
            key=lambda x:
            x.get(
                "confidence",
                0
            ),
            reverse=True
        )





    def _generate_candidates(
        self,
        targets,
        n=100000
    ):
        """
        Generate candidate operational states.
        Vectorized Monte Carlo search.
        """

        rng = np.random.default_rng(
            42
        )

        states=[]

        for _ in range(n):

            states.append(
                {
                    "quality": rng.uniform(80,100),
                    "competency": rng.uniform(85,100),
                    "attendance": rng.uniform(80,100),
                    "release": rng.uniform(50,75),
                    "transfer": rng.uniform(0,20),
                }
            )

        return states



    def _score_state(
        self,
        predictions,
        targets
    ):

        errors=[]

        if targets.get(
            "operational_health"
        ) is not None:

            errors.append(
                (
                    predictions["oh"]
                    -
                    targets["operational_health"]
                )**2
            )


        if targets.get(
            "nps"
        ) is not None:

            errors.append(
                (
                    predictions["nps"]
                    -
                    targets["nps"]
                )**2
            )


        if targets.get(
            "release"
        ) is not None:

            errors.append(
                (
                    predictions["release"]
                    -
                    targets["release"]
                )**2
            )


        return np.sqrt(
            sum(errors)
        )





    def _batch_score(
        self,
        predictions,
        targets
    ):

        scores = np.zeros(
            len(predictions["oh"])
        )


        if targets.get(
            "operational_health"
        ) is not None:

            scores += (
                predictions["oh"]
                -
                targets["operational_health"]
            ) ** 2


        if targets.get(
            "nps"
        ) is not None:

            scores += (
                predictions["nps"]
                -
                targets["nps"]
            ) ** 2


        if targets.get(
            "release"
        ) is not None:

            scores += (
                predictions["release"]
                -
                targets["release"]
            ) ** 2


        return np.sqrt(scores)


    def _predict_state(
        self,
        state
    ):

        oh_features=self._build_features(
            state,
            self.oh_bundle
        )

        nps_features=self._build_features(
            state,
            self.nps_bundle
        )


        oh=[]

        for name,model in self._models(
            self.oh_bundle
        ).items():

            try:

                oh.append(
                    float(
                        np.asarray(
                            model.predict(
                                oh_features
                            )
                        ).flatten()[0]
                    )
                )

            except Exception:
                pass



        nps=[]

        for name,model in self._models(
            self.nps_bundle
        ).items():

            try:

                nps.append(
                    float(
                        np.asarray(
                            model.predict(
                                nps_features
                            )
                        ).flatten()[0]
                    )
                )

            except Exception:
                pass



        return {

            "oh":float(
                np.median(oh)
            ) if oh else 0,

            "nps":float(
                np.median(nps)
            ) if nps else 0

        }





    def _batch_predict_states(self, states):
        """
        Vectorized model council prediction.
        """

        oh_df = self._build_feature_batch(
            states,
            self.oh_bundle
        )

        nps_df = self._build_feature_batch(
            states,
            self.nps_bundle
        )


        oh_votes=[]

        for name, model in self._models(
            self.oh_bundle
        ).items():

            try:

                oh_votes.append(
                    np.asarray(
                        model.predict(
                            oh_df
                        )
                    )
                )

            except Exception as e:

                print(
                    "OH model skipped:",
                    name,
                    e
                )


        nps_votes=[]

        for name, model in self._models(
            self.nps_bundle
        ).items():

            try:

                raw=np.asarray(
                    model.predict(
                        nps_df
                    )
                )


                if (
                    raw.ndim == 2
                    and raw.shape[1] == 11
                ):

                    promoters=(
                        raw[:,9]
                        +
                        raw[:,10]
                    )

                    detractors=(
                        raw[:,:7]
                        .sum(axis=1)
                    )

                    total=raw.sum(
                        axis=1
                    )

                    nps_votes.append(
                        (
                            promoters
                            -
                            detractors
                        )
                        /
                        total
                        *
                        100
                    )

                else:

                    nps_votes.append(
                        raw.reshape(-1)
                    )


            except Exception as e:

                print(
                    "NPS model skipped:",
                    name,
                    e
                )


        return {

            "oh":
            np.median(
                np.vstack(
                    oh_votes
                ),
                axis=0
            ),


            "nps":
            np.median(
                np.vstack(
                    nps_votes
                ),
                axis=0
            )

        }


    def _build_final_leaderboard(
        self,
        state
    ):
        """
        Final model council vote
        for the winning operational state.
        """

        oh_features = self._build_features(
            state,
            self.oh_bundle
        )

        nps_features = self._build_features(
            state,
            self.nps_bundle
        )


        prediction = self.council.predict(
            oh_features,
            nps_features
        )


        return {
            "OH": prediction["OH"],
            "NPS": prediction["NPS"]
        }


    def find_target_state(
        self,
        targets,
        total_candidates: int = 100000,
        batch_size: int = 5000
    ):

        print(
            "Streaming candidate search..."
        )


        best = None
        best_score = float(
            "inf"
        )


        processed = 0


        while processed < total_candidates:


            current = min(
                batch_size,
                total_candidates - processed
            )


            print(
                f"Searching batch "
                f"{processed + current}/"
                f"{total_candidates}"
            )


            candidates = self._generate_candidates(
                targets,
                current
            )


            predictions = self._batch_predict_states(
                candidates
            )


            scores = self._batch_score(
                {
                    "oh":
                    predictions["oh"],

                    "nps":
                    predictions["nps"],

                    "release":
                    np.array(
                        [
                            s["release"]
                            for s in candidates
                        ]
                    )
                },
                targets
            )


            index = int(
                np.argmin(scores)
            )


            score = float(
                scores[index]
            )


            if score < best_score:

                best_score = score

                best = {

                    "state":
                    candidates[index],

                    "prediction":
                    {
                        "oh":
                        float(
                            predictions["oh"][index]
                        ),

                        "nps":
                        float(
                            predictions["nps"][index]
                        ),

                        "release":
                        float(
                            candidates[index]["release"]
                        )
                    }
                }


            processed += current



        return {

            "targets":
                targets,

            "recommended_state":
                best["state"],

            "distance":
                round(
                    best_score,
                    3
                ),

            "consensus":
                best["prediction"],

            "leaderboards":
                self._build_final_leaderboard(
                    best["state"]
                )

        }



# Module-level surface API (backward compatibility).  Delegates to a
# TargetStateEngine built from the legacy/production model files.
def find_target_state(targets, total_candidates=100000, batch_size=5000):
    """Compatibility alias for ``TargetStateEngine.find_target_state``."""
    return TargetStateEngine().find_target_state(
        targets,
        total_candidates=total_candidates,
        batch_size=batch_size,
    )
