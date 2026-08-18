"""
Engine 2 Service

Serves REAL NPS predictions from the canonical ``NPSPredictor`` engine - the
same predictor/loader used by the working CLI and ForecastAI paths. The
prediction path is intentionally decoupled from persistence (no SQLAlchemy
needed for a prediction request).

DEFAULT MODEL: the API resolves ``NPSService()`` to the CANONICAL PRODUCTION
artifact ``models/production_NPS.pkl`` (the same artifact the ForecastAI path
loads). It does NOT default to the legacy ``nps_predictor_model.pkl``.

Legacy / stress / test artifacts are compatibility-only and are loaded only
when a caller explicitly passes ``model_path=`` to ``NPSService``.

The mock path exists ONLY as an explicit escape hatch (``mock=True``). It is
never the default, so the API can never silently fabricate NPS.
"""

import hashlib
import json
import logging
import os
from datetime import date

from core.nps_predictor import NPSPredictor
from core.forecast_ai.prediction.model_selector import (
    MODELS_DIR,
    NPS_LEGACY,
    NPS_SUFFIX,
    PRODUCTION_FAMILY,
)

logger = logging.getLogger(__name__)

# Canonical production NPS artifact - the single production source.
PRODUCTION_NPS_FILE = f"{PRODUCTION_FAMILY}{NPS_SUFFIX}"  # production_NPS.pkl
DEFAULT_MODEL_PATH = os.path.join(str(MODELS_DIR), PRODUCTION_NPS_FILE)
_MANIFEST_PATH = os.path.join(str(MODELS_DIR), "manifest.json")

# Explicit legacy/compatibility artifact (opt-in only via model_path=).
LEGACY_MODEL_PATH = os.path.join(str(MODELS_DIR), NPS_LEGACY)

# Mirrors ``PredictionService._build_nps_row`` survey estimation.
_DEFAULT_SURVEY_RATE = 0.10


class NPSServiceUnavailableError(RuntimeError):
    """Raised when the real NPS model is unavailable and no mock was requested."""


class NPSService:
    def __init__(self, model_path=None):
        # Default = canonical production artifact. A caller may explicitly
        # opt into a legacy/test/stress artifact by passing ``model_path=``.
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.predictor = None
        self.load_model()

    def load_model(self):
        """Load the trained model through the canonical NPSPredictor loader."""
        if not os.path.exists(self.model_path):
            logger.warning("NPS model not found: %s", self.model_path)
            self.predictor = None
            return None
        try:
            is_default = os.path.abspath(self.model_path) == os.path.abspath(DEFAULT_MODEL_PATH)
            if is_default:
                # Default production path must pass integrity verification AND
                # resolve to a canonical production role (never legacy/stress).
                self._verify_baseline_artifact(self.model_path, require_production=True)
            elif os.path.abspath(self.model_path) == os.path.abspath(LEGACY_MODEL_PATH):
                # Explicit legacy opt-in is permitted but never canonical.
                self._verify_baseline_artifact(self.model_path, require_production=False)
            predictor = NPSPredictor()
            predictor.load_model(self.model_path)
            self.predictor = predictor
            logger.info("NPS predictor loaded: %s", self.model_path)
        except Exception as e:
            logger.error("Failed to load NPS model: %s", e)
            self.predictor = None
        return self.predictor

    @staticmethod
    def _verify_baseline_artifact(model_path: str, require_production: bool = False) -> None:
        if not os.path.exists(_MANIFEST_PATH):
            raise RuntimeError("Model integrity manifest is missing")
        with open(_MANIFEST_PATH, encoding="utf-8") as handle:
            manifest = json.load(handle)
        entry = manifest.get(os.path.basename(model_path), {})
        expected = entry.get("sha256")
        if not expected:
            raise RuntimeError(f"No integrity hash registered for {os.path.basename(model_path)}")
        digest = hashlib.sha256()
        with open(model_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if not digest.hexdigest() == expected:
            raise RuntimeError(f"Model integrity check failed for {os.path.basename(model_path)}")
        if require_production:
            role = entry.get("role")
            if role != "production":
                raise RuntimeError(
                    f"Default model {os.path.basename(model_path)} is not a "
                    f"canonical production artifact (role={role!r}); refusing."
                )
            if entry.get("legacy"):
                raise RuntimeError(
                    f"Default model {os.path.basename(model_path)} is a legacy "
                    f"artifact; refusing to use it as production."
                )

    def is_loaded(self):
        """Check whether the real model is loaded."""
        return self.predictor is not None

    @staticmethod
    def _build_prediction_row(input_data: dict) -> dict:
        """Normalise API input into the canonical NPS feature row.

        None-valued optional fields are dropped so the engine's documented
        defaults apply (missing features are aligned to 0 by the engine).
        """
        row = {k: v for k, v in dict(input_data).items() if v is not None}
        if "total_surveys" not in row:
            total_calls = int(row.get("total_calls_received", 0) or 0)
            release_rate = float(row.get("actual_release_rate", 0.0) or 0.0)
            row["total_surveys"] = max(
                1,
                int(total_calls * release_rate / 100.0 * _DEFAULT_SURVEY_RATE),
            )
        if not row.get("date"):
            row["date"] = date.today().isoformat()
        return row

    def predict(self, input_data: dict, mock: bool = False):
        """Run the real NPS engine.

        Parameters
        ----------
        input_data:
            Raw NPS feature dictionary (canonical KPI field names).
        mock:
            Explicit escape hatch only - never the default. When ``True`` and
            the real model is unavailable or fails, a fabricated mock result is
            returned. When ``False`` (default), errors propagate so callers can
            never receive fabricated NPS silently.
        """
        if self.predictor is None:
            if mock:
                return self._mock_predict(input_data)
            raise NPSServiceUnavailableError(
                f"NPS model unavailable at {self.model_path}"
            )
        try:
            row = self._build_prediction_row(input_data)
            return self.predictor.predict(row)
        except Exception as e:
            if mock:
                logger.error("NPS prediction failed (explicit mock fallback): %s", e)
                return self._mock_predict(input_data)
            logger.error("NPS prediction failed: %s", e)
            raise
    
    def _mock_predict(self, input_data: dict):
        """Fabricated mock prediction - ONLY for explicit ``mock=True`` callers.

        This is retained purely as a development escape hatch and is never used
        by the API (the API never passes ``mock=True``).
        """
        health = input_data.get('operational_health', 75)

        # Base NPS from health
        base_nps = 60 + (health / 100) * 25

        # Add release rate influence
        release = input_data.get('actual_release_rate', 70)
        base_nps += (release - 70) * 0.2

        # Add intelligence factors
        base_nps += input_data.get('business_intelligence_factor', 0.5) * 5
        base_nps += input_data.get('member_intelligence_factor', 0.5) * 5

        # Add noise
        import numpy as np
        base_nps += np.random.normal(0, 2)

        # Clip to [75, 100]
        nps = float(np.clip(base_nps, 75, 100))

        # Generate distribution
        total_surveys = int(input_data.get('total_calls_received', 2000) * 0.10)
        promoters = int(total_surveys * (nps / 100))
        passives = int(total_surveys * 0.08)
        detractors = total_surveys - promoters - passives

        # Create 11-score distribution
        distribution = [0] * 11
        # Distribute scores with most in 9-10
        for i in range(11):
            if i <= 6:
                distribution[i] = max(0, int(detractors * (1 - (i / 7))))
            elif i <= 8:
                distribution[i] = int(passives * (1 - ((i - 7) / 2)))
            else:
                distribution[i] = int(promoters * ((i - 8) / 2))

        return {
            "nps": nps,
            "promoters": float(promoters),
            "passives": float(passives),
            "detractors": float(detractors),
            "distribution": {f"score_{i}": float(distribution[i]) for i in range(11)},
            "ensemble_details": {
                "catboost": float(nps + 1.2),
                "mlp": float(nps - 0.8),
                "ensemble_weighted": float(nps)
            }
        }


# Module-level compatibility surface
def load_model():
    return NPSService().load_model()


def is_loaded():
    return NPSService().is_loaded()


def predict(input_data: dict, mock: bool = False):
    return NPSService().predict(input_data, mock=mock)

