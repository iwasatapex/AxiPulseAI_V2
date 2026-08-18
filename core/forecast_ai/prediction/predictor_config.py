"""
ForecastAI predictor configuration.

Connects ForecastAI to the trained production predictors.

V2 changes (model-selection workflow):
  * ``create_oh_predictor`` / ``create_nps_predictor`` accept an optional
    ``model_family`` parameter.  When provided the V2 naming convention
    ``{family}_OH.pkl`` / ``{family}_NPS.pkl`` is used.
  * When *model_family* is ``None`` the legacy filenames
    ``operation_health_predictor.joblib`` / ``nps_predictor_model.pkl``
    are loaded (backward compatibility for pre-V2 callers).
  * ``load_model_pair`` loads and validates a complete OH+NPS pair by
    family name, raising ``ModelPairError`` on any problem.
"""
import hashlib
import json
import os
from pathlib import Path

from core.operation_health_predictor.predictor import OperationalHealthPredictor
from core.nps_predictor.predictor import NPSPredictor
from .model_selector import (
    MODELS_DIR,
    OH_SUFFIX,
    NPS_SUFFIX,
    OH_LEGACY,
    NPS_LEGACY,
    PRODUCTION_FAMILY,
    ModelPairError,
    validate_model_pair,
)

MANIFEST_NAME = "manifest.json"


class ProductionIntegrityError(RuntimeError):
    """Raised when a canonical production artifact fails integrity verification."""


def _manifest_path():
    return MODELS / MANIFEST_NAME


def _load_manifest() -> dict:
    p = _manifest_path()
    if not p.exists():
        raise ProductionIntegrityError(
            f"Production integrity manifest missing: {p}"
        )
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProductionIntegrityError(
            f"Production integrity manifest unreadable: {p}"
        ) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_production_artifact(
    path: Path,
    *,
    is_oh: bool,
    require_role: bool = True,
) -> None:
    """Verify a canonical production artifact against the integrity manifest.

    Fail-closed checks (all required for the canonical production family):
      1. manifest present and resolvable
      2. artifact entry present in manifest
      3. file SHA-256 matches the manifest hash
      4. role is ``production`` (never legacy/test/stress)
      5. not marked legacy
      6. non-empty source provenance
      7. artifact loads and is trained
      8. feature count matches the manifest (when recorded)
      9. output dimensionality matches the manifest (when recorded)

    Raises :class:`ProductionIntegrityError` on any failure.  Never silently
    substitutes another artifact.
    """
    artifact_name = path.name
    manifest = _load_manifest()
    entry = manifest.get(artifact_name)
    if entry is None:
        raise ProductionIntegrityError(
            f"No manifest entry for production artifact {artifact_name}; refusing."
        )
    expected = entry.get("sha256")
    if not expected:
        raise ProductionIntegrityError(
            f"No integrity hash registered for {artifact_name}; refusing."
        )
    actual = _sha256(path)
    if not _const_eq(actual, expected):
        raise ProductionIntegrityError(
            f"SHA-256 mismatch for production artifact {artifact_name}; refusing."
        )
    if require_role:
        role = entry.get("role")
        if role != "production":
            raise ProductionIntegrityError(
                f"Production artifact {artifact_name} has role={role!r} "
                f"(expected 'production'); refusing."
            )
        if entry.get("legacy"):
            raise ProductionIntegrityError(
                f"Production artifact {artifact_name} is a legacy artifact; refusing."
            )
    # Source provenance must be non-empty for canonical production.
    if not entry.get("source"):
        raise ProductionIntegrityError(
            f"Production artifact {artifact_name} has empty provenance; refusing."
        )

    # Verify the artifact actually loads and matches the manifest's structural
    # contract (feature count, output dimensionality, trained state).
    try:
        import joblib
        obj = joblib.load(str(path))
    except Exception as exc:
        raise ProductionIntegrityError(
            f"Production artifact {artifact_name} failed to load: {exc}"
        ) from exc

    if not obj.get("trained"):
        raise ProductionIntegrityError(
            f"Production artifact {artifact_name} is not trained; refusing."
        )

    model = obj.get("model")
    feature_names = obj.get("feature_names") or []
    n_features = getattr(model, "n_features_in_", None)

    manifest_feats = entry.get("n_features_in") or entry.get("feature_count")
    if manifest_feats is not None and n_features is not None:
        if int(manifest_feats) != int(n_features):
            raise ProductionIntegrityError(
                f"Production artifact {artifact_name} feature count "
                f"({n_features}) does not match manifest ({manifest_feats}); refusing."
            )

    manifest_outputs = entry.get("n_outputs")
    n_outputs = getattr(model, "n_outputs_", None)
    if n_outputs is None and (not is_oh):
        # NPS output count is recorded in the artifact metadata (num_scores),
        # since MultiOutputRegressor does not expose n_outputs_.
        meta = obj.get("metadata") or {}
        num_scores = meta.get("num_scores") or meta.get("num_score_buckets")
        if num_scores is not None:
            try:
                n_outputs = int(num_scores)
            except (TypeError, ValueError):
                n_outputs = None
    if manifest_outputs is not None and n_outputs is not None:
        if int(manifest_outputs) != int(n_outputs):
            raise ProductionIntegrityError(
                f"Production artifact {artifact_name} output count "
                f"({n_outputs}) does not match manifest ({manifest_outputs}); refusing."
            )

    # NPS must remain 11-output — enforced directly from the loaded artifact,
    # not only when the manifest records the field.
    if (not is_oh) and n_outputs is not None and int(n_outputs) != 11:
        raise ProductionIntegrityError(
            f"Production NPS artifact {artifact_name} is not 11-output "
            f"(got {n_outputs}); refusing."
        )

    # Feature schema / order: if the manifest records the exact names, the
    # artifact must match. This is a hard fail-closed check: an exact
    # feature-name/order mismatch or malformed schema metadata must never be
    # silently accepted as a valid production artifact.
    manifest_feature_meta = entry.get("feature_metadata")
    if manifest_feature_meta:
        try:
            recorded = json.loads(manifest_feature_meta)
        except Exception as exc:
            raise ProductionIntegrityError(
                f"Production artifact {artifact_name} has malformed feature "
                f"schema metadata in manifest; refusing."
            ) from exc
        if not isinstance(recorded, list):
            raise ProductionIntegrityError(
                f"Production artifact {artifact_name} feature schema metadata "
                f"is not a list in manifest; refusing."
            )
        if feature_names and list(recorded) != list(feature_names):
            raise ProductionIntegrityError(
                f"Production artifact {artifact_name} feature schema/order "
                f"does not match manifest; refusing."
            )


def _const_eq(a: str, b: str) -> bool:
    """Constant-time string comparison."""
    return hashlib.sha256(a.encode()).digest() == hashlib.sha256(b.encode()).digest()


def _is_production_family(model_family) -> bool:
    return not model_family or model_family == PRODUCTION_FAMILY




class _FallbackOHPredictor:
    """Explicit degraded-mode predictor used when no model artifact is packaged."""
    trained = True
    model = object()
    feature_names = []
    _all_models = {}
    def predict(self, row):
        quality = float(row.get("actual_quality", row.get("quality", 87)))
        competency = float(row.get("actual_competency", row.get("competency", 93)))
        attendance = float(row.get("actual_attendance", row.get("attendance", 90)))
        release = float(row.get("actual_release_rate", row.get("release", 60)))
        transfer = float(row.get("actual_transfer_rate", row.get("transfer", 9)))
        score = (quality + competency + attendance + release + (100.0 - transfer)) / 5.0
        return float(max(0.0, min(120.0, score)))


class _FallbackNPSPredictor:
    """Explicit degraded-mode NPS predictor used only without an artifact."""
    trained = True
    model = object()
    feature_names = []
    _all_models = {}
    def predict(self, row):
        from core.nps_predictor.inference import fallback_predict
        return fallback_predict(self, row)


ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "models"


def _allow_fallback() -> bool:
    """Allow degraded predictors only when explicitly enabled outside production."""
    return os.getenv("AXIPULSE_ALLOW_FALLBACK_MODE", "").lower() in {"1", "true", "yes"}


def model_pair_paths(model_family):
    """Return ``(oh_path, nps_path)`` for a model family name."""
    oh_path = MODELS / f"{model_family}{OH_SUFFIX}"
    nps_path = MODELS / f"{model_family}{NPS_SUFFIX}"
    return oh_path, nps_path


def create_oh_predictor(model_family=None):
    """Create and load an OH predictor.

    Parameters
    ----------
    model_family:
        When provided, loads ``models/{family}_OH.pkl``.
        When ``None``, resolves to the canonical **production** artifact
        ``models/production_OH.pkl``.  It never silently falls back to the
        legacy ``operation_health_predictor.joblib`` — that file is only used
        if it is the *explicit* ``model_family`` (i.e. ``OH_LEGACY``) and the
        canonical production artifact is absent.
    """
    predictor = OperationalHealthPredictor()

    if model_family:
        path = MODELS / f"{model_family}{OH_SUFFIX}"
    else:
        # Canonical production artifact, not the legacy filename.
        path = MODELS / f"{PRODUCTION_FAMILY}{OH_SUFFIX}"

    if not Path(path).exists():
        # Protect canonical production: never silently degrade to legacy.
        # Fallback only occurs when the caller explicitly opts in AND the
        # canonical production artifact is genuinely absent.
        if model_family is None and not path.exists() and _allow_fallback():
            legacy = MODELS / OH_LEGACY
            if legacy.exists():
                path = legacy
            else:
                return _FallbackOHPredictor()
        raise FileNotFoundError(f"OH model not found: {path}")

    # Canonical production must pass fail-closed manifest integrity checks
    # (hash, role, provenance) before it is loaded.
    if _is_production_family(model_family):
        verify_production_artifact(path, is_oh=True)

    predictor.load_model(str(path))
    return predictor


def create_nps_predictor(model_family=None):
    """Create and load an NPS predictor.

    Parameters
    ----------
    model_family:
        When provided, loads ``models/{family}_NPS.pkl``.
        When ``None``, resolves to the canonical **production** artifact
        ``models/production_NPS.pkl``.  It never silently falls back to the
        legacy ``nps_predictor_model.pkl`` — that file is only used if it is
        the *explicit* ``model_family`` (i.e. ``NPS_LEGACY``) and the canonical
        production artifact is absent.
    """
    predictor = NPSPredictor()

    if model_family:
        path = MODELS / f"{model_family}{NPS_SUFFIX}"
    else:
        # Canonical production artifact, not the legacy filename.
        path = MODELS / f"{PRODUCTION_FAMILY}{NPS_SUFFIX}"

    if not Path(path).exists():
        # Protect canonical production: never silently degrade to legacy.
        # Fallback only occurs when the caller explicitly opts in AND the
        # canonical production artifact is genuinely absent.
        if model_family is None and not path.exists() and _allow_fallback():
            legacy = MODELS / NPS_LEGACY
            if legacy.exists():
                path = legacy
            else:
                return _FallbackNPSPredictor()
        raise FileNotFoundError(f"NPS model not found: {path}")

    # Canonical production must pass fail-closed manifest integrity checks
    # (hash, role, provenance) before it is loaded.
    if _is_production_family(model_family):
        verify_production_artifact(path, is_oh=False)

    predictor.load_model(str(path))
    return predictor


def load_model_pair(model_family):
    """Load and validate a complete OH+NPS model pair by family name.

    Validation performed:
      1. Both ``{family}_OH.pkl`` and ``{family}_NPS.pkl`` exist.
      2. Each model bundle has a non-None ``model`` attribute.
      3. Each predictor is marked ``trained``.
      4. Each predictor has non-empty ``feature_names``.

    Returns ``(oh_predictor, nps_predictor)``.
    Raises :class:`ModelPairError` on any validation failure.
    """
    # 1. Path / existence validation
    oh_path, nps_path = validate_model_pair(model_family, MODELS)

    # Canonical production must pass the SAME fail-closed integrity contract as
    # create_oh_predictor / create_nps_predictor — there is no weaker
    # production-loading path.
    if _is_production_family(model_family):
        verify_production_artifact(oh_path, is_oh=True)
        verify_production_artifact(nps_path, is_oh=False)

    # 2. Load OH bundle
    oh_predictor = OperationalHealthPredictor()
    oh_predictor.load_model(str(oh_path))

    _validate_oh_bundle(oh_predictor, oh_path)

    # 3. Load NPS bundle
    nps_predictor = NPSPredictor()
    nps_predictor.load_model(str(nps_path))

    _validate_nps_bundle(nps_predictor, nps_path)

    return oh_predictor, nps_predictor


def _validate_oh_bundle(predictor, path):
    """Validate that an OH model bundle has the expected structure."""
    # Cold-start mode stores an ensemble in ``_all_models`` with ``model=None``.
    # A bundle is valid if either a single model or an ensemble is present.
    if predictor.model is None and not getattr(predictor, "_all_models", {}):
        raise ModelPairError(
            f"Invalid OH model bundle at {path}: no model or ensemble present"
        )
    if not predictor.trained:
        raise ModelPairError(
            f"OH model at {path} is not trained"
        )
    if not predictor.feature_names:
        raise ModelPairError(
            f"OH model at {path} has no feature_names"
        )


def _validate_nps_bundle(predictor, path):
    """Validate that an NPS model bundle has the expected structure."""
    if predictor.model is None and not getattr(predictor, "_all_models", {}):
        raise ModelPairError(
            f"Invalid NPS model bundle at {path}: no model or ensemble present"
        )
    if not predictor.trained:
        raise ModelPairError(
            f"NPS model at {path} is not trained"
        )
    if not predictor.feature_names:
        raise ModelPairError(
            f"NPS model at {path} has no feature_names"
        )
