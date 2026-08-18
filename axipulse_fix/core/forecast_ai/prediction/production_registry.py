"""Canonical production model registration.

Bridges the V2 model-family convention (``{family}_OH.pkl`` /
``{family}_NPS.pkl``) and the pre-V2 legacy artifact names
(``operation_health_predictor.joblib`` / ``nps_predictor_model.pkl``).

Production isolation + integrity contract:
  * A production promotion (``register_production``) validates that the
    candidate family is production-safe BEFORE copying anything.
      - test / stress / staging / smoke families are REJECTED.
      - legacy-only artifacts are REJECTED as production.
      - a family whose production role cannot be established is REJECTED.
  * Promotion STAGES the complete OH+NPS+legacy set and the manifest, then
    atomically activates it (``os.replace``). A failure before activation
    leaves the existing production pair and manifest untouched.
  * The manifest records per-artifact SHA-256 plus production provenance
    (source family, role, algorithm, features, output dims, version).  An
    empty ``source`` is never an acceptable production provenance state.
  * Legacy mirrors (``nps_predictor_model.pkl``) are created ONLY for
    explicit backward compatibility and are marked ``legacy: true`` so they
    can never be mistaken for canonical production.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .model_selector import (
    MODELS_DIR,
    NPS_LEGACY,
    NPS_SUFFIX,
    OH_LEGACY,
    OH_SUFFIX,
    PRODUCTION_FAMILY,
    ModelPairError,
    validate_model_pair,
)

logger = logging.getLogger(__name__)

# Legacy manifest (kept in sync with ``api.services.nps_service``).
MANIFEST_NAME = "manifest.json"

# Canonical artifact roles.
ROLE_PRODUCTION = "production"
ROLE_CANDIDATE = "candidate"
ROLE_TEST = "test"
ROLE_STRESS = "stress"
ROLES = {ROLE_PRODUCTION, ROLE_CANDIDATE, ROLE_TEST, ROLE_STRESS}

# Families whose name markers mark them as non-production (test/stress/
# staging/smoke).  Promotion of these is refused.
UNSAFE_FAMILY_MARKERS = ("smoke", "test", "stress", "staging", "tmp")


# Root dir that holds immutable, complete production generations.
_GENERATIONS_DIRNAME = "production_generations"

# Name of the symlink that points at the currently-active generation.
_CURRENT_LINKNAME = "current"


def _models_dir(models_dir=None) -> Path:
    return Path(models_dir) if models_dir else MODELS_DIR


def generations_dir(models_dir=None) -> Path:
    """Directory holding one immutable subdirectory per production generation."""
    return _models_dir(models_dir) / _GENERATIONS_DIRNAME


def _new_generation_id() -> str:
    import datetime as _dt
    return _dt.datetime.now().strftime("%Y%m%d%H%M%S%f")


def _resolve_through_generation(d: Path, filename: str) -> Path:
    """Return ``d/filename``, resolving through the active generation when one
    exists.  The canonical on-disk path (``models/<filename>``) is a symlink
    into ``production_generations/current/`` when generations are in use, so
    existing consumers reading the canonical path transparently observe the
    active (complete) generation.
    """
    path = d / filename
    if not path.is_symlink():
        return path
    try:
        target = path.resolve(strict=True)
    except Exception:
        return path
    return target if target.exists() else path


def production_paths(models_dir=None) -> Tuple[Path, Path]:
    """Return the canonical production ``(oh, nps)`` artifact paths.

    The canonical paths are ``models/production_OH.pkl`` /
    ``models/production_NPS.pkl`` (which, after promotion, are symlinks into
    the active immutable generation directory).  Consumers reading these paths
    always observe a complete generation.
    """
    d = _models_dir(models_dir)
    return _resolve_through_generation(d, f"{PRODUCTION_FAMILY}{OH_SUFFIX}"), \
           _resolve_through_generation(d, f"{PRODUCTION_FAMILY}{NPS_SUFFIX}")


def legacy_paths(models_dir=None) -> Tuple[Path, Path]:
    """Return the legacy ``(oh, nps)`` artifact paths (symlinked into the
    active generation when generations are in use)."""
    d = _models_dir(models_dir)
    return _resolve_through_generation(d, OH_LEGACY), \
           _resolve_through_generation(d, NPS_LEGACY)


def manifest_path(models_dir=None) -> Path:
    """Path to the active generation's manifest (or the on-disk manifest)."""
    d = _models_dir(models_dir)
    return _resolve_through_generation(d, MANIFEST_NAME)


def active_generation_dir(models_dir=None) -> Optional[Path]:
    """Resolve the currently-active generation directory (via symlink), or None."""
    link = generations_dir(models_dir) / _CURRENT_LINKNAME
    if not link.is_symlink():
        if link.is_file():  # pointer file fallback
            try:
                gen_id = link.read_text(encoding="utf-8").strip()
            except Exception:
                return None
            target = generations_dir(models_dir) / gen_id
            return target if target.is_dir() else None
        return None
    try:
        target = link.resolve(strict=True)
    except Exception:
        return None
    return target if target.is_dir() else None


def _artifact_role(name: str) -> str:
    """Resolve an artifact's role from its filename.

    Explicit role prefixes (``production_``, ``candidate_``, ``test_``,
    ``stress_``) are authoritative.  A plain ``{family}_OH.pkl`` /
    ``{family}_NPS.pkl`` pair has no role and is reported as ``unknown``.
    """
    lower = name.lower()
    if lower.startswith(ROLE_PRODUCTION + "_"):
        return ROLE_PRODUCTION
    if lower.startswith(ROLE_CANDIDATE + "_"):
        return ROLE_CANDIDATE
    if lower.startswith(ROLE_TEST + "_"):
        return ROLE_TEST
    if lower.startswith(ROLE_STRESS + "_"):
        return ROLE_STRESS
    return "unknown"


def _is_unsafe_family(family: str) -> bool:
    """Whether a candidate family name marks it as non-production.

    A name containing test/stress/staging/smoke/tmp markers is unsafe to
    promote.  This is a conservative guard in ADDITION to the artifact-role
    check; it never weakens isolation.
    """
    low = str(family or "").strip().lower()
    if not low:
        return True
    if low.startswith("."):
        return True
    return any(marker in low for marker in UNSAFE_FAMILY_MARKERS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_metadata(path: Path) -> Dict[str, Any]:
    """Extract shallow metadata (algorithm, version, features, output dims,
    runtime) from a serialised model bundle.

    Handles BOTH dict-shaped bundles (``joblib.load`` returns a dict for the
    NPS/OH production artifacts) and attribute-based bundle objects.

    FAIL CLOSED: a promotion candidate that cannot be read/deserialised is an
    error, never silently treated as empty metadata.  ``joblib.load`` failures
    propagate so promotion aborts.
    """
    import joblib
    obj = joblib.load(str(path))
    out: Dict[str, Any] = {}

    def _get(key: str):
        # Bundle is a dict -> use .get; otherwise use getattr.
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    for key, attr in (
        ("algorithm", "model_name"),
        ("model_version", "version"),
        ("feature_metadata", "feature_names"),
        ("runtime", "runtime"),
    ):
        val = _get(attr)
        if val is not None:
            out[key] = val
    model = _get("model")
    if model is not None:
        nf = getattr(model, "n_features_in_", None)
        if nf is not None:
            out["n_features_in"] = int(nf)
        nout = getattr(model, "n_outputs_", None)
        if nout is not None:
            out["n_outputs_"] = int(nout)
    trained = _get("trained")
    if trained is not None:
        out["trained"] = bool(trained)
    # Runtime provenance: library_versions (incl. python) live in the bundle's
    # ``metadata`` dict.  Carry them through so the manifest records the
    # runtime the artifact was built under.
    meta = _get("metadata")
    if isinstance(meta, dict) and meta.get("library_versions"):
        out["library_versions"] = meta["library_versions"]
        py = meta["library_versions"].get("python")
        if py:
            out["python_version"] = py
    return out


def _validate_candidate_bundle(kind: str, meta: Dict[str, Any], path: Path) -> None:
    """Validate a promotion candidate is loadable and trained.

    FAIL CLOSED: a candidate must carry an explicit ``trained=True``.  A bundle
    whose ``trained`` attribute is absent is treated as untrained (a production
    candidate must positively assert it is trained).
    """
    if not path.exists():
        raise ModelPairError(
            f"{kind} candidate bundle missing: {path.name}"
        )
    if not meta.get("trained") is True:
        raise ModelPairError(
            f"{kind} candidate bundle at {path.name} is not explicitly trained; refusing."
        )


def _build_manifest(
    d: Path,
    *,
    oh_staged: Path,
    nps_staged: Path,
    oh_legacy_staged: Optional[Path],
    nps_legacy_staged: Optional[Path],
    source_family: str,
    role: str,
    algorithm: Optional[str],
    model_version: Optional[str],
    feature_meta: Optional[Any],
    runtime: Optional[Any],
    training_meta: Optional[Any],
    n_features_in: Optional[int],
    n_outputs: Optional[int],
) -> Dict[str, Dict[str, Any]]:
    """Build the production manifest for the STAGED artifact set.

    Production entries require a non-empty ``source`` (never ``""``).  If the
    source family cannot be established the manifest build fails closed.
    """
    if not source_family or not str(source_family).strip():
        raise ModelPairError(
            "Production promotion requires a non-empty source family; "
            "refusing to write empty provenance."
        )

    manifest: Dict[str, Dict[str, Any]] = {}

    def _entry(artifact_name: str, path: Path, is_legacy: bool = False) -> None:
        if not path.exists():
            return
        # Per-artifact metadata from THIS artifact's own bundle (OH and NPS
        # differ: OH=CatBoost/19 features, NPS=XGBoost/34 features).  The
        # artifact's own values are authoritative; the shared args are only a
        # fallback when the artifact doesn't carry them.
        own = _bundle_metadata(path)
        entry: Dict[str, Any] = {
            "sha256": _sha256(path),
            "source": source_family,
            "role": "legacy" if is_legacy else role,
        }
        if own.get("algorithm") is not None:
            entry["algorithm"] = own["algorithm"]
        elif algorithm is not None:
            entry["algorithm"] = algorithm
        if model_version is not None:
            entry["model_version"] = model_version
        if own.get("feature_metadata") is not None:
            entry["feature_metadata"] = json.dumps(own["feature_metadata"])
        elif feature_meta is not None:
            entry["feature_metadata"] = json.dumps(feature_meta)
        feat_count = own.get("n_features_in") if own.get("n_features_in") is not None else n_features_in
        if feat_count is not None:
            entry["n_features_in"] = feat_count
        out_count = own.get("n_outputs_") if own.get("n_outputs_") is not None else n_outputs
        if out_count is not None:
            entry["n_outputs"] = out_count
        if own.get("python_version"):
            entry["python_version"] = own["python_version"]
        if own.get("library_versions"):
            entry["library_versions"] = own["library_versions"]
        if runtime is not None:
            entry["runtime"] = json.dumps(runtime)
        if training_meta is not None:
            entry["training_metadata"] = json.dumps(training_meta)
        if is_legacy:
            entry["legacy"] = True
        manifest[artifact_name] = entry

    _entry(oh_staged.name.replace("__staged__", ""), oh_staged)
    _entry(nps_staged.name.replace("__staged__", ""), nps_staged)
    if oh_legacy_staged is not None:
        _entry(oh_legacy_staged.name.replace("__staged__", ""), oh_legacy_staged, is_legacy=True)
    if nps_legacy_staged is not None:
        _entry(nps_legacy_staged.name.replace("__staged__", ""), nps_legacy_staged, is_legacy=True)

    # Canonical production must always be present and must NOT be legacy.
    for name in (f"{PRODUCTION_FAMILY}{OH_SUFFIX}", f"{PRODUCTION_FAMILY}{NPS_SUFFIX}"):
        if name not in manifest:
            raise ModelPairError(
                f"Production manifest missing required artifact {name}; refusing."
            )
        if manifest[name].get("legacy"):
            raise ModelPairError(
                f"Production artifact {name} cannot be a legacy entry."
            )
        if not manifest[name].get("source"):
            raise ModelPairError(
                f"Production artifact {name} has empty provenance; refusing."
            )

    return manifest


def register_production(
    family: str,
    models_dir=None,
    role: str = ROLE_PRODUCTION,
    algorithm: Optional[str] = None,
    model_version: Optional[str] = None,
    feature_meta: Optional[Any] = None,
    runtime: Optional[Any] = None,
    training_meta: Optional[Any] = None,
) -> Dict[str, str]:
    """Atomically promote a trained family to the production OH+NPS pair.

    Validation performed before any production state changes:
      1. Candidate family is production-safe (rejects test/stress/staging/
         smoke/legacy-only families).
      2. Complete OH+NPS pair exists.
      3. Both bundles load and are trained.
      4. Non-empty source provenance.

    The complete set (production OH+NPS + legacy mirrors + manifest) is staged
    in a temporary directory and then activated atomically via ``os.replace``.
    A failure before activation leaves the existing production artifacts and
    manifest untouched.

    Raises :class:`ModelPairError` on any validation/staging failure.
    """
    if role != ROLE_PRODUCTION:
        raise ModelPairError(
            f"register_production only promotes to role 'production', got {role!r}."
        )

    d = _models_dir(models_dir)
    oh_prod, nps_prod = production_paths(d)
    oh_legacy, nps_legacy = legacy_paths(d)

    # 1. Role / provenance validation.
    if not family or not str(family).strip():
        raise ModelPairError(
            "Production promotion requires a non-empty source family."
        )
    if _is_unsafe_family(family):
        raise ModelPairError(
            f"Refusing to promote family '{family}': it is marked as a "
            f"test/stress/staging/smoke artifact."
        )
    if family == NPS_LEGACY or family == OH_LEGACY:
        raise ModelPairError(
            f"Refusing to promote legacy-only artifact family '{family}'."
        )

    # 2. Resolve + validate the candidate pair.
    validate_model_pair(family, d)
    oh_src = d / f"{family}{OH_SUFFIX}"
    nps_src = d / f"{family}{NPS_SUFFIX}"
    oh_meta = _bundle_metadata(oh_src)
    nps_meta = _bundle_metadata(nps_src)
    _validate_candidate_bundle("OH", oh_meta, oh_src)
    _validate_candidate_bundle("NPS", nps_meta, nps_src)

    algo = algorithm or oh_meta.get("algorithm")
    ver = model_version or oh_meta.get("model_version")
    feats = feature_meta if feature_meta is not None else oh_meta.get("feature_metadata")
    rt = runtime if runtime is not None else oh_meta.get("runtime")
    n_features = oh_meta.get("n_features_in")
    n_outputs = oh_meta.get("n_outputs_")

    # 3. Stage the complete generation in a temp directory on the same
    #    filesystem, then atomically swap the active-generation pointer so a
    #    reader observes the previous OR the new complete generation — never a
    #    mixed OH/NPS/manifest set.
    gens_dir = generations_dir(d)
    gen_id = _new_generation_id()
    staged_gen = gens_dir / f".staging_{gen_id}"
    gens_dir.mkdir(parents=True, exist_ok=True)
    if staged_gen.exists():
        shutil.rmtree(staged_gen, ignore_errors=True)
    staged_gen.mkdir()

    oh_name = f"{PRODUCTION_FAMILY}{OH_SUFFIX}"
    nps_name = f"{PRODUCTION_FAMILY}{NPS_SUFFIX}"
    final_gen = None
    try:
        # Stage every artifact of the new generation (OH, NPS, legacy mirrors).
        shutil.copyfile(oh_src, staged_gen / oh_name)
        shutil.copyfile(nps_src, staged_gen / nps_name)
        if not (str(d / OH_LEGACY) == str(oh_src) or str(d / NPS_LEGACY) == str(nps_src)):
            shutil.copyfile(oh_src, staged_gen / OH_LEGACY)
            shutil.copyfile(nps_src, staged_gen / NPS_LEGACY)

        # Build + validate the manifest for the staged generation.
        manifest_data = _build_manifest(
            staged_gen,
            oh_staged=staged_gen / oh_name,
            nps_staged=staged_gen / nps_name,
            oh_legacy_staged=staged_gen / OH_LEGACY if (staged_gen / OH_LEGACY).exists() else None,
            nps_legacy_staged=staged_gen / NPS_LEGACY if (staged_gen / NPS_LEGACY).exists() else None,
            source_family=family,
            role=role,
            algorithm=algo,
            model_version=ver,
            feature_meta=feats,
            runtime=rt,
            training_meta=training_meta,
            n_features_in=n_features,
            n_outputs=n_outputs,
        )
        (staged_gen / MANIFEST_NAME).write_text(
            json.dumps(manifest_data, indent=2), encoding="utf-8"
        )

        # Final directory name (immutable once activated).
        final_gen = gens_dir / gen_id
        if final_gen.exists():
            shutil.rmtree(final_gen, ignore_errors=True)
        os.replace(staged_gen, final_gen)

        # Ensure the canonical root symlinks exist (they point THROUGH the
        # ``current`` pointer).  This is idempotent and runs BEFORE the pointer
        # swap, so it never mutates the currently-active set mid-promotion.
        _link_canonical_to_generation(d, final_gen)

        # Single atomic pointer swap: point "current" at the new generation.
        new_link = gens_dir / f".current_{gen_id}"
        try:
            new_link.symlink_to(final_gen.name, target_is_directory=True)
            link = gens_dir / _CURRENT_LINKNAME
            os.replace(new_link, link)
        except (OSError, NotImplementedError):
            # Platform without symlink support: fall back to a pointer file.
            link = gens_dir / _CURRENT_LINKNAME
            tmp_pointer = gens_dir / f".pointer_{gen_id}"
            tmp_pointer.write_text(gen_id, encoding="utf-8")
            os.replace(tmp_pointer, link)

        logger.info(
            "Promoted production family '%s' (generation %s).",
            family, gen_id,
        )
    except Exception:
        # No partial state: if activation (the pointer swap) failed, the
        # previous ``current`` pointer still resolves to the previous complete
        # generation.  Remove any newly-staged / newly-immutable generation dir.
        shutil.rmtree(staged_gen, ignore_errors=True)
        if final_gen is not None:
            shutil.rmtree(final_gen, ignore_errors=True)
        raise

    # Canonical paths resolve through the active generation.
    oh_prod = _resolve_through_generation(d, oh_name)
    nps_prod = _resolve_through_generation(d, nps_name)
    oh_legacy = _resolve_through_generation(d, OH_LEGACY)
    nps_legacy = _resolve_through_generation(d, NPS_LEGACY)

    return {
        "family": PRODUCTION_FAMILY,
        "source_family": family,
        "role": role,
        "generation": gen_id,
        "oh_path": str(oh_prod),
        "nps_path": str(nps_prod),
        "oh_legacy": str(oh_legacy),
        "nps_legacy": str(nps_legacy),
    }


def _link_canonical_to_generation(d: Path, gen_dir: Path) -> None:
    """Point the canonical root artifact paths at the active generation.

    ``models/production_OH.pkl`` etc. become symlinks into
    ``production_generations/current/`` so all existing consumers (loaders,
    tests) transparently read the complete, immutable generation.

    This is IDEMPOTENT: a canonical path that is ALREADY a symlink into the
    generations directory is left untouched (it resolves through ``current``
    and follows the active generation).  A canonical path that is a real file
    (pre-generation layout) is atomically replaced with a symlink via
    ``os.replace`` of a temp symlink — a single per-path atomic swap, done
    BEFORE the pointer swap, so it never mutates the active set mid-promotion.
    """
    active = generations_dir(d) / _CURRENT_LINKNAME
    for filename in (
        f"{PRODUCTION_FAMILY}{OH_SUFFIX}",
        f"{PRODUCTION_FAMILY}{NPS_SUFFIX}",
        OH_LEGACY,
        NPS_LEGACY,
        MANIFEST_NAME,
    ):
        if not (gen_dir / filename).exists():
            continue
        canonical = d / filename
        # Already a symlink into the generations dir? Leave it (idempotent).
        if canonical.is_symlink():
            try:
                if str(canonical.resolve()).startswith(str(generations_dir(d))):
                    continue
            except Exception:
                pass
        # Otherwise replace the real file (or missing path) with a symlink
        # through ``current``, atomically.
        target = active / filename
        tmp = d / f".link_{filename}"
        if tmp.exists() or tmp.is_symlink():
            try:
                tmp.unlink()
            except OSError:
                pass
        try:
            tmp.symlink_to(target)
            os.replace(tmp, canonical)
        except (OSError, NotImplementedError):
            # Platform without symlink support: fall back to a hard copy.
            try:
                tmp.unlink()
            except OSError:
                pass
            shutil.copyfile(gen_dir / filename, canonical)


def write_manifest(
    models_dir=None,
    source_family: Optional[str] = None,
    role: str = ROLE_PRODUCTION,
    algorithm: Optional[str] = None,
    model_version: Optional[str] = None,
    feature_meta: Optional[Any] = None,
    runtime: Optional[Any] = None,
    training_meta: Optional[Any] = None,
) -> Path:
    """Rewrite ``manifest.json`` for the current on-disk production artifacts.

    This is a diagnostic/regeneration helper.  It reads hashes from the actual
    production files and records provenance.  ``register_production`` is the
    authoritative promotion path (staged + atomic); ``write_manifest`` simply
    refreshes the manifest from whatever production artifacts exist.
    """
    d = _models_dir(models_dir)
    oh_prod, nps_prod = production_paths(d)
    _, nps_legacy = legacy_paths(d)
    mpath = manifest_path(d)

    # Regeneration helper: when no source_family is supplied, inherit the
    # provenance already recorded in the existing manifest (so a re-write
    # never turns previously-promoted production into empty provenance).
    if not source_family:
        try:
            existing = json.loads(mpath.read_text(encoding="utf-8"))
            for name in (f"{PRODUCTION_FAMILY}{OH_SUFFIX}", f"{PRODUCTION_FAMILY}{NPS_SUFFIX}"):
                src = (existing.get(name) or {}).get("source")
                if src:
                    source_family = src
                    break
        except Exception:
            source_family = None

    oh_meta = _bundle_metadata(oh_prod)
    nps_meta = _bundle_metadata(nps_prod)

    manifest_data = _build_manifest(
        d,
        oh_staged=oh_prod,
        nps_staged=nps_prod,
        oh_legacy_staged=oh_prod if False else None,
        nps_legacy_staged=nps_legacy if nps_legacy.exists() else None,
        source_family=source_family or "",
        role=role,
        algorithm=algorithm or oh_meta.get("algorithm"),
        model_version=model_version or oh_meta.get("model_version"),
        feature_meta=feature_meta if feature_meta is not None else oh_meta.get("feature_metadata"),
        runtime=runtime if runtime is not None else oh_meta.get("runtime"),
        training_meta=training_meta,
        n_features_in=oh_meta.get("n_features_in"),
        n_outputs=oh_meta.get("n_outputs_"),
    )

    d.mkdir(parents=True, exist_ok=True)
    tmp = d / ".manifest.json.tmp"
    tmp.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    os.replace(tmp, mpath)
    logger.info("Wrote model integrity manifest: %s", mpath)
    return mpath
