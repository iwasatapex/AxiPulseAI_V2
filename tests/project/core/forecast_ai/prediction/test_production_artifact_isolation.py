"""Production artifact isolation + release-gate regression tests.

Covers:
  1. API NPS default resolves to the canonical production artifact.
  2. Legacy/stress/test artifacts cannot become the implicit default.
  3. Production promotion rejects test/stress/legacy candidates.
  4. Production loading verifies manifest integrity and fails closed.
  5. Production promotion is atomic (no partial OH/NPS/manifest state).
  6. Every production-facing entrypoint resolves to canonical production.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from core.forecast_ai.prediction import model_selector as ms
from core.forecast_ai.prediction import production_registry as pr
from core.forecast_ai.prediction import predictor_config as pc

ROOT = Path(__file__).resolve().parents[5]
MODELS = ROOT / "models"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _touch(path: Path, data: bytes = b"model-bundle") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class _FakeEstimator:
    n_features_in_ = 3
    n_outputs_ = 1
    def predict(self, X):
        return [0.0] * len(X)


class _FakeNPSEstimator:
    n_features_in_ = 34
    n_outputs_ = 11
    def predict(self, X):
        return [[0.0] * 11 for _ in range(len(X))]


def _make_pair(models_dir: Path, family: str) -> None:
    """Write a complete, LOADABLE joblib OH+NPS pair (fail-closed promotion
    requires a real deserialisable candidate).  NPS carries 34 features / 11
    outputs so the production 11-output contract is satisfiable."""
    import joblib as _joblib
    for suffix in (ms.OH_SUFFIX, ms.NPS_SUFFIX):
        p = models_dir / f"{family}{suffix}"
        p.parent.mkdir(parents=True, exist_ok=True)
        is_nps = suffix == ms.NPS_SUFFIX
        if is_nps:
            est, feats = _FakeNPSEstimator(), [f"nps_feat_{i}" for i in range(34)]
        else:
            est, feats = _FakeEstimator(), ["f1", "f2", "f3"]
        _joblib.dump(
            {
                "model_name": "XGBoost",
                "trained": True,
                "feature_names": feats,
                "model": est,
                "metadata": {"training_rows": 100, "num_scores": 11 if is_nps else None},
            },
            str(p),
        )


@pytest.fixture
def isolated_models_dir(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(ms, "MODELS_DIR", models_dir)
    monkeypatch.setattr(pr, "MODELS_DIR", models_dir)
    monkeypatch.setattr(pc, "MODELS", models_dir)
    monkeypatch.setattr(pc, "MODELS_DIR", models_dir)
    return models_dir


# =====================================================================
# 1. API NPS default identity
# =====================================================================

@pytest.mark.unit
def test_api_nps_default_is_production_artifact():
    """NPSService() must resolve to the canonical production artifact."""
    from api.services import nps_service as ns

    assert os.path.basename(ns.DEFAULT_MODEL_PATH) == "production_NPS.pkl"
    svc = ns.NPSService()
    assert os.path.basename(svc.model_path) == "production_NPS.pkl"
    assert svc.is_loaded() is True


@pytest.mark.unit
def test_api_default_is_not_stress_or_legacy():
    """The API default must be the CANONICAL production artifact — never the
    legacy compatibility mirror and never a stress/test/candidate artifact.

    The old stress artifact (``stess_test_NPS.pkl``) was intentionally deleted
    during the 100k-10yr production promotion, so isolation is proven via
    canonical identity + active-manifest provenance + the authoritative
    production-integrity verifier instead of byte-comparing a removed file.
    """
    from api.services import nps_service as ns
    from core.forecast_ai.prediction import predictor_config as pc2

    # 1. The API default IS the canonical production artifact name.
    assert os.path.basename(ns.DEFAULT_MODEL_PATH) == "production_NPS.pkl"

    default = MODELS / os.path.basename(ns.DEFAULT_MODEL_PATH)
    prod = MODELS / "production_NPS.pkl"

    # 2. The default path resolves to the canonical production artifact bytes.
    assert default.read_bytes() == prod.read_bytes()
    assert _sha256(default) == _sha256(prod)

    # 3. The authoritative production-integrity verifier accepts it:
    #    hash, role=production, legacy unset, source set, trained, 34 in / 11 out.
    pc2.verify_production_artifact(prod, is_oh=False)

    # 4. Active-generation manifest provenance: production / 100k-10yr / not legacy.
    manifest = json.loads(pr.manifest_path().read_text(encoding="utf-8"))
    entry = manifest["production_NPS.pkl"]
    assert entry["role"] == "production"
    assert entry["source"] == "100k-10yr"
    assert not entry.get("legacy")
    assert entry["sha256"] == _sha256(prod)

    # 5. The default must resolve inside the ACTIVE production generation —
    #    it can never silently point at a stress/test/candidate artifact.
    active = pr.active_generation_dir()
    assert active is not None
    assert str(default.resolve()).startswith(str(active))

    # 6. Never the legacy compatibility mirror.
    assert os.path.abspath(default) != os.path.abspath(MODELS / ns.NPS_LEGACY)


@pytest.mark.unit
def test_api_default_uses_production_not_legacy_content():
    """The API default artifact's bytes must equal the canonical
    production_NPS.pkl artifact and the service must load that artifact.

    The legacy mirror may equal production (it is a compatibility mirror of
    the production pair); the real isolation guarantee is that the default is
    the canonical production model — never the deleted stress/test artifact.
    """
    from api.services import nps_service as ns
    from core.forecast_ai.prediction import predictor_config as pc2
    import joblib as _joblib

    default = MODELS / os.path.basename(ns.DEFAULT_MODEL_PATH)
    prod = MODELS / "production_NPS.pkl"
    assert default.read_bytes() == prod.read_bytes()

    # End-to-end: the API service actually loads the canonical production
    # artifact and reports itself ready (fail-open would surface is_loaded
    # False or a different model_path).
    svc = ns.NPSService()
    assert os.path.basename(svc.model_path) == "production_NPS.pkl"
    assert svc.is_loaded() is True

    # Authoritative production-integrity verification (hash/role/source/
    # legacy/trained + 34-feature + 11-output structural contract).
    pc2.verify_production_artifact(prod, is_oh=False)

    # Explicit canonical artifact contract from the serialised bundle itself.
    bundle = _joblib.load(str(prod))
    assert bundle.get("trained") is True
    assert bundle.get("model_name") == "XGBoost"
    assert len(bundle.get("feature_names") or []) == 34
    assert type(bundle.get("model")).__name__ == "MultiOutputRegressor"
    assert int(bundle["model"].n_features_in_) == 34
    assert int((bundle.get("metadata") or {}).get("num_scores")) == 11


@pytest.mark.unit
def test_api_explicit_legacy_is_opt_in():
    """The legacy artifact is only used when explicitly requested via model_path."""
    from api.services import nps_service as ns

    svc = ns.NPSService(model_path=str(MODELS / ns.NPS_LEGACY))
    assert os.path.basename(svc.model_path) == ns.NPS_LEGACY
    # Explicit opt-in loads it (compatibility), but it is not the default.
    assert svc.is_loaded() is True
    default = ns.NPSService()
    assert os.path.basename(default.model_path) != ns.NPS_LEGACY


# =====================================================================
# 2. Production promotion role isolation
# =====================================================================

def test_promote_test_candidate_rejected(isolated_models_dir):
    _make_pair(isolated_models_dir, "test_foo")
    with pytest.raises(ms.ModelPairError, match="test|stress|staging|smoke"):
        pr.register_production("test_foo", isolated_models_dir)


def test_promote_stress_candidate_rejected(isolated_models_dir):
    _make_pair(isolated_models_dir, "stess_test")  # contains 'test'
    with pytest.raises(ms.ModelPairError, match="test|stress|staging|smoke"):
        pr.register_production("stess_test", isolated_models_dir)


def test_promote_stress_named_candidate_rejected(isolated_models_dir):
    _make_pair(isolated_models_dir, "stress_run")
    with pytest.raises(ms.ModelPairError, match="test|stress|staging|smoke"):
        pr.register_production("stress_run", isolated_models_dir)


def test_promote_legacy_only_candidate_rejected(isolated_models_dir):
    _make_pair(isolated_models_dir, ms.NPS_LEGACY)
    with pytest.raises(ms.ModelPairError):
        pr.register_production(ms.NPS_LEGACY, isolated_models_dir)


def test_promote_valid_production_candidate_accepted(isolated_models_dir):
    _make_pair(isolated_models_dir, "1mil-10yr")
    result = pr.register_production("1mil-10yr", isolated_models_dir)
    assert result["family"] == "production"
    assert result["source_family"] == "1mil-10yr"


def test_promote_with_empty_source_rejected(isolated_models_dir):
    _make_pair(isolated_models_dir, "good_family")
    with pytest.raises(ms.ModelPairError, match="empty|non-empty|source"):
        pr.register_production("", isolated_models_dir)


def test_promotion_requires_production_role(isolated_models_dir):
    _make_pair(isolated_models_dir, "1mil-10yr")
    with pytest.raises(ms.ModelPairError, match="production"):
        pr.register_production("1mil-10yr", isolated_models_dir, role="candidate")


# =====================================================================
# 3. Production loading integrity (fail closed)
# =====================================================================

def test_load_rejects_hash_mismatch(isolated_models_dir, monkeypatch):
    _make_pair(isolated_models_dir, "production")
    (isolated_models_dir / "production_NPS.pkl").write_bytes(b"tampered")
    _write_manifest(isolated_models_dir, hashes=False)  # mismatched hashes
    from core.forecast_ai.prediction import predictor_config as pc2
    from types import SimpleNamespace
    monkeypatch.setattr(
        pc2, "NPSPredictor",
        lambda: SimpleNamespace(load_model=lambda p: None, model=None, trained=True, feature_names=["x"]),
    )
    with pytest.raises(Exception):
        pc2.create_nps_predictor()


def test_load_rejects_missing_manifest_entry(isolated_models_dir, monkeypatch):
    _make_pair(isolated_models_dir, "production")
    _write_manifest(isolated_models_dir, drop_nps=True)
    from types import SimpleNamespace
    from core.forecast_ai.prediction import predictor_config as pc2
    monkeypatch.setattr(
        pc2, "NPSPredictor",
        lambda: SimpleNamespace(load_model=lambda p: None, model=None, trained=True, feature_names=["x"]),
    )
    with pytest.raises(Exception, match="no manifest entry|refusing"):
        pc2.create_nps_predictor()


def test_load_rejects_wrong_role(isolated_models_dir, monkeypatch):
    _make_pair(isolated_models_dir, "production")
    _write_manifest(isolated_models_dir, role="test")
    from types import SimpleNamespace
    from core.forecast_ai.prediction import predictor_config as pc2
    monkeypatch.setattr(
        pc2, "NPSPredictor",
        lambda: SimpleNamespace(load_model=lambda p: None, model=None, trained=True, feature_names=["x"]),
    )
    with pytest.raises(Exception, match="role"):
        pc2.create_nps_predictor()


def test_load_rejects_empty_provenance(isolated_models_dir, monkeypatch):
    _make_pair(isolated_models_dir, "production")
    _write_manifest(isolated_models_dir, source="")
    from types import SimpleNamespace
    from core.forecast_ai.prediction import predictor_config as pc2
    monkeypatch.setattr(
        pc2, "NPSPredictor",
        lambda: SimpleNamespace(load_model=lambda p: None, model=None, trained=True, feature_names=["x"]),
    )
    with pytest.raises(Exception, match="provenance|empty"):
        pc2.create_nps_predictor()


def _write_manifest(models_dir, source="1mil-10yr", hashes=True, role="production", drop_nps=False):
    oh = models_dir / "production_OH.pkl"
    nps = models_dir / "production_NPS.pkl"
    entries = {
        "production_OH.pkl": {
            "sha256": _sha256(oh) if hashes else "0" * 64,
            "source": source, "role": role,
        },
    }
    if not drop_nps:
        entries["production_NPS.pkl"] = {
            "sha256": _sha256(nps) if hashes else "0" * 64,
            "source": source, "role": role,
        }
    (models_dir / pr.MANIFEST_NAME).write_text(json.dumps(entries), encoding="utf-8")


# =====================================================================
# 4. Atomic promotion
# =====================================================================

def test_promotion_succeeds_leaves_consistent_pair(isolated_models_dir):
    _make_pair(isolated_models_dir, "candidate_a")
    result = pr.register_production("candidate_a", isolated_models_dir)
    manifest = json.loads((isolated_models_dir / pr.MANIFEST_NAME).read_text())
    for name in ("production_OH.pkl", "production_NPS.pkl"):
        assert name in manifest
        assert manifest[name]["source"] == "candidate_a"
        assert manifest[name]["role"] == "production"
        assert manifest[name]["sha256"] == _sha256(isolated_models_dir / name)


def test_promotion_failure_does_not_partially_update(isolated_models_dir, monkeypatch):
    """If NPS copy fails, production artifacts + manifest stay untouched."""
    _make_pair(isolated_models_dir, "production")
    pr.register_production("production", isolated_models_dir)
    before_manifest = (isolated_models_dir / pr.MANIFEST_NAME).read_bytes()
    before_oh = (isolated_models_dir / "production_OH.pkl").read_bytes()
    before_nps = (isolated_models_dir / "production_NPS.pkl").read_bytes()

    _make_pair(isolated_models_dir, "new_candidate")
    # Force the NPS source copy to fail after the OH copy staged.
    monkeypatch.setattr(pr.shutil, "copyfile", _failing_copy)

    with pytest.raises(OSError):
        pr.register_production("new_candidate", isolated_models_dir)

    assert (isolated_models_dir / "production_OH.pkl").read_bytes() == before_oh
    assert (isolated_models_dir / "production_NPS.pkl").read_bytes() == before_nps
    assert (isolated_models_dir / pr.MANIFEST_NAME).read_bytes() == before_manifest


def test_manifest_failure_does_not_partially_update(isolated_models_dir, monkeypatch):
    """If manifest build fails AFTER staging copies, nothing is activated."""
    _make_pair(isolated_models_dir, "production")
    pr.register_production("production", isolated_models_dir)
    before_manifest = (isolated_models_dir / pr.MANIFEST_NAME).read_bytes()
    before_oh = (isolated_models_dir / "production_OH.pkl").read_bytes()
    before_nps = (isolated_models_dir / "production_NPS.pkl").read_bytes()

    _make_pair(isolated_models_dir, "good_family")

    def _boom_manifest(*a, **k):
        raise pr.ModelPairError("injected manifest build failure")

    monkeypatch.setattr(pr, "_build_manifest", _boom_manifest)

    with pytest.raises(pr.ModelPairError, match="manifest"):
        pr.register_production("good_family", isolated_models_dir)

    assert (isolated_models_dir / "production_OH.pkl").read_bytes() == before_oh
    assert (isolated_models_dir / "production_NPS.pkl").read_bytes() == before_nps
    assert (isolated_models_dir / pr.MANIFEST_NAME).read_bytes() == before_manifest


def _failing_copy(src, dst, *a, **k):
    if "production_NPS" in str(dst):
        raise OSError("injected NPS copy failure")
    if "NPS" in str(dst):
        raise OSError("injected NPS copy failure")
    return Path(dst)


# =====================================================================
# 5. Production entrypoint identity
# =====================================================================

def test_forecastai_predictor_defaults_to_production():
    """Canonical ForecastAI loaders resolve to production artifacts."""
    assert pc.MODELS is not None
    oh_path = MODELS / f"{ms.PRODUCTION_FAMILY}{ms.OH_SUFFIX}"
    nps_path = MODELS / f"{ms.PRODUCTION_FAMILY}{ms.NPS_SUFFIX}"
    assert oh_path.exists() and nps_path.exists()
    assert oh_path.name == "production_OH.pkl"
    assert nps_path.name == "production_NPS.pkl"


# =====================================================================
# 6. Generation-based atomic promotion (task 1)
# =====================================================================

def _active_snapshot(models_dir: Path) -> dict:
    oh, nps = pr.production_paths(models_dir)
    manifest = pr.manifest_path(models_dir)
    return {
        "oh": oh.read_bytes() if oh.exists() else None,
        "nps": nps.read_bytes() if nps.exists() else None,
        "manifest": json.loads(manifest.read_text()) if manifest.exists() else None,
    }


def test_promotion_is_generation_pointer_swap(isolated_models_dir):
    """Promotion must activate via a single generation pointer swap: a reader
    observes either the old or new complete generation."""
    _make_pair(isolated_models_dir, "production")
    r1 = pr.register_production("production", isolated_models_dir)
    assert "generation" in r1

    _make_pair(isolated_models_dir, "next_family")
    r2 = pr.register_production("next_family", isolated_models_dir)
    assert r2["generation"] != r1["generation"]

    # Canonical root paths are symlinks into the active generation.
    assert (isolated_models_dir / "production_OH.pkl").is_symlink()
    # Old generation preserved immutably alongside the new one.
    gens = [p.name for p in pr.generations_dir(isolated_models_dir).iterdir()]
    assert r1["generation"] in gens and r2["generation"] in gens
    # Active generation resolves to the latest.
    assert pr.active_generation_dir(isolated_models_dir).name == r2["generation"]


def test_activation_failure_keeps_previous_generation(isolated_models_dir, monkeypatch):
    """If the pointer swap fails, the previously-active generation must remain
    current and the promoted bytes must not leak."""
    _make_pair(isolated_models_dir, "production")
    pr.register_production("production", isolated_models_dir)
    before = _active_snapshot(isolated_models_dir)
    old_gen = pr.active_generation_dir(isolated_models_dir).name

    _make_pair(isolated_models_dir, "next_family")

    # Force the generation pointer swap to fail AFTER staging.
    real_replace = os.replace
    def _boom_replace(src, dst, *a, **k):
        if str(dst).endswith("current"):
            raise OSError("injected pointer swap failure")
        return real_replace(src, dst, *a, **k)
    monkeypatch.setattr(pr.os, "replace", _boom_replace)

    with pytest.raises(OSError, match="pointer"):
        pr.register_production("next_family", isolated_models_dir)

    # Previous generation still current; production unchanged.
    assert pr.active_generation_dir(isolated_models_dir).name == old_gen
    after = _active_snapshot(isolated_models_dir)
    assert after["oh"] == before["oh"]
    assert after["nps"] == before["nps"]
    assert after["manifest"] == before["manifest"]


def test_corrupt_candidate_rejected_and_production_untouched(isolated_models_dir):
    """A corrupt (non-loadable) candidate must be rejected and leave the
    existing production generation untouched."""
    _make_pair(isolated_models_dir, "production")
    pr.register_production("production", isolated_models_dir)
    before = _active_snapshot(isolated_models_dir)

    # A candidate whose NPS is invalid bytes.
    _make_pair(isolated_models_dir, "bad_candidate")
    (isolated_models_dir / "bad_candidate_NPS.pkl").write_bytes(b"\x00\x01corrupt-not-a-model")

    with pytest.raises(Exception):
        pr.register_production("bad_candidate", isolated_models_dir)

    after = _active_snapshot(isolated_models_dir)
    assert after["oh"] == before["oh"]
    assert after["nps"] == before["nps"]
    assert after["manifest"] == before["manifest"]


def test_corrupt_oh_candidate_rejected(isolated_models_dir):
    """A corrupt OH candidate must also fail closed."""
    _make_pair(isolated_models_dir, "production")
    pr.register_production("production", isolated_models_dir)
    before = _active_snapshot(isolated_models_dir)

    _make_pair(isolated_models_dir, "bad_oh")
    (isolated_models_dir / "bad_oh_OH.pkl").write_bytes(b"\x00corrupt")

    with pytest.raises(Exception):
        pr.register_production("bad_oh", isolated_models_dir)

    after = _active_snapshot(isolated_models_dir)
    assert after["oh"] == before["oh"]
    assert after["nps"] == before["nps"]


# =====================================================================
# 7. load_model_pair("production") integrity (task 5)
# =====================================================================

def test_load_model_pair_production_valid(isolated_models_dir, monkeypatch):
    """load_model_pair('production') succeeds with a valid, complete pair."""
    from core.forecast_ai.prediction import predictor_config as pc2
    _make_pair(isolated_models_dir, "production")
    pr.register_production("production", isolated_models_dir)
    monkeypatch.setattr(pc2, "MODELS", isolated_models_dir)
    oh, nps = pc2.load_model_pair("production")
    assert oh.trained and nps.trained


def test_load_model_pair_production_hash_mismatch(isolated_models_dir, monkeypatch):
    from core.forecast_ai.prediction import predictor_config as pc2
    _make_pair(isolated_models_dir, "production")
    pr.register_production("production", isolated_models_dir)
    monkeypatch.setattr(pc2, "MODELS", isolated_models_dir)
    (isolated_models_dir / "production_NPS.pkl").write_bytes(b"tampered")
    with pytest.raises(Exception):
        pc2.load_model_pair("production")


def test_load_model_pair_production_missing_manifest(isolated_models_dir, monkeypatch):
    """Without a manifest, load_model_pair('production') must fail closed."""
    from core.forecast_ai.prediction import predictor_config as pc2
    _make_pair(isolated_models_dir, "production")
    monkeypatch.setattr(pc2, "MODELS", isolated_models_dir)
    with pytest.raises(Exception):
        pc2.load_model_pair("production")


def test_load_model_pair_production_legacy_rejected(isolated_models_dir, monkeypatch):
    """A legacy-only artifact must be rejected as production."""
    from core.forecast_ai.prediction import predictor_config as pc2
    _make_pair(isolated_models_dir, "production")
    pr.register_production("production", isolated_models_dir)
    monkeypatch.setattr(pc2, "MODELS", isolated_models_dir)
    mpath = pr.manifest_path(isolated_models_dir)
    m = json.loads(mpath.read_text())
    m["production_NPS.pkl"]["legacy"] = True
    mpath.write_text(json.dumps(m))
    with pytest.raises(Exception):
        pc2.load_model_pair("production")


def test_relink_failure_keeps_previous_generation(isolated_models_dir, monkeypatch):
    """A failure while linking canonical symlinks (which runs BEFORE the
    pointer swap) must leave the previously-active generation current and the
    canonical paths pointing at it — the pointer swap is the single atomic
    activation step."""
    _make_pair(isolated_models_dir, "production")
    pr.register_production("production", isolated_models_dir)
    old_gen = pr.active_generation_dir(isolated_models_dir).name
    before = _active_snapshot(isolated_models_dir)

    _make_pair(isolated_models_dir, "next_family")

    # Force the canonical-link helper to fail during promotion.
    def _boom_link(*a, **k):
        raise OSError("injected canonical relink failure")
    monkeypatch.setattr(pr, "_link_canonical_to_generation", _boom_link)

    with pytest.raises(OSError, match="relink"):
        pr.register_production("next_family", isolated_models_dir)

    # Previous generation still current; production unchanged.
    assert pr.active_generation_dir(isolated_models_dir).name == old_gen
    after = _active_snapshot(isolated_models_dir)
    assert after["oh"] == before["oh"]
    assert after["nps"] == before["nps"]
    assert after["manifest"] == before["manifest"]


# =====================================================================
# P0-B: PRODUCTION MODEL SCHEMA INTEGRITY MUST FAIL CLOSED
# =====================================================================

def _write_oh_artifact_with_features(models_dir, feature_names):
    """Write a loadable OH production artifact carrying the given features."""
    import joblib as _joblib
    from tests.project.core.forecast_ai.prediction.test_production_lifecycle import (
        _FakeEstimator,
    )
    path = models_dir / "production_OH.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    _joblib.dump(
        {
            "model_name": "XGBoost",
            "trained": True,
            "feature_names": feature_names,
            "model": _FakeEstimator(),
            "metadata": {"training_rows": 100},
        },
        str(path),
    )
    return path


def _manifest_with_feature_meta(models_dir, feature_metadata):
    """Write a manifest entry whose feature_metadata is ``feature_metadata``
    (already JSON-encoded as the real registry does)."""
    oh = models_dir / "production_OH.pkl"
    entry = {
        "production_OH.pkl": {
            "sha256": _sha256(oh),
            "source": "100k-10yr",
            "role": "production",
            "n_features_in": 3,
            "n_outputs": 1,
            "feature_metadata": feature_metadata,
        },
    }
    (models_dir / pr.MANIFEST_NAME).write_text(json.dumps(entry), encoding="utf-8")


def test_valid_artifact_with_matching_schema_passes(isolated_models_dir):
    """A valid artifact whose feature names/order match the manifest passes."""
    _write_oh_artifact_with_features(isolated_models_dir, ["f1", "f2", "f3"])
    _manifest_with_feature_meta(
        isolated_models_dir, json.dumps(["f1", "f2", "f3"])
    )
    pc.verify_production_artifact(isolated_models_dir / "production_OH.pkl", is_oh=True)


def test_feature_name_mismatch_fails_closed(isolated_models_dir):
    """An artifact whose feature NAMES differ from the manifest must fail."""
    _write_oh_artifact_with_features(isolated_models_dir, ["f1", "f2", "f3"])
    _manifest_with_feature_meta(
        isolated_models_dir, json.dumps(["f1", "f2", "different"])
    )
    with pytest.raises(Exception, match="schema|refusing"):
        pc.verify_production_artifact(isolated_models_dir / "production_OH.pkl", is_oh=True)


def test_feature_order_mismatch_fails_closed(isolated_models_dir):
    """An artifact whose feature ORDER differs from the manifest must fail."""
    _write_oh_artifact_with_features(isolated_models_dir, ["f1", "f2", "f3"])
    _manifest_with_feature_meta(
        isolated_models_dir, json.dumps(["f3", "f2", "f1"])
    )
    with pytest.raises(Exception, match="schema|refusing"):
        pc.verify_production_artifact(isolated_models_dir / "production_OH.pkl", is_oh=True)


def test_malformed_schema_metadata_fails_closed(isolated_models_dir):
    """Malformed feature schema metadata in the manifest must fail closed."""
    _write_oh_artifact_with_features(isolated_models_dir, ["f1", "f2", "f3"])
    _manifest_with_feature_meta(isolated_models_dir, "{ not valid json !!!")
    with pytest.raises(Exception, match="malformed|refusing"):
        pc.verify_production_artifact(isolated_models_dir / "production_OH.pkl", is_oh=True)
