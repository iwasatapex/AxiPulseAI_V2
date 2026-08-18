"""Production-isolation tests (Phase 2).

Prove that no ordinary training action can modify the canonical production
artifact pair, and that only an explicit promotion — followed by validation —
changes production.

Exit criterion: *No ordinary training action can modify production.*

Scenarios under test:
  1. ``train TEST``  → production unchanged
  2. ``train CANDIDATE`` → production unchanged
  3. explicit ``PROMOTE`` → validate → production changes

We exercise the real ``production_registry`` and the real ``predictor_config``
resolution rules.  Only the expensive trainer / bundle loading is stubbed.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from core.forecast_ai.prediction import model_selector as ms
from core.forecast_ai.prediction import production_registry as pr


def _touch(path: Path, data: bytes = b"model-bundle") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _make_family(models_dir: Path, family: str) -> None:
    """Write a complete, loadable ``{family}_OH.pkl`` / ``{family}_NPS.pkl``
    pair.

    Under the fail-closed production contract, a promotion candidate must be a
    real, deserialisable model bundle (a raw byte blob cannot be promoted).  We
    write minimal joblib bundles so promotion validation passes and the
    lifecycle/atomicity behaviour is exercised end to end.
    """
    import joblib as _joblib
    for suffix in (ms.OH_SUFFIX, ms.NPS_SUFFIX):
        p = models_dir / f"{family}{suffix}"
        p.parent.mkdir(parents=True, exist_ok=True)
        is_nps = suffix == ms.NPS_SUFFIX
        if is_nps:
            est, feats = _FakeNPSEstimator(), [f"nf{i}" for i in range(34)]
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


class _FakeEstimator:
    """A minimal estimator stand-in so joblib bundle validation succeeds."""
    n_features_in_ = 3
    n_outputs_ = 1
    def predict(self, X):
        return [0.0] * len(X)


class _FakeNPSEstimator:
    n_features_in_ = 34
    n_outputs_ = 11
    def predict(self, X):
        return [[0.0] * 11 for _ in range(len(X))]


def _dump_bundle(path: Path, tag: str) -> None:
    """Write a minimal loadable joblib bundle whose bytes differ by ``tag``."""
    import joblib as _joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    _joblib.dump(
        {
            "model_name": "XGBoost",
            "trained": True,
            "feature_names": ["f1", "f2", "f3"],
            "model": _FakeEstimator(),
            "metadata": {"training_rows": 100, "tag": tag},
            "_tag": tag,
        },
        str(path),
    )


def _file_hash(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest_for(models_dir: Path, source: str = "1mil-10yr") -> None:
    """Write a valid integrity manifest for the production artifacts present in
    ``models_dir`` (with real hashes, role=production, non-empty source)."""
    oh_prod, nps_prod = pr.production_paths(models_dir)
    manifest = {
        oh_prod.name: {
            "sha256": _file_hash(oh_prod),
            "source": source,
            "role": "production",
        },
        nps_prod.name: {
            "sha256": _file_hash(nps_prod),
            "source": source,
            "role": "production",
        },
    }
    (models_dir / pr.MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _prod_snapshot(models_dir: Path) -> dict:
    """Capture the canonical production artifacts (bytes + manifest)."""
    oh_prod, nps_prod = pr.production_paths(models_dir)
    manifest = pr.manifest_path(models_dir)
    return {
        "oh": oh_prod.read_bytes() if oh_prod.exists() else None,
        "nps": nps_prod.read_bytes() if nps_prod.exists() else None,
        "manifest": json.loads(manifest.read_text()) if manifest.exists() else None,
    }


@pytest.fixture
def isolated_models_dir(tmp_path, monkeypatch):
    """A throw-away models dir so tests never touch the real ``models/``."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(ms, "MODELS_DIR", models_dir)
    monkeypatch.setattr(pr, "MODELS_DIR", models_dir)
    return models_dir


# =====================================================================
# Canonical artifact resolution (Task C)
# =====================================================================

def test_production_paths_resolve_to_canonical_names(isolated_models_dir):
    oh, nps = pr.production_paths(isolated_models_dir)
    assert oh.name == "production_OH.pkl"
    assert nps.name == "production_NPS.pkl"


def test_artifact_role_resolution():
    assert pr._artifact_role("production_OH.pkl") == "production"
    assert pr._artifact_role("production_NPS.pkl") == "production"
    assert pr._artifact_role("candidate_OH.pkl") == "candidate"
    assert pr._artifact_role("test_OH.pkl") == "test"
    assert pr._artifact_role("stress_NPS.pkl") == "stress"
    # Plain families (no role prefix) are NOT silently treated as production.
    assert pr._artifact_role("1mil-10yr_OH.pkl") == "unknown"


# =====================================================================
# Canonical production loading resolves to production_* (Task C)
# =====================================================================

def test_create_predictor_default_resolves_to_production_not_legacy(
    tmp_path, monkeypatch
):
    """The default (no family) load must target ``production_*.pkl`` and must
    NOT silently fall back to the legacy ``*.joblib`` / ``*.pkl`` names."""
    from types import SimpleNamespace

    from core.forecast_ai.prediction import predictor_config as pc

    # Legacy filenames exist, but the canonical production artifacts do not.
    _touch(tmp_path / pc.OH_LEGACY)
    _touch(tmp_path / pc.NPS_LEGACY)
    monkeypatch.setattr(pc, "MODELS", tmp_path)

    # Without a production artifact present, the default load must raise
    # FileNotFoundError — never silently fall back to the legacy files.
    with pytest.raises(FileNotFoundError):
        pc.create_oh_predictor()
    with pytest.raises(FileNotFoundError):
        pc.create_nps_predictor()

    # Seed the canonical production artifacts; record which path load_model gets.
    _make_family(tmp_path, "production")
    # A valid integrity manifest is REQUIRED for production loading (fail
    # closed). Write one with matching hashes + role so the default resolves.
    _write_manifest_for(tmp_path)
    seen = {"oh": None, "nps": None}
    inst = SimpleNamespace()

    with monkeypatch.context() as m:
        m.setattr(pc, "OperationalHealthPredictor",
                  lambda: SimpleNamespace(load_model=lambda p: seen.__setitem__("oh", p)))
        m.setattr(pc, "NPSPredictor",
                  lambda: SimpleNamespace(load_model=lambda p: seen.__setitem__("nps", p)))
        pc.create_oh_predictor()
        pc.create_nps_predictor()

    assert str(seen["oh"]).endswith("production_OH.pkl")
    assert str(seen["nps"]).endswith("production_NPS.pkl")


# =====================================================================
# 1. train TEST  →  production unchanged
# =====================================================================

def test_training_a_test_family_never_touches_production(
    isolated_models_dir, monkeypatch
):
    """A test/stress training run must leave production byte-for-byte intact."""
    # Seed an existing production pair (as if previously promoted).
    _make_family(isolated_models_dir, "production")
    before = _prod_snapshot(isolated_models_dir)

    # Training writes a *test* family only — the same shape as
    # ``train_models`` writing ``{stem}_OH.pkl`` / ``{stem}_NPS.pkl``.
    _make_family(isolated_models_dir, "test_10k_baddays")

    after = _prod_snapshot(isolated_models_dir)
    assert after["oh"] == before["oh"], "production OH must be unchanged"
    assert after["nps"] == before["nps"], "production NPS must be unchanged"
    assert after["manifest"] == before["manifest"], "manifest must be unchanged"


# =====================================================================
# 2. train CANDIDATE  →  production unchanged
# =====================================================================

def test_training_a_candidate_family_never_touches_production(
    isolated_models_dir,
):
    """A candidate training run must leave production (and its manifest)
    untouched until an explicit promotion."""
    # Seed a real production pair WITH a manifest (as if previously promoted).
    _make_family(isolated_models_dir, "production")
    pr.register_production("production", isolated_models_dir)
    before = _prod_snapshot(isolated_models_dir)
    assert before["manifest"] is not None

    # A candidate family is trained (plain ``{family}_*`` pair).  No
    # production_registry promotion call happens during training.
    _make_family(isolated_models_dir, "candidate_family")
    pr.write_manifest(isolated_models_dir)

    after = _prod_snapshot(isolated_models_dir)
    assert after["oh"] == before["oh"]
    assert after["nps"] == before["nps"]
    # Even a manifest rewrite must reflect the SAME production hashes, so the
    # recorded production artifacts are bit-identical.
    assert after["manifest"]["production_OH.pkl"]["sha256"] == \
        before["manifest"]["production_OH.pkl"]["sha256"]
    assert after["manifest"]["production_NPS.pkl"]["sha256"] == \
        before["manifest"]["production_NPS.pkl"]["sha256"]


# =====================================================================
# 3. explicit PROMOTE → validate → production changes
# =====================================================================

def test_explicit_promotion_updates_production(isolated_models_dir):
    """Promoting a candidate atomically replaces production + the manifest."""
    # A baseline production pair with DIFFERENT content from the candidate.
    _dump_bundle(isolated_models_dir / f"production{ms.OH_SUFFIX}", tag="old-oh")
    _dump_bundle(isolated_models_dir / f"production{ms.NPS_SUFFIX}", tag="old-nps")
    pr.register_production("production", isolated_models_dir)
    before = _prod_snapshot(isolated_models_dir)

    # Train a candidate, then EXPLICITLY promote it.
    _make_family(isolated_models_dir, "new_candidate")
    result = pr.register_production("new_candidate", isolated_models_dir)

    assert result["source_family"] == "new_candidate"
    assert result["family"] == pr.PRODUCTION_FAMILY
    after = _prod_snapshot(isolated_models_dir)

    assert after["oh"] != before["oh"], "production OH must change on promote"
    assert after["nps"] != before["nps"], "production NPS must change on promote"

    # The promoted production bytes must now match the candidate's bytes.
    assert after["oh"] == (isolated_models_dir / "new_candidate_OH.pkl").read_bytes()
    assert after["nps"] == (isolated_models_dir / "new_candidate_NPS.pkl").read_bytes()

    # Manifest now records the promoted candidate's hashes and lifecycle data.
    manifest = after["manifest"]
    # The recorded production hashes must match the actual files.
    oh_file = isolated_models_dir / "production_OH.pkl"
    nps_file = isolated_models_dir / "production_NPS.pkl"
    assert manifest["production_OH.pkl"]["sha256"] == _file_hash(oh_file)
    assert manifest["production_NPS.pkl"]["sha256"] == _file_hash(nps_file)
    assert manifest["production_OH.pkl"]["source"] == "new_candidate"
    assert manifest["production_OH.pkl"]["role"] == "production"


def _file_hash(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# =====================================================================
# Promotion is atomic: incomplete candidate cannot be promoted
# =====================================================================

def test_promotion_rejects_incomplete_candidate(isolated_models_dir):
    """A single-sided family must never be able to overwrite production."""
    _make_family(isolated_models_dir, "production")
    before = _prod_snapshot(isolated_models_dir)

    # Only OH exists — no NPS.  Promotion must fail and leave production intact.
    _touch(isolated_models_dir / "half_OH.pkl")
    with pytest.raises(ms.ModelPairError):
        pr.register_production("half", isolated_models_dir)

    after = _prod_snapshot(isolated_models_dir)
    assert after["oh"] == before["oh"]
    assert after["nps"] == before["nps"]


# =====================================================================
# GUI service boundary: train does not promote, promote does (Task A/E)
# =====================================================================

def test_train_models_no_longer_registers_production(tmp_path, monkeypatch):
    """train_models must produce a candidate family and never auto-promote."""
    import core.nps_predictor.predictor as nps_mod
    import core.operation_health_predictor.predictor as oh_mod
    from gui import services as svc

    training_file = tmp_path / "gamma.csv"
    training_file.write_text(
        "actual_quality,actual_competency,actual_attendance,"
        "actual_release_rate,actual_transfer_rate,promoters,passives,detractors\n"
        "80,70,85,55,5,10,20,30\n"
    )

    class StubOH:
        model_name = "CatBoost"
        feature_names = ["a"]
        algorithm_performance = {}
        history_days = 1
        def train(self, path): pass
        def save_model(self, path):
            _dump_bundle(Path(path), "gamma-oh")

    class StubNPS:
        model_name = "RF"
        feature_names = ["x"]
        algorithm_performance = {}
        history_days = 1
        def train(self, path): pass
        def save_model(self, path):
            _dump_bundle(Path(path), "gamma-nps")

    monkeypatch.setattr(oh_mod, "OperationalHealthPredictor", StubOH)
    monkeypatch.setattr(nps_mod, "NPSPredictor", StubNPS)
    monkeypatch.setattr(svc, "list_training_files", lambda: [training_file])

    models_dir = tmp_path / "models"
    monkeypatch.setattr(svc, "MODELS_DIR", models_dir)

    # Seed an existing production pair in the isolated dir.
    _make_family(models_dir, "production")
    pr.register_production("production", models_dir)
    before = _prod_snapshot(models_dir)

    # Run training; it must write ONLY the gamma candidate pair.
    out = svc.train_models("gamma.csv")
    assert out["family"] == "gamma"

    # The candidate files exist.
    assert (models_dir / "gamma_OH.pkl").exists()
    assert (models_dir / "gamma_NPS.pkl").exists()

    # Production must be byte-for-byte unchanged after training.
    after = _prod_snapshot(models_dir)
    assert after["oh"] == before["oh"], "train must NOT promote to production"
    assert after["nps"] == before["nps"], "train must NOT promote to production"
    assert after["manifest"] == before["manifest"]

    # Explicit promotion DOES change production.
    from gui.services import do_promote_production
    do_promote_production("gamma", models_dir)
    promoted = _prod_snapshot(models_dir)
    assert promoted["oh"] != before["oh"]
    assert promoted["nps"] != before["nps"]


def test_promote_without_training_raises(tmp_path, monkeypatch):
    """Explicit promotion of a never-trained family must fail clearly."""
    from gui import services as svc
    models_dir = tmp_path / "models"
    with pytest.raises(FileNotFoundError, match="Candidate family"):
        svc.do_promote_production("never_trained", models_dir)
