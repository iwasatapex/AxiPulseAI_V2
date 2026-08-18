"""
Model selection utilities for the V2 training/model-selection workflow.

Provides:
  - Training file discovery and interactive selection.
  - Model family discovery and interactive selection.
  - Pair validation (OH + NPS must exist together).

This module is **pure infrastructure** — it knows about *paths* and *names*
only.  It never loads a trained estimator or changes any model mathematics.
Model loading / validation of the bundle contents lives in
``predictor_config.load_model_pair``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# Path roots
# ============================================================

ROOT = Path(__file__).resolve().parents[3]
TRAINING_DIR = ROOT / "training"
MODELS_DIR = ROOT / "models"

# ------------------------------------------------------------
# Naming convention (V2)
# ------------------------------------------------------------
OH_SUFFIX = "_OH.pkl"
NPS_SUFFIX = "_NPS.pkl"

# Legacy model filenames (backward compatibility for pre-V2 models)
OH_LEGACY = "operation_health_predictor.joblib"
NPS_LEGACY = "nps_predictor_model.pkl"

# Canonical production model family name. The production OH+NPS pair is the
# matched model set that also backs the legacy filenames (see
# ``production.register_production``).
PRODUCTION_FAMILY = "production"


# ============================================================
# Exceptions
# ============================================================

class ModelPairError(Exception):
    """Raised when a model pair is missing, mismatched, or incomplete."""


# ============================================================
# Helpers
# ============================================================

def _training_dir(training_dir=None) -> Path:
    return Path(training_dir) if training_dir else TRAINING_DIR


def _models_dir(models_dir=None) -> Path:
    return Path(models_dir) if models_dir else MODELS_DIR


# ============================================================
# Training-file discovery
# ============================================================

def list_training_files(training_dir=None) -> List[Path]:
    """Return **all** files in the training directory, sorted by name.

    Hidden files (starting with ``.``) are included because the task
    requires listing *ALL* files in the training folder.

    Returns a list of :class:`pathlib.Path` objects (empty if the
    directory does not exist or is empty).
    """
    d = _training_dir(training_dir)
    if not d.exists() or not d.is_dir():
        return []
    return sorted([f for f in d.iterdir() if f.is_file()])


# ============================================================
# Model-family discovery
# ============================================================






# ============================================================
# Model-family discovery
# ============================================================

def list_model_families(models_dir=None) -> List[str]:
    """Return model-family names that have **both** ``_OH.pkl`` and ``_NPS.pkl``.

    A family name is the stem shared by the pair
    ``{family}_OH.pkl`` / ``{family}_NPS.pkl``.  Families where only
    one side exists are **not** returned (they are incomplete).
    """
    d = _models_dir(models_dir)
    if not d.exists() or not d.is_dir():
        return []

    families: List[str] = []
    seen: set = set()

    for f in d.iterdir():
        if not f.is_file():
            continue
        name = f.name
        if name.endswith(OH_SUFFIX):
            family = name[: -len(OH_SUFFIX)]
            if family not in seen:
                nps_path = d / f"{family}{NPS_SUFFIX}"
                if nps_path.exists():
                    families.append(family)
                seen.add(family)
        elif name.endswith(NPS_SUFFIX):
            family = name[: -len(NPS_SUFFIX)]
            if family not in seen:
                oh_path = d / f"{family}{OH_SUFFIX}"
                if oh_path.exists():
                    families.append(family)
                seen.add(family)

    return sorted(families)


# ============================================================
# Pair validation
# ============================================================

def validate_model_pair(family, models_dir=None) -> Tuple[Path, Path]:
    """Validate that a complete OH+NPS pair exists for *family*.

    Checks:
      * ``{family}_OH.pkl`` exists
      * ``{family}_NPS.pkl`` exists

    Returns ``(oh_path, nps_path)`` on success.
    Raises :class:`ModelPairError` if the pair is incomplete.
    """
    d = _models_dir(models_dir)

    oh_path = d / f"{family}{OH_SUFFIX}"
    nps_path = d / f"{family}{NPS_SUFFIX}"

    if not oh_path.exists() and not nps_path.exists():
        raise ModelPairError(
            f"Model family '{family}' not found in {d}. "
            f"Expected both {oh_path.name} and {nps_path.name}."
        )
    if not oh_path.exists():
        raise ModelPairError(
            f"OH model file missing for family '{family}': "
            f"expected {oh_path.name} in {d}."
        )
    if not nps_path.exists():
        raise ModelPairError(
            f"NPS model file missing for family '{family}': "
            f"expected {nps_path.name} in {d}."
        )

    return oh_path, nps_path


# ============================================================
# Interactive selection
# ============================================================

def select_training_file(training_dir=None, input_fn=input):
    """List all training files and prompt the user to choose ONE.

    Returns the selected :class:`pathlib.Path`.
    Raises ``FileNotFoundError`` when no training files exist.
    """
    files = list_training_files(training_dir)

    if not files:
        raise FileNotFoundError(
            f"No files found in {training_dir or TRAINING_DIR}"
        )

    print("\nAvailable training datasets:")
    print("─" * 70)
    for i, f in enumerate(files):
        print(f"  [{i}] {f.name}")
    print("─" * 70)

    while True:
        raw = input_fn(
            f"\nSelect a dataset (0-{len(files) - 1}): "
        ).strip()

        if not raw:
            print("Selection cannot be empty.")
            continue

        try:
            idx = int(raw)
        except ValueError:
            print("Enter a valid number.")
            continue

        if 0 <= idx < len(files):
            selected = files[idx]
            print(f"✅ Selected: {selected.name}")
            return selected

        print(f"Enter a number between 0 and {len(files) - 1}.")


def select_model_family(models_dir=None, input_fn=input):
    """List complete model families and prompt the user to choose ONE.

    Only families that have **both** ``_OH.pkl`` and ``_NPS.pkl`` are
    shown.

    Returns the family-name string.
    Raises ``ValueError`` when no complete families are found.
    """
    families = list_model_families(models_dir)

    if not families:
        d = _models_dir(models_dir)
        raise ValueError(
            f"No complete model pairs found in {d}. "
            f"A model family requires both '{{family}}_OH.pkl' and "
            f"'{{family}}_NPS.pkl'."
        )

    print("\nAvailable model families (complete OH+NPS pairs):")
    print("─" * 70)
    for i, fam in enumerate(families):
        print(f"  [{i}] {fam}")
    print("─" * 70)

    while True:
        raw = input_fn(
            f"\nSelect a model family (0-{len(families) - 1}): "
        ).strip()

        if not raw:
            print("Selection cannot be empty.")
            continue

        try:
            idx = int(raw)
        except ValueError:
            print("Enter a valid number.")
            continue

        if 0 <= idx < len(families):
            selected = families[idx]
            print(f"✅ Selected model family: {selected}")
            return selected

        print(f"Enter a number between 0 and {len(families) - 1}.")
