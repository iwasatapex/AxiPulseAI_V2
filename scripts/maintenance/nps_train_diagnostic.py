#!/usr/bin/env python3
"""NPS training RAM diagnostic mode.

Trains the NPS predictor on a synthetic 10k-row dataset and reports:
    - peak parent RSS
    - peak CV worker RSS
    - final-fit RSS
    - model selected
    - whether GPU was used

Used to verify that NPS candidate CV is RAM-safe (serial, bounded sample,
per-fold RAM guard) and that only the selected model receives the full-data
refit.

Usage:
    python -m scripts.maintenance.nps_train_diagnostic [--rows 10000]
"""
import argparse
import gc
import os
import resource
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.nps_predictor.config import Config  # noqa: E402
from core.nps_predictor.predictor import NPSPredictor  # noqa: E402


def _parent_peak_rss_mb():
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


def _current_rss_mb():
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        return float("nan")


def _synthetic(n_rows, n_dates=250, seed=0):
    """Build a valid NPS training frame of the given row count."""
    rng = np.random.default_rng(seed)
    dates = (
        pd.Timestamp("2024-01-01")
        + pd.to_timedelta(rng.integers(0, n_dates, size=n_rows), unit="D")
    )

    total_calls = rng.integers(200, 2000, size=n_rows).astype(np.float64)

    # score_0..score_10 must sum to total_surveys (consistency invariant).
    raw = rng.dirichlet(np.ones(11), size=n_rows)
    scores = np.round(raw * total_calls[:, None]).astype(np.float64)
    total_surveys = scores.sum(axis=1).astype(np.float64)

    df = pd.DataFrame(
        {
            "date": dates,
            "operational_health": rng.uniform(0, 120, size=n_rows),
            "business_intelligence_factor": rng.uniform(0, 100, size=n_rows),
            "member_intelligence_factor": rng.uniform(0, 100, size=n_rows),
            "target_release_rate": rng.uniform(0, 100, size=n_rows),
            "actual_release_rate": rng.uniform(0, 100, size=n_rows),
            "total_calls_received": total_calls,
            "total_surveys": total_surveys,
        }
    )
    for i in range(11):
        df[f"score_{i}"] = scores[:, i]

    df["promoters"] = df["score_9"] + df["score_10"]
    df["passives"] = df["score_7"] + df["score_8"]
    df["detractors"] = (
        df["score_0"]
        + df["score_1"]
        + df["score_2"]
        + df["score_3"]
        + df["score_4"]
        + df["score_5"]
        + df["score_6"]
    )

    return df.sort_values("date").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="NPS training RAM diagnostic mode."
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=10000,
        help="Number of synthetic training rows (default 10000).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write the synthetic CSV (else a temp file).",
    )
    args = parser.parse_args()

    rows = args.rows

    config = Config(
        verbose=False,
        use_gpu=True,
        sample_for_selection=True,
        sample_size=500,
        cv_folds=2,
        cv_n_jobs=1,
        cv_memory_ceiling_mb=2048.0,
        cv_mlp_timeout=30.0,
    )

    print(f"[diag] building synthetic NPS dataset with {rows:,} rows ...")
    df = _synthetic(rows)

    csv_path = args.out
    own_temp = False
    if csv_path is None:
        fd, csv_path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        own_temp = True
    else:
        csv_path = str(csv_path)
    df.to_csv(csv_path, index=False)
    print(f"[diag] dataset written to {csv_path}")

    peak_cv_worker = {"mb": float("nan")}
    gpu_used = {"value": None}

    try:
        predictor = NPSPredictor(config=config)

        # Instrument the real CV fold subprocess path and the GPU final-fit
        # decision for the duration of training.
        _install_instrumentation(peak_cv_worker, gpu_used)
        try:
            predictor.train(str(csv_path))
        finally:
            _restore_instrumentation()

        final_rss = _current_rss_mb()
        print("\n===== NPS TRAIN DIAGNOSTIC =====")
        print(f"rows                      : {rows:,}")
        print(f"peak parent RSS (MiB)     : {_parent_peak_rss_mb():.0f}")
        print(f"peak CV worker RSS (MiB)  : {peak_cv_worker['mb']:.0f}")
        print(f"final-fit RSS (MiB)       : {final_rss:.0f}")
        print(f"model selected            : {predictor.model_name}")
        print(f"gpu used                  : {gpu_used['value']}")
        print("=============================")

        if not (peak_cv_worker["mb"] == peak_cv_worker["mb"]):
            print(
                "[warn] peak CV worker RSS was NOT recorded; "
                "cannot claim a RAM-safe fix."
            )
            return 1
        return 0
    finally:
        if own_temp:
            try:
                Path(csv_path).unlink()
            except OSError:
                pass


_ORIG_EVAL = None
_ORIG_APPLY = None


def _install_instrumentation(peak, gpu_used):
    global _ORIG_EVAL, _ORIG_APPLY

    import core.nps_predictor.trainer as trainer_mod
    import core.nps_predictor.gpu as gpu_mod

    _ORIG_EVAL = trainer_mod._evaluate_fold_in_subprocess
    _ORIG_APPLY = gpu_mod.apply_gpu_params

    def tracked_eval(
        name,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        timeout,
        heartbeat=None,
        memory_ceiling_mb=None,
        on_spawn=None,
    ):
        res = _ORIG_EVAL(
            name,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            timeout,
            heartbeat=heartbeat,
            memory_ceiling_mb=memory_ceiling_mb,
            on_spawn=on_spawn,
        )
        rss = res.get("peak_rss_mb")
        if rss is not None and rss == rss:  # not None and not NaN
            current = peak["mb"]
            if not (current == current) or float(rss) > current:
                peak["mb"] = float(rss)
        return res

    def tracked_apply(model, model_name, config):
        applied = _ORIG_APPLY(model, model_name, config)
        gpu_used["value"] = bool(applied)
        return applied

    trainer_mod._evaluate_fold_in_subprocess = tracked_eval
    gpu_mod.apply_gpu_params = tracked_apply
    gc.collect()


def _restore_instrumentation():
    global _ORIG_EVAL, _ORIG_APPLY

    import core.nps_predictor.trainer as trainer_mod
    import core.nps_predictor.gpu as gpu_mod

    if _ORIG_EVAL is not None:
        trainer_mod._evaluate_fold_in_subprocess = _ORIG_EVAL
    if _ORIG_APPLY is not None:
        gpu_mod.apply_gpu_params = _ORIG_APPLY
    _ORIG_EVAL = None
    _ORIG_APPLY = None


if __name__ == "__main__":
    sys.exit(main())
