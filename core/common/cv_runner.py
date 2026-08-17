"""Shared subprocess runner for hardened CV candidate evaluation.

Both the Operation Health and NPS trainers evaluate each candidate fold in an
isolated subprocess (via :mod:`core.common.cv_worker`) with a hard per-fold
timeout. This is the ONLY safe way to enforce a timeout without leaving
sklearn / CatBoost / XGBoost / LightGBM state corrupted in the main training
process: the child can be SIGKILLed on timeout and the parent simply discards
that candidate. No signal/alarm is ever raised in-process.
"""
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _proc_rss_mb(pid):
    """Return a worker process's current RSS in MiB, or None if unknown.

    Reads ``/proc/<pid>/status`` on Linux so the parent can enforce a hard
    per-fold RAM ceiling without touching the child's address space.
    """
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:  # noqa: BLE001 - process may have exited
        return None
    return None


def evaluate_fold_in_subprocess(
    name,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    timeout,
    metric="mae",
    heartbeat=None,
    memory_ceiling_mb=None,
    on_spawn=None,
):
    """Run a single candidate fold in a subprocess with a hard timeout.

    ``metric`` is either ``"mae"`` (single-output, OH) or ``"nps"``
    (11-bucket, NPS). Returns one of:
      {"status": "ok",           ...metric keys..., "elapsed", "peak_rss_mb",
                                   "worker_pid"}
      {"status": "error",        "error", "elapsed"}
      {"status": "timeout",      "timeout", "elapsed"}
      {"status": "memory_limit", "ceiling_mb", "peak_rss_mb", "elapsed"}
                                          (child was SIGKILLed)

    ``memory_ceiling_mb`` (optional) arms a hard per-fold RAM guard: the parent
    polls the child's RSS and, if it exceeds the ceiling, SIGKILLs the worker
    and reports ``memory_limit`` so the caller can exclude that candidate
    instead of risking an out-of-memory crash of the machine.

    ``on_spawn`` (optional) is called with the worker PID immediately after the
    child is spawned, before the fold starts running, so callers can log the
    worker PID as part of a "before fold" line.

    The fold result travels on a dedicated binary-safe temp file, NEVER on
    stdout. Child stdout/stderr are redirected to DEVNULL so that library
    logging -- e.g. XGBoost's per-iteration progress, which is written to
    stdout -- can neither contaminate the pickled result channel nor fill up
    a pipe and deadlock the child before it finishes.
    """
    worker_path = str(Path(__file__).parent / "cv_worker.py")

    # Result channel: a private temp file the child fills in, read back here.
    fd, result_path = tempfile.mkstemp(prefix="axicv_result_", suffix=".pkl")
    os.close(fd)

    payload = {
        "name": name,
        "model": model,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "metric": metric,
        "result_path": result_path,
    }

    start = time.monotonic()
    deadline = start + float(timeout)
    ceiling = (
        float(memory_ceiling_mb)
        if memory_ceiling_mb is not None
        else None
    )
    peak_rss = 0.0

    def _peak_rss_of(pid):
        nonlocal peak_rss
        rss = _proc_rss_mb(pid)
        if rss is not None:
            peak_rss = max(peak_rss, rss)
        return rss

    try:
        proc = subprocess.Popen(
            [sys.executable, worker_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        worker_pid = proc.pid

        if on_spawn is not None:
            try:
                on_spawn(worker_pid)
            except Exception:  # noqa: BLE001 - advisory callback
                pass

        try:
            proc.stdin.write(pickle.dumps(payload))
            proc.stdin.close()
        except BrokenPipeError:
            pass

        while time.monotonic() < deadline:
            if proc.poll() is not None:
                try:
                    with open(result_path, "rb") as fh:
                        result = pickle.load(fh)
                except FileNotFoundError:
                    result = None
                if isinstance(result, dict):
                    result["elapsed"] = time.monotonic() - start
                    result["worker_pid"] = worker_pid
                    result["peak_rss_mb"] = max(peak_rss, float(result.get("peak_rss_mb", 0.0)))
                    return result
                return {
                    "status": "error",
                    "error": "worker produced no result",
                    "elapsed": time.monotonic() - start,
                    "worker_pid": worker_pid,
                    "peak_rss_mb": peak_rss,
                }

            # Hard per-fold RAM guard. Kill the worker before it OOMs the box.
            if ceiling is not None:
                rss = _peak_rss_of(worker_pid)
                if rss is not None and rss > ceiling:
                    proc.kill()
                    proc.communicate()
                    return {
                        "status": "memory_limit",
                        "ceiling_mb": ceiling,
                        "peak_rss_mb": peak_rss,
                        "worker_pid": worker_pid,
                        "elapsed": time.monotonic() - start,
                    }

            if heartbeat is not None:
                heartbeat()
            time.sleep(0.25)

        # Timeout: kill the child. Cannot corrupt the parent's model state
        # because all fitting happened in the child process.
        proc.kill()
        proc.communicate()
        return {
            "status": "timeout",
            "timeout": float(timeout),
            "elapsed": time.monotonic() - start,
            "worker_pid": worker_pid,
            "peak_rss_mb": peak_rss,
        }
    except Exception as exc:  # noqa: BLE001 - never let the child hang the run
        try:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        except Exception:  # noqa: BLE001 - advisory
            pass
        return {
            "status": "error",
            "error": repr(exc),
            "elapsed": time.monotonic() - start,
        }
    finally:
        try:
            os.remove(result_path)
        except OSError:  # noqa: BLE001 - already gone / never created
            pass
