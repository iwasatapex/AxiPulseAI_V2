"""Train view: list datasets, pick one, train OH+NPS from the same file."""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from gui import components as c
from gui import contracts as ct
from gui import services as svc


def _train_worker(dataset_name: str, progress: List[str], lock: threading.Lock) -> None:
    try:
        result = svc.train_models(dataset_name, progress=progress, progress_lock=lock)
        with lock:
            progress.append("__DONE__" + json.dumps(result, default=str))
    except Exception as exc:  # noqa: BLE001
        with lock:
            progress.append("__ERROR__" + str(exc))


def render() -> None:
    c.page_title("Train", help_text="Train OH + NPS from one selected dataset")

    datasets = svc.list_datasets()
    if not datasets:
        c.empty_state("No files found in `training/`. Add a dataset to begin.",
                      icon="📂")
        return

    # ---- Dataset table ----
    df = pd.DataFrame([
        {
            "File": d["name"],
            "Type": d.get("type"),
            "Extension": d.get("extension"),
            "Trainable": "✓" if d.get("trainable") else "✗",
            "Size (KB)": round(d.get("size_bytes", 0) / 1024, 1),
            "Modified": (d.get("modified") or "")[:19],
        }
        for d in datasets
    ])
    st.caption("All files in `training/` (only ✓-marked formats can be trained/previewed)")
    st.dataframe(df, width="stretch", hide_index=True)

    trainable = [d for d in datasets if d.get("trainable")]
    names = [d["name"] for d in datasets]
    selected = st.selectbox("Select dataset", names, index=0)

    with st.expander("Preview dataset"):
        try:
            preview = c.guarded(svc.preview_dataset, selected)
            if preview:
                st.write("**Columns:**", ", ".join(preview["columns"]))
                st.dataframe(preview["rows"], width="stretch", hide_index=True)
        except Exception as exc:  # noqa: BLE001 - surface, never swallow
            st.error(f"Could not preview dataset: {exc}")

    st.divider()

    # ---- Train ----
    if not trainable:
        st.warning(
            "No supported training files found in `training/`. Supported "
            "formats: " + ", ".join(sorted(ct.SUPPORTED_DATASET_FORMATS))
        )
        return

    selected_info = next((d for d in trainable if d["name"] == selected), None)
    if selected_info is None:
        st.warning(
            f"`{selected}` is not a supported training format. Choose one of: "
            + ", ".join(sorted(ct.SUPPORTED_DATASET_FORMATS))
        )
        return

    run = st.button("Train OH + NPS from this dataset",
                    type="primary",
                    disabled=False,
                    help="Trains both models from the SAME selected file. "
                         "Re-training a family replaces its existing pair.")

    if run:
        progress: List[str] = []
        lock = threading.Lock()
        thread = threading.Thread(
            target=_train_worker, args=(selected, progress, lock), daemon=True
        )
        thread.start()

        logs: List[str] = []
        with st.status("Training models…", expanded=True) as status:
            while thread.is_alive():
                with lock:
                    new = list(progress)
                for line in new[len(logs):]:
                    if not line.startswith("__"):
                        status.write(line)
                logs = [l for l in new if not l.startswith("__")]
                time.sleep(0.15)
            thread.join()
            with lock:
                final = list(progress)
        final_lines = [l for l in final if l.startswith("__")]
        for l in final_lines:
            logs.append(l)

        # ---- Persist result ----
        done = [l for l in final_lines if l.startswith("__DONE__")]
        errs = [l for l in final_lines if l.startswith("__ERROR__")]
        if errs:
            st.session_state["train_error"] = errs[0][len("__ERROR__"):]
            st.session_state.pop("train_result", None)
        else:
            result = json.loads(done[0][len("__DONE__"):]) if done else {}
            st.session_state["train_result"] = result
            st.session_state.pop("train_error", None)
            # Activate the freshly trained family on the MAIN thread so only
            # this session's selection is updated (never a background thread).
            try:
                svc.select_model_family(result["family"])
            except Exception as exc:  # noqa: BLE001
                st.session_state["train_error"] = (
                    f"Trained {result['family']} but could not activate it: {exc}"
                )
                st.session_state.pop("train_result", None)
        st.session_state["train_logs"] = logs

    # ---- Show persistent logs ----
    if st.session_state.get("train_logs"):
        with st.expander("Training log", expanded=True):
            for line in st.session_state["train_logs"]:
                if line.startswith("__"):
                    continue
                st.code(line)

    # ---- Show result / error ----
    if st.session_state.get("train_error"):
        st.error(f"Training failed: {st.session_state['train_error']}")
        return

    result = st.session_state.get("train_result")
    if result:
        st.success(
            f"✅ Trained model family **{result['family']}** — OH+NPS from the same dataset."
        )

        m1, m2 = st.columns(2)
        m1.markdown("#### Operational Health")
        m1.metric("Algorithm", result.get("oh_algorithm") or "—")
        m1.metric("Features", result.get("oh_features"))
        _render_metrics(m1, result.get("oh_metrics"))

        m2.markdown("#### NPS")
        m2.metric("Algorithm", result.get("nps_algorithm") or "—")
        m2.metric("Features", result.get("nps_features"))
        _render_metrics(m2, result.get("nps_metrics"))

        st.caption(f"Saved: `{result['oh_path']}` + `{result['nps_path']}` "
                   f"· Trained at {result.get('trained_at', '')[:19]}")

        st.info("This family is now the **active** model family.")

        # ---- Analytics ----
        st.divider()
        df = _load_dataset_for_analytics(selected)
        if df is not None:
            from gui.analytics import training as _ta
            _ta.render_analytics(st, df, result)
        else:
            st.info("Training analytics unavailable — dataset could not be loaded.")


def _load_dataset_for_analytics(name: str):
    """Load a bounded sample for GUI analytics.

    IMPORTANT: Never cache the complete training dataset in Streamlit.
    Training itself remains responsible for loading the full dataset.
    """
    from gui import services as _svc
    files = {f.name: f for f in _svc.list_training_files()}
    path = files.get(name)
    if path is None:
        return None
    return ct.load_dataset_sample(path, n_rows=50_000)



def _render_metrics(container: Any, metrics: Dict[str, Any]) -> None:
    if not metrics:
        container.caption("No performance metrics reported.")
        return
    for k, v in metrics.items():
        if isinstance(v, dict):
            container.caption(f"**{k}**")
            for kk, vv in v.items():
                container.metric(str(kk), str(vv))
        else:
            container.metric(str(k), str(v))
