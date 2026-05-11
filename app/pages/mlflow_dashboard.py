"""
app/pages/mlflow_dashboard.py
==============================
Browse all MLflow experiments, runs, and metrics.
"""

from __future__ import annotations

import mlflow
import pandas as pd
import plotly.express as px
import streamlit as st
from mlflow.tracking import MlflowClient


def _get_metric(run, possible_keys: list[str], default: float = 0.0) -> float:
    """Safely extracts a metric by checking multiple possible key names."""
    for key in possible_keys:
        if key in run.data.metrics:
            return run.data.metrics[key]
    return default


def render(mlflow_uri: str) -> None:
    st.header("🔬 MLflow Experiment Dashboard")
    st.caption(f"Tracking URI: `{mlflow_uri}`  —  [Open full MLflow UI →](http://localhost:5000)")

    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()

    # ── Experiments ───────────────────────────────────────────────────────
    experiments = [e for e in client.search_experiments() if e.name != "Default"]

    if not experiments:
        st.info(
            "No experiments found yet.\n\n"
            "Start training to populate MLflow:\n```\nmake train-yolo\n```"
        )
        return

    exp_names = [e.name for e in experiments]
    selected_exp = st.selectbox("Select experiment", exp_names)
    exp = next(e for e in experiments if e.name == selected_exp)

    # ── Runs table ────────────────────────────────────────────────────────
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=100,
    )

    if not runs:
        st.info(f"No runs in experiment '{selected_exp}' yet.")
        return

    rows = []
    for r in runs:
        # Robust metric extraction using the helper function
        map50 = _get_metric(r, ["metrics/mAP50(B)", "val/mAP50", "test/mAP50"])
        map50_95 = _get_metric(r, ["metrics/mAP50-95(B)", "val/mAP50_95", "test/mAP50_95"])
        precision = _get_metric(r, ["metrics/precision(B)", "val/precision", "test/precision"])
        recall = _get_metric(r, ["metrics/recall(B)", "val/recall", "test/recall"])

        rows.append(
            {
                "Run name": r.info.run_name or r.info.run_id[:8],
                "Status": r.info.status,
                "Architecture": r.data.params.get("architecture", "?"),
                "HP combo": r.data.params.get("hp_combination", "?"),
                "Epochs": r.data.params.get("epochs", "?"),
                "LR": r.data.params.get("lr", "?"),
                "Batch": r.data.params.get("batch", "?"),
                "Optimizer": r.data.params.get("optimizer", "?"),
                "val/mAP50": round(map50, 4),
                "val/mAP50-95": round(map50_95, 4),
                "val/precision": round(precision, 4),
                "val/recall": round(recall, 4),
                "run_id": r.info.run_id,
            }
        )

    df = pd.DataFrame(rows)

    st.markdown("---")
    st.subheader(f"Runs in '{selected_exp}'")

    # Highlight best run
    best_idx = df["val/mAP50"].idxmax() if "val/mAP50" in df else None

    st.dataframe(
        df.drop(columns=["run_id"]),
        use_container_width=True,
        height=300,
    )

    if best_idx is not None:
        best = df.loc[best_idx]
        # Only show the best run if it actually has data (greater than 0)
        if best["val/mAP50"] > 0.0:
            st.success(
                f"🏆 Best run: **{best['Run name']}** "
                f"({best['Architecture']} {best['HP combo']}) — "
                f"mAP@50 = **{best['val/mAP50']:.4f}**"
            )

    # ── HP sweep scatter plot ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("HP sweep — mAP@50 by configuration")
    if len(df) >= 2:
        fig = px.scatter(
            df,
            x="HP combo",
            y="val/mAP50",
            color="Architecture",
            symbol="Optimizer",
            size_max=15,
            hover_data=["LR", "Batch", "Epochs", "val/mAP50-95"],
            title="Validation mAP@50 per HP Combination",
            height=380,
        )
        fig.update_traces(marker_size=12)
        st.plotly_chart(fig, use_container_width=True)

    # ── Architecture comparison bar chart ─────────────────────────────────
    if "Architecture" in df.columns and df["Architecture"].nunique() > 1:
        st.markdown("---")
        st.subheader("Architecture comparison — best result per arch")
        best_per_arch = df.groupby("Architecture")["val/mAP50"].max().reset_index()
        fig2 = px.bar(
            best_per_arch,
            x="Architecture",
            y="val/mAP50",
            color="Architecture",
            title="Best val/mAP50 per Architecture",
            height=320,
            color_discrete_sequence=["#2d6a4f", "#e05c00", "#1565c0"],
        )
        fig2.update_layout(showlegend=False)
        fig2.update_traces(texttemplate="%{y:.4f}", textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Run details expander ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Inspect a specific run")
    run_names = df["Run name"].tolist()
    sel_run = st.selectbox("Select run to inspect", run_names)
    sel_row = df[df["Run name"] == sel_run].iloc[0]

    with st.expander("All parameters", expanded=False):
        r = client.get_run(sel_row["run_id"])
        params_df = pd.DataFrame(
            [{"Parameter": k, "Value": v} for k, v in sorted(r.data.params.items())]
        )
        st.dataframe(params_df, use_container_width=True)

    with st.expander("All metrics", expanded=False):
        r = client.get_run(sel_row["run_id"])
        metrics_df = pd.DataFrame(
            [{"Metric": k, "Value": round(v, 6)} for k, v in sorted(r.data.metrics.items())]
        )
        st.dataframe(metrics_df, use_container_width=True)

    with st.expander("Artifacts", expanded=False):
        try:
            artifacts = client.list_artifacts(sel_row["run_id"])
            if artifacts:
                for a in artifacts:
                    st.markdown(f"- `{a.path}` ({a.file_size or '?'} bytes)")
            else:
                st.info("No artifacts logged for this run.")
        except Exception as e:
            st.warning(f"Could not list artifacts: {e}")
