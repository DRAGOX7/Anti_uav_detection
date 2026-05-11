"""
app/pages/model_comparison.py
==============================
Side-by-side MLflow run comparison for the report.
"""

from __future__ import annotations

import mlflow
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from mlflow.tracking import MlflowClient


def render(mlflow_uri: str) -> None:
    st.header("📊 Model Comparison")
    st.caption("Select two MLflow runs to compare architectures and HP configurations side-by-side.")

    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()

    # ── Load all experiments ──────────────────────────────────────────────
    experiments = client.search_experiments()
    if not experiments:
        st.warning(
            "No MLflow experiments found.\n\n"
            "Train at least one model first:\n```\nmake train-yolo\n```"
        )
        return

    # Flatten all runs across all experiments
    all_runs = []
    for exp in experiments:
        if exp.name.startswith("data-preparation"):
            continue
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["metrics.`val/mAP50` DESC"],
            max_results=50,
        )
        for r in runs:
            all_runs.append({
                "run_id":   r.info.run_id[:8],
                "full_id":  r.info.run_id,
                "name":     r.info.run_name or r.info.run_id[:8],
                "arch":     r.data.params.get("architecture", "?"),
                "combo":    r.data.params.get("hp_combination", "?"),
                "mAP50":    r.data.metrics.get("val/mAP50",    r.data.metrics.get("metrics/mAP50(B)", 0)),
                "mAP5095":  r.data.metrics.get("val/mAP50_95", 0),
                "prec":     r.data.metrics.get("val/precision", 0),
                "recall":   r.data.metrics.get("val/recall",    0),
                "experiment": exp.name,
            })

    if not all_runs:
        st.warning("No finished runs found. Complete at least one training run.")
        return

    df_runs = pd.DataFrame(all_runs)
    run_labels = [
        f"{r['arch']} {r['combo']} — mAP50={r['mAP50']:.3f} ({r['name']})"
        for _, r in df_runs.iterrows()
    ]

    # ── Run selectors ─────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Run A")
        sel_a = st.selectbox("Select run A", run_labels, key="run_a")
    with col2:
        st.markdown("#### Run B")
        default_b = 1 if len(run_labels) > 1 else 0
        sel_b = st.selectbox("Select run B", run_labels, index=default_b, key="run_b")

    if sel_a == sel_b:
        st.warning("Select two different runs for a meaningful comparison.")
        return

    run_a = df_runs.iloc[run_labels.index(sel_a)]
    run_b = df_runs.iloc[run_labels.index(sel_b)]

    st.markdown("---")

    # ── Metric cards ──────────────────────────────────────────────────────
    metrics_to_compare = [
        ("mAP@50",       "mAP50"),
        ("mAP@50-95",    "mAP5095"),
        ("Precision",    "prec"),
        ("Recall",       "recall"),
    ]

    st.subheader("Metric comparison")
    cols = st.columns(len(metrics_to_compare))
    for col, (label, key) in zip(cols, metrics_to_compare):
        val_a = run_a[key] if key in run_a else 0.0
        val_b = run_b[key] if key in run_b else 0.0
        delta = val_a - val_b
        with col:
            st.markdown(f"**{label}**")
            c1, c2 = st.columns(2)
            c1.metric(run_a["arch"] + " " + run_a["combo"],
                      f"{val_a:.4f}", delta=f"{delta:+.4f}")
            c2.metric(run_b["arch"] + " " + run_b["combo"],
                      f"{val_b:.4f}", delta=f"{-delta:+.4f}")

    st.markdown("---")

    # ── Radar chart ──────────────────────────────────────────────────────
    st.subheader("Radar chart — multi-metric comparison")
    categories = ["mAP@50", "mAP@50-95", "Precision", "Recall"]
    vals_a = [run_a["mAP50"], run_a["mAP5095"], run_a["prec"], run_a["recall"]]
    vals_b = [run_b["mAP50"], run_b["mAP5095"], run_b["prec"], run_b["recall"]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_a + [vals_a[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name=f"{run_a['arch']} {run_a['combo']}",
        line_color="#2d6a4f",
        fillcolor="rgba(45,106,79,0.2)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals_b + [vals_b[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name=f"{run_b['arch']} {run_b['combo']}",
        line_color="#e05c00",
        fillcolor="rgba(224,92,0,0.2)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=400,
        margin=dict(l=60, r=60, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Learning curves from MLflow (if metric history available) ─────────
    st.markdown("---")
    st.subheader("Learning curves — val/mAP50 over epochs")

    def _get_metric_history(run_id: str, metric: str):
        try:
            history = client.get_metric_history(run_id, metric)
            return [(h.step, h.value) for h in history]
        except Exception:
            return []

    hist_a = _get_metric_history(run_a["full_id"], "val/mAP50")
    hist_b = _get_metric_history(run_b["full_id"], "val/mAP50")

    if hist_a or hist_b:
        fig2 = go.Figure()
        if hist_a:
            steps, vals = zip(*hist_a)
            fig2.add_trace(go.Scatter(
                x=list(steps), y=list(vals),
                name=f"{run_a['arch']} {run_a['combo']}",
                line=dict(color="#2d6a4f", width=2),
            ))
        if hist_b:
            steps, vals = zip(*hist_b)
            fig2.add_trace(go.Scatter(
                x=list(steps), y=list(vals),
                name=f"{run_b['arch']} {run_b['combo']}",
                line=dict(color="#e05c00", width=2),
            ))
        fig2.update_layout(
            xaxis_title="Epoch",
            yaxis_title="val/mAP50",
            height=350,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info(
            "Epoch-level metric history not available for these runs.\n"
            "Ensure mlflow_callbacks are attached during training."
        )

    # ── Parameter diff table ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Hyperparameter differences")
    try:
        r_a = client.get_run(run_a["full_id"])
        r_b = client.get_run(run_b["full_id"])
        params_a = r_a.data.params
        params_b = r_b.data.params
        all_keys = set(params_a) | set(params_b)
        diff_rows = []
        for k in sorted(all_keys):
            va = params_a.get(k, "—")
            vb = params_b.get(k, "—")
            if va != vb:
                diff_rows.append({"Parameter": k, "Run A": va, "Run B": vb})
        if diff_rows:
            st.dataframe(pd.DataFrame(diff_rows), use_container_width=True)
        else:
            st.info("Runs have identical hyperparameters.")
    except Exception as e:
        st.warning(f"Could not load parameter details: {e}")
