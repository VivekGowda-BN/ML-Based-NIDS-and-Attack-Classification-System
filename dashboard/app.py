"""
dashboard/app.py — Streamlit alert dashboard for the NIDS (Mode 1).

Features
--------
• Sidebar: scenario selector + replay controls.
• Main panel: live alert table (auto-refreshed every N seconds).
• Metric cards: total flows, attacks, top attack category, mean confidence.
• Confusion-matrix and SHAP images (if available in reports/).
• Alert detail expander with per-record SHAP waterfall (TODO).

Run
---
    streamlit run dashboard/app.py
    # or:
    make dashboard

Safety
------
  ✓ Reads only pre-computed result files.
  ✗ Does NOT touch live network interfaces.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="NIDS Dashboard — Mode 1",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
ALERTS_FILE   = ROOT / "reports" / "metrics" / "alerts.jsonl"
BINARY_METRICS = ROOT / "reports" / "metrics" / "binary_metrics.json"
MULTI_METRICS  = ROOT / "reports" / "metrics" / "multiclass_metrics.json"
FIGURES_DIR   = ROOT / "reports" / "figures"
EXPLANATIONS_DIR = ROOT / "reports" / "explanations"
SCENARIOS_DIR = ROOT / "data" / "scenarios"


# ─── Helper functions ─────────────────────────────────────────────────────────

def load_alerts(path: Path) -> pd.DataFrame:
    """Load the alerts JSONL file into a DataFrame."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_json_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def scenario_files() -> list[Path]:
    return sorted(SCENARIOS_DIR.glob("*.csv")) if SCENARIOS_DIR.exists() else []


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ NIDS Control Panel")
    st.caption("Mode 1 — Offline Replay")
    st.divider()

    # Scenario picker
    scenarios = scenario_files()
    scenario_labels = [s.name for s in scenarios] if scenarios else ["(no scenarios found)"]
    selected_scenario = st.selectbox("📂 Scenario file", scenario_labels)

    threshold = st.slider(
        "🎚 Attack probability threshold",
        min_value=0.1, max_value=0.99, value=0.50, step=0.01,
    )
    delay = st.number_input("⏱ Replay delay (s)", min_value=0.0, max_value=5.0, value=0.05, step=0.01)

    st.divider()
    run_replay = st.button("▶ Run Replay", type="primary", use_container_width=True)
    clear_alerts = st.button("🗑 Clear alerts", use_container_width=True)

    if clear_alerts and ALERTS_FILE.exists():
        ALERTS_FILE.write_text("")
        st.success("Alerts cleared.")

    st.divider()
    auto_refresh = st.toggle("🔄 Auto-refresh (5 s)", value=False)
    st.caption("Runs replay externally via `make replay`.")

# ─── Trigger replay (blocking — runs inline via subprocess) ───────────────────
if run_replay:
    if not scenarios:
        st.error("No scenario CSV files found in data/scenarios/.")
    else:
        scenario_path = SCENARIOS_DIR / selected_scenario
        with st.spinner(f"Replaying {selected_scenario} …"):
            try:
                import subprocess
                result = subprocess.run(
                    [
                        "python", "-m", "nids.replay",
                        "--scenario", str(scenario_path),
                        "--threshold", str(threshold),
                        "--delay", str(delay),
                    ],
                    capture_output=True, text=True, cwd=str(ROOT),
                )
                if result.returncode == 0:
                    st.success(result.stdout.strip() or "Replay complete.")
                else:
                    st.error(f"Replay error:\n{result.stderr}")
            except Exception as exc:
                st.error(f"Failed to run replay: {exc}")


# ─── Main panel ───────────────────────────────────────────────────────────────
st.title("🛡️ ML-Based Network Intrusion Detection System")
st.caption("Mode 1 — UNSW-NB15 Dataset  |  Binary + Multiclass Classification  |  Offline Replay")
st.divider()

# ── Model metrics row ─────────────────────────────────────────────────────────
bin_m = load_json_metrics(BINARY_METRICS)
mul_m = load_json_metrics(MULTI_METRICS)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Binary Accuracy",  f"{bin_m.get('accuracy',  '—')}")
m2.metric("Binary ROC-AUC",   f"{bin_m.get('roc_auc',   '—')}")
m3.metric("Binary F1 (macro)",f"{bin_m.get('f1_macro',  '—')}")
m4.metric("Multi  Accuracy",  f"{mul_m.get('accuracy',  '—')}")
m5.metric("Multi  F1 (macro)",f"{mul_m.get('f1_macro',  '—')}")

st.divider()

# ── Alert table ───────────────────────────────────────────────────────────────
st.subheader("🚨 Alert Feed")

alerts_df = load_alerts(ALERTS_FILE)

if alerts_df.empty:
    st.info(
        "No alerts yet.  \n"
        "1. Download the UNSW-NB15 dataset (see data/README.md).  \n"
        "2. Run the pipeline: `make data-prep preprocess train evaluate`.  \n"
        "3. Place a scenario CSV in `data/scenarios/` and click **▶ Run Replay**."
    )
else:
    # Summary metrics
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total alerts",    len(alerts_df))
    top_cat = (
        alerts_df["attack_category"].value_counts().idxmax()
        if "attack_category" in alerts_df.columns and not alerts_df["attack_category"].isna().all()
        else "—"
    )
    a2.metric("Top attack type", top_cat)
    a3.metric("Avg confidence",  f"{alerts_df['confidence'].mean():.2%}" if "confidence" in alerts_df.columns else "—")
    a4.metric("Unique src IPs",  alerts_df["src_ip"].nunique() if "src_ip" in alerts_df.columns else "—")

    st.dataframe(alerts_df, use_container_width=True, hide_index=True)

    # Attack category breakdown
    if "attack_category" in alerts_df.columns:
        st.subheader("📊 Attack Category Breakdown")
        cat_counts = alerts_df["attack_category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig = px.bar(cat_counts, x="Category", y="Count", color="Category",
                     color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Confusion matrices ────────────────────────────────────────────────────────
st.subheader("🔢 Confusion Matrices")
col_a, col_b = st.columns(2)
bin_cm = FIGURES_DIR / "binary_confusion_matrix.png"
mul_cm = FIGURES_DIR / "multiclass_confusion_matrix.png"

with col_a:
    if bin_cm.exists():
        st.image(str(bin_cm), caption="Binary Classifier", use_container_width=True)
    else:
        st.info("Run `make evaluate` to generate the binary confusion matrix.")

with col_b:
    if mul_cm.exists():
        st.image(str(mul_cm), caption="Multiclass Classifier", use_container_width=True)
    else:
        st.info("Run `make evaluate` to generate the multiclass confusion matrix.")

st.divider()

# ── SHAP feature importance ───────────────────────────────────────────────────
st.subheader("🔍 SHAP Feature Importance")
shap_json = EXPLANATIONS_DIR / "binary_shap_importance.json"
if shap_json.exists():
    importance = json.loads(shap_json.read_text())
    shap_df = pd.DataFrame(
        list(importance.items()), columns=["Feature", "Mean |SHAP|"]
    ).head(20)
    fig2 = px.bar(
        shap_df, x="Mean |SHAP|", y="Feature", orientation="h",
        color="Mean |SHAP|", color_continuous_scale="Blues",
        title="Top 20 Features — Binary Classifier",
    )
    fig2.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Run `make explain` to generate SHAP feature importance.")

# ─── Auto-refresh ─────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(5)
    st.rerun()
